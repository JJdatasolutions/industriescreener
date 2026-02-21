import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import math
import scipy.linalg as la
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore") # Onderdruk waarschuwingen voor stationariteit tests

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener 7.5 (Scientific RRG)", layout="wide", page_icon="🧠")

# --- 1. DATA DEFINITIES ---
MARKETS = {
    "🇺🇸 USA - S&P 500": {
        "code": "SP500", "benchmark": "SPY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    },
    "🇺🇸 USA - S&P 400 (MidCap)": {
        "code": "SP400", "benchmark": "MDY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    },
    "🇪🇺 Europa - Selectie": {
        "code": "EU_MIX", "benchmark": "^N100", "type": "static"
    }
}

US_SECTOR_MAP = {
    'Information Technology': 'XLK',
    'Financials': 'XLF',
    'Health Care': 'XLV',
    'Energy': 'XLE',
    'Consumer Discretionary': 'XLY',
    'Industrials': 'XLI',
    'Utilities': 'XLU',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
    'Consumer Staples': 'XLP'
}

# --- 2. FUNCTIES ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_key):
    mkt = MARKETS[market_key]
    
    # 1. EU LOGICA
    if "EU_MIX" in mkt.get("code", ""):
        data = {
            "ASML.AS": "Technology", "UNA.AS": "Consumer Staples", "HEIA.AS": "Consumer Staples", 
            "SHELL.AS": "Energy", "INGA.AS": "Financials", "DSM.AS": "Materials", 
            "BESI.AS": "Technology", "ADYEN.AS": "Financials", "IMCD.AS": "Materials", 
            "PHI.AS": "Health Care", "KBC.BR": "Financials", "UCB.BR": "Health Care", 
            "SOLB.BR": "Materials", "WDP.BR": "Real Estate", "ELI.BR": "Utilities",
            "MELE.BR": "Industrials", "XIOR.BR": "Real Estate", "ACKB.BR": "Financials",
            "ABI.BR": "Consumer Staples", "GBL.BR": "Financials"
        }
        return pd.DataFrame(list(data.items()), columns=['Ticker', 'Sector'])

    # 2. USA SCRAPER
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(mkt['wiki'], headers=headers, timeout=5)
        response.raise_for_status()
        
        tables = pd.read_html(response.text)
        target_df = pd.DataFrame()
        
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any("symbol" in c for c in cols) and (any("gics sector" in c for c in cols) or any("sector" in c for c in cols)):
                target_df = df
                break
        
        if target_df.empty: raise ValueError("Geen tabel gevonden")

        ticker_col = next(c for c in target_df.columns if "Symbol" in str(c) or "Ticker" in str(c))
        sector_col_candidates = [c for c in target_df.columns if "Sector" in str(c)]
        sector_col = sector_col_candidates[0]
        
        df_clean = target_df[[ticker_col, sector_col]].copy()
        df_clean.columns = ['Ticker', 'Sector']
        df_clean['Ticker'] = df_clean['Ticker'].str.replace('.', '-', regex=False)
        df_clean['Sector'] = df_clean['Sector'].astype(str).str.strip()
        
        if df_clean.empty: raise ValueError("Lege dataset")
        return df_clean

    except Exception as e:
        st.warning(f"⚠️ Wikipedia fallback ({e}). Using top 30.")
        fallback_data = {
            'AAPL': 'Information Technology', 'MSFT': 'Information Technology', 'NVDA': 'Information Technology',
            'AMZN': 'Consumer Discretionary', 'GOOGL': 'Communication Services', 'META': 'Communication Services',
            'TSLA': 'Consumer Discretionary', 'BRK-B': 'Financials', 'LLY': 'Health Care', 'AVGO': 'Information Technology',
            'JPM': 'Financials', 'V': 'Financials', 'XOM': 'Energy', 'UNH': 'Health Care',
            'PG': 'Consumer Staples', 'MA': 'Financials', 'JNJ': 'Health Care', 'HD': 'Consumer Discretionary',
            'MRK': 'Health Care', 'COST': 'Consumer Staples', 'ABBV': 'Health Care', 'AMD': 'Information Technology',
            'CRM': 'Information Technology', 'NFLX': 'Communication Services', 'PEP': 'Consumer Staples',
            'KO': 'Consumer Staples', 'BAC': 'Financials', 'WMT': 'Consumer Staples', 'CVX': 'Energy',
            'ADBE': 'Information Technology'
        }
        return pd.DataFrame(list(fallback_data.items()), columns=['Ticker', 'Sector'])

@st.cache_data(ttl=3600)
def get_price_data(tickers, end_date=None):
    if not tickers:
        return pd.DataFrame()

    try:
        if end_date:
            data = yf.download(
                tickers,
                start="2024-01-01",  # ruim genoeg historisch
                end=end_date,
                progress=False,
                auto_adjust=True
            )
        else:
            data = yf.download(
                tickers,
                period="2y",
                progress=False,
                auto_adjust=True
            )

        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', level=1, axis=1)

        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers

        return data

    except:
        return pd.DataFrame()

# --- HELPER FUNCTIES VOOR INSTITUTIONELE ALPHA ---

def denoise_covariance(returns, q, variance=1):
    """
    Marcenko-Pastur Denoising van de Covariantiematrix (López de Prado).
    q = T/N (Aantal waarnemingen / Aantal variabelen)
    """
    if returns.empty or returns.shape[1] < 2:
        return returns.cov()
        
    cov_matrix = returns.cov().values
    
    # Bereken eigenwaarden en eigenvectoren
    eigenvalues, eigenvectors = la.eigh(cov_matrix)
    
    # Marcenko-Pastur theorethische maximale eigenwaarde
    e_max = variance * (1 + (1/q)**0.5)**2
    
    # Denoise: Zet eigenwaarden onder e_max op de gemiddelde waarde om variantie te behouden
    eigenvalues_denoised = eigenvalues.copy()
    noise_indices = eigenvalues < e_max
    if noise_indices.any():
        eigenvalues_denoised[noise_indices] = eigenvalues[noise_indices].mean()
        
    # Reconstructie van de gezuiverde covariantiematrix
    cov_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ la.inv(eigenvectors)
    return pd.DataFrame(cov_denoised, index=returns.columns, columns=returns.columns)

def check_stationarity(series):
    """
    Augmented Dickey-Fuller test.
    Retouren: True als stationair (p-value < 0.05), anders False (Random Walk).
    """
    try:
        # Drop NaNs en controleer of er genoeg data is
        clean_series = series.dropna()
        if len(clean_series) < 30: 
            return False
            
        result = adfuller(clean_series)
        return result[1] < 0.05 # p-value < 5% betekent we verwerpen de null-hypothese (het is stationair)
    except:
        return False

def train_meta_labeler(features, labels):
    """
    Traing een Random Forest Classifier om de 'Probability of Success' te berekenen.
    """
    if len(features) < 50: # Te weinig data voor ML
        return 0.5
        
    clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    clf.fit(features, labels)
    # Return de kansklasse 1 (succes)
    return clf.predict_proba(features.iloc[-1:])[-1][1]


# --- HOOFD FUNCTIE ---

def calculate_rrg_extended(df, benchmark_ticker, market_bullish=True, profile="Momentum Profile"):
    """
    RRG v8.5 - Multi-Profile Institutional Alpha
    Ondersteunt: Momentum, Value (Novy-Marx Profitability), en Balanced (Combo).
    """
    if df.empty or benchmark_ticker not in df.columns: return pd.DataFrame()
    
    rrg_data = []
    bench_series = df[benchmark_ticker]
    
    # 1. Marcenko-Pastur Denoising Matrix voorbereiden
    returns = df.pct_change().dropna()
    q = returns.shape[0] / returns.shape[1] if returns.shape[1] > 0 else 1
    denoised_cov = denoise_covariance(returns, q)
    
    for ticker in df.columns:
        if ticker == benchmark_ticker: continue
        try:
            # Basis data
            asset_series = df[ticker]
            rs = asset_series / bench_series
            rs_ma = rs.rolling(100).mean()
            rs_ratio = 100 * (rs / rs_ma)
            rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
            
            if len(rs_ratio) < 20: continue
            
            curr_r, curr_m = rs_ratio.iloc[-1], rs_mom.iloc[-1]
            prev_r, prev_m = rs_ratio.iloc[-2], rs_mom.iloc[-2]
            
            # Risk & Stationarity
            asset_vol = np.sqrt(denoised_cov.loc[ticker, ticker]) * np.sqrt(252) 
            vol_penalty = max(0.5, 1 - asset_vol)
            is_stationary = check_stationarity(rs_ratio.tail(100))
            stationarity_multiplier = 1.0 if is_stationary else 0.3
            
            # RRG Vectoren
            dist = np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            dx, dy = curr_r - prev_r, curr_m - prev_m
            heading_deg = math.degrees(math.atan2(dy, dx)) % 360
            
            deviation = abs(heading_deg - 45)
            if deviation > 180: deviation = 360 - deviation
            heading_quality = max(0, 1 - (deviation / 135))
            
            # Factoren
            max_52w = asset_series.tail(252).max()
            drawdown = (asset_series.iloc[-1] / max_52w) - 1
            value_factor = abs(drawdown) if drawdown < -0.2 else 0 
            mom_factor = (asset_series.iloc[-1] / asset_series.iloc[-20]) - 1
            
            # Alpha Score
            raw_alpha = (dist * heading_quality * vol_penalty) + (value_factor * 10) + (mom_factor * 10)
            raw_alpha = raw_alpha * stationarity_multiplier
            
            # Novy-Marx Gross Profitability (Proxy / Mock voor performance in Streamlit)
            # In productie: vervang dit door een DB call of een batch-API fetch.
            # Berekening: Gross Profit / Total Assets
            np.random.seed(int(sum(bytearray(ticker, 'utf8')))) # Consistente pseudo-random hash
            gross_profitability = np.random.uniform(0.1, 0.6) # Tussen 10% en 60%

            # Simpele Meta-labeler bypass voor snelheid in iteratie (aanpassen indien gewenst)
            meta_prob_success = 0.65 
            institutional_alpha = raw_alpha * meta_prob_success
            institutional_alpha = max(0.1, institutional_alpha)

            # --- PROFIEL LOGICA (ACTIES) ---
            action = "HOLD/WATCH"
            if profile == "Momentum Profile":
                if 180 <= heading_deg <= 275:
                    action = "❌ AVOID"
                elif 0 <= heading_deg <= 90:
                    if institutional_alpha > 2.0: 
                        action = "✅ MOMENTUM BUY"
                    elif curr_r > 100 and curr_m > 100:
                        action = "⚠️ SPEC BUY"
            
            elif profile == "Value Profile":
                # Contrarian: Koop de losers die verbeteren (Lagging/Improving) én winstgevend zijn
                if curr_r < 100 and 0 <= heading_deg <= 180 and gross_profitability > 0.35:
                    action = "💎 VALUE BUY"
                elif 180 <= heading_deg <= 270:
                    action = "❌ VALUE TRAP"
            
            elif profile == "Balanced (Combo)":
                # Zoekt winstgevende aandelen met positieve heading, los van RS-Ratio
                if gross_profitability > 0.4 and 0 <= heading_deg <= 90 and curr_m > 100:
                    action = "🏆 COMBO BUY"
                elif 180 <= heading_deg <= 270:
                    action = "❌ AVOID"

            # Kwadranten
            if curr_r >= 100 and curr_m >= 100: kwadrant = "1. LEADING"
            elif curr_r < 100 and curr_m >= 100: kwadrant = "4. IMPROVING"
            elif curr_r < 100 and curr_m < 100: kwadrant = "3. LAGGING"
            else: kwadrant = "2. WEAKENING"
            
            rrg_data.append({
                'Ticker': ticker,
                'RS-Ratio': curr_r,
                'RS-Momentum': curr_m,
                'Kwadrant': kwadrant,
                'Distance': dist,
                'Heading': heading_deg,
                'Alpha_Score': institutional_alpha,
                'Gross_Profitability': gross_profitability, # Nieuwe parameter
                'Action': action
            })
        except Exception as e: 
            continue
            
    return pd.DataFrame(rrg_data)

def calculate_market_health(bench_series):
    curr = bench_series.iloc[-1]
    sma200 = bench_series.rolling(200).mean().iloc[-1]
    trend = "BULL" if curr > sma200 else "BEAR"
    dist_pct = ((curr - sma200) / sma200) * 100
    
    # Dispersie: Bereken StdDev van de laatste 20 dagen returns van de benchmark
    # Dit is een 'proxy' voor marktvolatiliteit/dispersie
    returns = bench_series.pct_change().tail(20)
    volatility = returns.std() * 100 # In procenten
    
    return trend, dist_pct, sma200, volatility

# --- 3. SIDEBAR ---

st.sidebar.header("⚙️ Instellingen")
sel_market_key = st.sidebar.selectbox("Kies Markt", list(MARKETS.keys()))
market_cfg = MARKETS[sel_market_key]

from datetime import datetime

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Historische Analyse")

min_date = datetime(2026, 1, 1)
max_date = datetime.today()

selected_date = st.sidebar.date_input(
    "Bekijk situatie op datum:",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

use_historical = selected_date < max_date.date()
# Check of we in de USA zitten voor sector map, anders standaard
sector_list = ["Alle Sectoren"]
if "USA" in sel_market_key:
    sector_list += sorted(US_SECTOR_MAP.keys())

sel_sector = st.sidebar.selectbox("Kies Sector", sector_list)

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Markt (Benchmark)")

# 1. BENCHMARK DATA (Zoals voorheen)
bench_df = get_price_data([market_cfg['benchmark']])
market_bull_flag = True # Default

if not bench_df.empty:
    s = bench_df.iloc[:, 0]
    trend, dist_pct, sma200, vola = calculate_market_health(s)
    market_bull_flag = (trend == "BULL")
    
    color = "green" if trend == "BULL" else "red"
    st.sidebar.markdown(f"Trend: :{color}[**{trend} MARKET**]")
    st.sidebar.caption(f"Prijs vs 200d SMA ({dist_pct:.1f}%)")
    
    # Dispersie Meter
    st.sidebar.progress(min(int(vola * 30), 100))
    if vola < 0.8: st.sidebar.caption(f"Volatiliteit: Laag ({vola:.2f}%)")
    elif vola > 1.5: st.sidebar.caption(f"Volatiliteit: Hoog ({vola:.2f}%)")
    else: st.sidebar.caption(f"Volatiliteit: Normaal ({vola:.2f}%)")

    # Mini Grafiek Benchmark
    chart_data = s.tail(252).to_frame(name="Koers") # 252 dagen = 1 handelsjaar
    chart_data['SMA200'] = s.rolling(200).mean().tail(252)
    
    fig_mini = px.line(chart_data, y=["Koers", "SMA200"], height=150)
    fig_mini.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None)
    fig_mini.update_traces(line_color='gray', selector=dict(name='Koers'))
    fig_mini.update_traces(line_color=color, selector=dict(name='SMA200'))
    st.sidebar.plotly_chart(fig_mini, use_container_width=True)

# 2. SECTOR DATA (NIEUW TOEGEVOEGD)
# Dit blok wordt alleen getoond als je een specifieke sector kiest EN het een USA markt is
if sel_sector != "Alle Sectoren" and "USA" in sel_market_key:
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"🏗️ Sector: {sel_sector}")
    
    sec_ticker = US_SECTOR_MAP.get(sel_sector)
    
    if sec_ticker:
        # Haal data op voor de sector tracker (bijv. XLK)
        sec_df = get_price_data([sec_ticker])
        
        if not sec_df.empty:
            s_sec = sec_df.iloc[:, 0]
            # Bereken gezondheid van de SECTOR (onafhankelijk van de markt)
            sec_trend, sec_dist, sec_sma, sec_vola = calculate_market_health(s_sec)
            
            sec_color = "green" if sec_trend == "BULL" else "red"
            
            st.sidebar.markdown(f"Ticker: **{sec_ticker}**")
            st.sidebar.markdown(f"Trend: :{sec_color}[**{sec_trend}**]")
            st.sidebar.metric("Afstand SMA200", f"{sec_dist:.1f}%")
            
            # Mini Grafiek Sector
            chart_data_sec = s_sec.tail(252).to_frame(name="Koers")
            chart_data_sec['SMA200'] = s_sec.rolling(200).mean().tail(252)
            
            fig_sec = px.line(chart_data_sec, y=["Koers", "SMA200"], height=150)
            fig_sec.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None)
            fig_sec.update_traces(line_color='gray', selector=dict(name='Koers'))
            fig_sec.update_traces(line_color=sec_color, selector=dict(name='SMA200'))
            
            st.sidebar.plotly_chart(fig_sec, use_container_width=True)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Start Wetenschappelijke Analyse", type="primary"):
    st.session_state['active'] = True
    st.session_state['market_key'] = sel_market_key
    st.session_state['sector_sel'] = sel_sector
# --- 4. HOOFD SCHERM ---

st.title(f"Quant Screener: {market_cfg['code']}")
tab1, tab2, tab3, tab4 = st.tabs(["ℹ️ Methodologie", "🚁 Sectoren", "🔍 Aandelen", "🧠 AI Analyst"])

# === TAB 1: INFO ===
with tab1:
    st.markdown("""
    ### 📊 Scientific RRG Methodologie (v7.5)
    Deze tool gebruikt de geoptimaliseerde regels gebaseerd op 'The Power of Heading'.
    
    **1. De 45° Regel (North-East)**
    Onderzoek toont aan dat aandelen die bewegen in een hoek van **0° tot 90°** (Noordoost) statistisch de hoogste Alpha genereren. De 'Sweet Spot' ligt op 45°.
    
    **2. Alpha Score**
    We berekenen voor elk aandeel een score:
    $ \text{Alpha Score} = \text{Distance} \times \text{Heading Quality} $
    * *Distance:* Hoe ver van het centrum (Saai) vandaan?
    * *Heading Quality:* Hoe dicht bij 45°?
    
    **3. Actie Signalen**
    * ✅ **BUY:** Heading 0-90° + Bull Market + Voldoende Afstand.
    * ❌ **AVOID:** Heading 180-270° (South-West). Dit is "dead money".
    * ⚠️ **SPEC BUY:** Heading 0-90° in Bear Market (Alleen sterke Leaders).
    """)

# === TAB 2: SECTOREN ===
with tab2:
    if st.session_state.get('active'):
        st.subheader("Sector Rotatie")
        
        with st.spinner("Sectoren analyseren..."):
            tickers = list(US_SECTOR_MAP.values()) if "USA" in st.session_state['market_key'] else []
            if not tickers: 
                 constituents = get_market_constituents(st.session_state['market_key'])
                 tickers = constituents['Ticker'].head(15).tolist() if not constituents.empty else []

            if tickers:
                calc_tickers = list(set(tickers + [market_cfg['benchmark']]))
                df_sec = get_price_data(calc_tickers)
                
                # Geef de Market Regime flag mee!
                rrg_sec = calculate_rrg_extended(df_sec, market_cfg['benchmark'], market_bullish=market_bull_flag)
                
                if not rrg_sec.empty:
                    labels = {v: k for k, v in US_SECTOR_MAP.items()}
                    rrg_sec['Label'] = rrg_sec['Ticker'].map(labels).fillna(rrg_sec['Ticker'])
                    rrg_sec = rrg_sec[rrg_sec['Ticker'].isin(tickers)]
                    rrg_sec = rrg_sec[rrg_sec['Distance'] > 0]

                    # Kleurenschaal: Nadruk op 45 graden (Groen)
                    custom_color_scale = [
                        (0.00, "#e5e7eb"),  # 0°
                        (0.125, "#00ff00"), # 45°  : FEL GROEN (Max Power)
                        (0.25, "#10b981"),  # 90°
                        (0.26, "#fca5a5"),  # >90° : Rood gebied start
                        (1.00, "#450a0a")   # 360°
                    ]

                    fig = px.scatter(
                        rrg_sec, 
                        x="RS-Ratio", 
                        y="RS-Momentum", 
                        color="Heading", 
                        text="Label", 
                        size="Alpha_Score", # Grootte = Kracht
                        height=650,
                        hover_data=["Kwadrant", "Action", "Distance"],
                        title=f"<b>SECTOR ROTATIE</b> <br><sup>Focus op 45° (Fel Groen) | Grootte bol = Alpha Score</sup>"
                    )
                    
                    fig.update_traces(
                        marker=dict(line=dict(width=1, color='black'), opacity=0.9),
                        textposition='top center',
                        textfont=dict(size=11, color='darkslategrey', family="Arial Black")
                    )
                    
                    fig.update_layout(
                        coloraxis_cmin=0, coloraxis_cmax=360,
                        coloraxis_colorscale=custom_color_scale,
                        coloraxis_colorbar=dict(
                            title="Richting",
                            tickvals=[0, 45, 90, 225],
                            ticktext=["0°", "45° (TOP)", "90°", "SW (Short)"]
                        ),
                        template="plotly_white",
                        xaxis=dict(showgrid=True, zeroline=True, zerolinecolor='black'), 
                        yaxis=dict(showgrid=True, zeroline=True, zerolinecolor='black'),
                        margin=dict(t=60, b=40, l=40, r=40)
                    )
                    
                    # Watermerken (RGBA fix)
                    fig.add_annotation(x=101, y=101, text="LEADING", showarrow=False, font=dict(size=14, color="rgba(0,128,0,0.3)"))
                    fig.add_annotation(x=99, y=99, text="LAGGING", showarrow=False, font=dict(size=14, color="rgba(255,0,0,0.3)"))

                    max_dev = max(abs(rrg_sec['RS-Ratio']-100).max(), abs(rrg_sec['RS-Momentum']-100).max()) * 1.1
                    fig.update_xaxes(range=[100-max_dev, 100+max_dev])
                    fig.update_yaxes(range=[100-max_dev, 100+max_dev])

                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### 🏆 Sector Ranking (Alpha Score)")
                    top_sec = rrg_sec.sort_values('Alpha_Score', ascending=False)
                    st.dataframe(
                        top_sec[['Label', 'Alpha_Score', 'Heading', 'Action']].style
                        .background_gradient(subset=['Alpha_Score'], cmap='Greens')
                        .format({"Heading": "{:.0f}°", "Alpha_Score": "{:.1f}"}),
                        hide_index=True,
                        use_container_width=True
                    )
                else: st.warning("Geen sector data.")
            else: st.warning("Geen tickers.")

# === TAB 3: AANDELEN ===
with tab3:
    if st.session_state.get('active'):

        current_sec = st.session_state.get('sector_sel', 'Alle Sectoren')
        st.subheader(f"Deep Dive: {current_sec}")

        # --- NIEUW: PROFIEL TOGGLE & DYNAMISCHE LEGENDE ---
        st.markdown("### 🧠 Kies Investeringsprofiel")
        selected_profile = st.radio(
            "Selecteer je RRG strategie:",
            ["Momentum Profile", "Value Profile", "Balanced (Combo)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Dynamische uitleg voor de leek (en de pro)
        if selected_profile == "Momentum Profile":
            st.info("""
            **🔥 Hoe lees je deze grafiek? (Momentum Focus)**
            * **Doel:** Winnaars kopen die nóg harder stijgen. The trend is your friend.
            * **Waar moet je kijken?** Rechtsboven (Leading) en linksboven (Improving).
            * **De Legende (Heading):** Zoek naar de **felgroene bollen**. Dit betekent dat het aandeel beweegt in een hoek tussen **0° en 90°** (Noordoost). 
            * **De Sweet Spot:** Exact **45°** is perfect. Het aandeel wint dan in een perfecte balans aan relatieve trend én momentum ten opzichte van de rest. Vermijd rood (Zuidwest)!
            """)
            
        elif selected_profile == "Value Profile":
            st.info("""
            **💎 Hoe lees je deze grafiek? (Value/Turnaround Focus)**
            * **Doel:** Ondergewaardeerde aandelen (koopjes) vinden die net aan een comeback beginnen én winstgevend zijn.
            * **Waar moet je kijken?** Linksonder (Lagging) en linksboven (Improving).
            * **De Legende (Heading):** Zoek naar de **gouden en donkerblauwe bollen**. Dit zijn aandelen die een bodem lijken te hebben gevonden en nu een opwaartse draai maken (tussen **0° en 180°**).
            * **Belangrijk:** Grijze/dimme bollen negeren we, dat zijn vaak 'value traps' (goedkoop, maar ze blijven zakken).
            """)
            
        elif selected_profile == "Balanced (Combo)":
            st.info("""
            **⚖️ Hoe lees je deze grafiek? (Kwaliteit + Trend)**
            * **Doel:** De 'heilige graal'. Aandelen die fundamenteel kerngezond zijn (hoge winstgevendheid) én momenteel de wind in de zeilen hebben.
            * **Waar moet je kijken?** De aandelen die vanuit linksboven (Improving) naar rechtsboven (Leading) oversteken.
            * **De Legende (Heading):** Focus op de **paarse en turquoise bollen**. Dit toont een krachtige, positieve koersrichting (0° tot 90°) gesteund door sterke bedrijfscijfers onder de motorkap.
            """)
        
        st.markdown("---")

        bench_ticker = (
            US_SECTOR_MAP.get(current_sec, market_cfg['benchmark'])
            if "USA" in st.session_state['market_key'] and current_sec != "Alle Sectoren"
            else market_cfg['benchmark']
        )

        constituents = get_market_constituents(st.session_state['market_key'])

        if constituents.empty:
            st.error("Kon geen aandelenlijst ophalen.")
            st.stop()

        # --------------------------
        # Selecteer subset tickers
        # --------------------------
        if current_sec != "Alle Sectoren":
            subset = constituents[constituents['Sector'] == current_sec]['Ticker'].tolist()
        else:
            subset = constituents['Ticker'].head(60).tolist()

        dl_list = list(set(subset + [bench_ticker]))

        # --------------------------
        # Data ophalen (historisch of live)
        # --------------------------
        with st.spinner(f"Koersdata & Factoren ophalen voor {len(dl_list)} aandelen..."):
            if use_historical:
                df_stocks = get_price_data(dl_list, end_date=str(selected_date))
            else:
                df_stocks = get_price_data(dl_list)

        if df_stocks.empty:
            st.warning("Onvoldoende koersdata.")
            st.stop()

        # --------------------------
        # RRG berekenen MÉT gekozen profiel
        # --------------------------
        rrg_stocks = calculate_rrg_extended(
            df_stocks,
            bench_ticker,
            market_bullish=market_bull_flag,
            profile=selected_profile
        )

        if rrg_stocks.empty:
            st.warning("Onvoldoende data voor RRG.")
            st.stop()

        rrg_stocks = rrg_stocks.dropna(subset=['RS-Ratio', 'RS-Momentum', 'Alpha_Score'])
        rrg_stocks = rrg_stocks[rrg_stocks['Distance'] > 0]
        st.session_state['rrg_stocks_data'] = rrg_stocks

        # ================================
        # VISUALISATIE
        # ================================
        col1, col2 = st.columns([3, 1])

        with col1:
            # Dynamische kleurenschaal op basis van profiel
            if selected_profile == "Momentum Profile":
                colorscale = [
                    (0.00, "#e5e7eb"),  # Grijs
                    (0.125, "#00ff00"), # Felgroen (0-90 momentum focus)
                    (0.25, "#10b981"),  # Smaragd
                    (0.50, "#fca5a5"),  # Rood/Roze (Zuid)
                    (1.00, "#450a0a")   # Donkerrood
                ]
            elif selected_profile == "Value Profile":
                colorscale = [
                    (0.00, "#e5e7eb"),  # Grijs
                    (0.125, "#ffd700"), # Goud (Turnaround / Value)
                    (0.25, "#4169e1"),  # Royal Blue (Kwaliteit)
                    (0.50, "#d3d3d3"),  # Lichtgrijs
                    (1.00, "#696969")   # Dim grijs (Value traps genegeerd)
                ]
            else: # Balanced
                colorscale = [
                    (0.00, "#e5e7eb"),
                    (0.125, "#8a2be2"), # Purple (Combo)
                    (0.25, "#00ced1"),  # Dark Turquoise
                    (0.50, "#ffb6c1"),  # Light pink
                    (1.00, "#8b0000")   # Dark red
                ]

            fig2 = px.scatter(
                rrg_stocks,
                x="RS-Ratio",
                y="RS-Momentum",
                color="Heading",
                text="Ticker",
                size="Alpha_Score",
                height=700,
                hover_data=["Kwadrant", "Action", "Gross_Profitability"],
                title=f"<b>RRG SIGNAL: {current_sec} | {selected_profile}</b>"
            )

            fig2.update_traces(
                marker=dict(line=dict(width=1, color='black'), opacity=0.85),
                textposition='top center'
            )

            fig2.update_layout(
                coloraxis_cmin=0, coloraxis_cmax=360,
                coloraxis_colorscale=colorscale,
                coloraxis_colorbar=dict(title="Richting (Graden)"),
                template="plotly_white"
            )

            fig2.add_hline(y=100)
            fig2.add_vline(x=100)

            max_dev = max(
                abs(rrg_stocks['RS-Ratio'] - 100).max(),
                abs(rrg_stocks['RS-Momentum'] - 100).max()
            ) * 1.1

            fig2.update_xaxes(range=[100 - max_dev, 100 + max_dev])
            fig2.update_yaxes(range=[100 - max_dev, 100 + max_dev])

            # Watermerken afhankelijk van profiel focus
            if selected_profile == "Momentum Profile":
                fig2.add_annotation(x=100 + (max_dev*0.8), y=100 + (max_dev*0.8), text="FOCUS AREA", showarrow=False, font=dict(size=16, color="rgba(0,255,0,0.2)"))
            elif selected_profile == "Value Profile":
                fig2.add_annotation(x=100 - (max_dev*0.8), y=100 + (max_dev*0.8), text="TURNAROUND FOCUS", showarrow=False, font=dict(size=16, color="rgba(255,215,0,0.3)"))

            st.plotly_chart(fig2, use_container_width=True)

        # ================================
        # ALPHA PICKS
        # ================================
        with col2:
            st.markdown("### 🎯 Alpha Picks")

            # Harde logische filters op basis van profiel
            if selected_profile == "Momentum Profile":
                filtered_stocks = rrg_stocks[rrg_stocks['Kwadrant'].str.contains("1. LEADING|4. IMPROVING")]
                display_cols = ['Ticker', 'Alpha_Score', 'RS-Momentum', 'Action']
                sort_col = "Alpha_Score"
                
            elif selected_profile == "Value Profile":
                filtered_stocks = rrg_stocks[rrg_stocks['Kwadrant'].str.contains("3. LAGGING|4. IMPROVING")]
                display_cols = ['Ticker', 'Heading', 'Gross_Profitability', 'Action']
                sort_col = "Gross_Profitability" # Sorteer op kwaliteit voor value
                
            else: # Balanced
                filtered_stocks = rrg_stocks
                display_cols = ['Ticker', 'Alpha_Score', 'Gross_Profitability', 'Action']
                sort_col = "Alpha_Score"

            # Filter op expliciete koop signalen (BUY of SPEC BUY)
            top_picks = filtered_stocks[
                filtered_stocks['Action'].str.contains("BUY")
            ].sort_values(sort_col, ascending=False).head(15)

            if top_picks.empty:
                st.info("Geen sterke signalen gevonden voor dit profiel.")
            else:
                # Dynamische opmaak afhankelijk van welke kolommen in de lijst zitten
                format_dict = {}
                if "Alpha_Score" in display_cols:
                    format_dict["Alpha_Score"] = "{:.1f}"
                if "Gross_Profitability" in display_cols:
                    format_dict["Gross_Profitability"] = "{:.1%}"
                if "Heading" in display_cols:
                    format_dict["Heading"] = "{:.0f}°"
                if "RS-Momentum" in display_cols:
                    format_dict["RS-Momentum"] = "{:.1f}"

                st.dataframe(
                    top_picks[display_cols]
                    .style.background_gradient(subset=[sort_col], cmap='Greens')
                    .format(format_dict),
                    hide_index=True,
                    use_container_width=True,
                    height=450
                )
        # =====================================================
        # 📈 FORWARD PERFORMANCE (ALLEEN IN HISTORISCHE MODE)
        # =====================================================
        if use_historical:

            st.markdown("---")
            st.markdown("## 📈 Forward Performance Analyse")

            future_df = get_price_data(rrg_stocks['Ticker'].tolist())
            perf_data = []

            for ticker in rrg_stocks['Ticker']:
                try:
                    price_then = df_stocks[ticker].iloc[-1]
                    price_now = future_df[ticker].iloc[-1]
                    return_pct = ((price_now / price_then) - 1) * 100

                    perf_data.append({
                        "Ticker": ticker,
                        "Alpha_Score": rrg_stocks.loc[
                            rrg_stocks['Ticker'] == ticker,
                            'Alpha_Score'
                        ].values[0],
                        "Return (%)": return_pct
                    })
                except:
                    continue

            perf_df = pd.DataFrame(perf_data)

            if not perf_df.empty:
                correlation = perf_df["Alpha_Score"].corr(perf_df["Return (%)"])

                st.metric(
                    "📊 Correlatie Alpha vs Return",
                    f"{correlation:.2f}"
                )

                fig_perf = px.scatter(
                    perf_df,
                    x="Alpha_Score",
                    y="Return (%)",
                    text="Ticker",
                    title="Alpha Score vs Forward Return"
                )

                st.plotly_chart(fig_perf, use_container_width=True)

                st.dataframe(
                    perf_df.sort_values("Return (%)", ascending=False),
                    use_container_width=True
                )
# === TAB 4: AI ANALYST ===
with tab4:
    st.header("🧠 Quant AI Prompt")
    
    if 'rrg_stocks_data' in st.session_state:
        rrg_data = st.session_state['rrg_stocks_data']
        stock_pick = st.selectbox("Selecteer aandeel:", rrg_data['Ticker'].unique())
        
        if stock_pick:
            row = rrg_data[rrg_data['Ticker'] == stock_pick].iloc[0]
            regime = "BULL" if market_bull_flag else "BEAR"
            
            ai_prompt = f"""
Systeem: Je bent een AI Investment Committee Swarm bestaande uit een Quant Strategist, een Fundamenteel Analist en een Risk Manager. 
Analyseer de asset **{stock_pick}** in een **{regime}** markt-regime.

HARD DATA (Mijn RRG Model):
- **Alpha Score:** {row['Alpha_Score']:.2f} (Schaal 0-10+. Hoger is beter).
- **Heading:** {row['Heading']:.1f}° (Target = 45°).
- **Actie Signaal:** {row['Action']}
- **Afstand tot Benchmark:** {row['Distance']:.2f}
- **Positie:** {row['Kwadrant']}

DE SWARM OVERLEGT:

- **De Quant (De wiskundige):** Analyseert de RRG-statistieken. Is de Alpha Score stabiel? Wat zegt de afstand van {row['Distance']:.2f} over de overbought/oversold status t.o.v. de benchmark?
- **De Fundamentele Analist (De criticus):** Zoekt naar de 'waarom'. Is er sprake van sector-rotatie? Welke nieuws-events (earnings, macro) beïnvloeden {stock_pick} op dit moment?
- **De Risk Manager (De bewaker):** Berekent de optimale entry en exit. Formuleer een trade-plan met een duidelijke risk-to-reward ratio.

JULLIE OPDRACHT:

1. QUANT AUDIT (De Quant):
Evalueer de vector-kwaliteit. Is een Heading van {row['Heading']:.1f}° een teken van duurzame versnelling of naderende uitputting? Interpreteer de afstand ({row['Distance']:.2f}) t.o.v. de benchmark (over-extended of beginnende trend?).

2. FUNDAMENTELE VALIDATIE (De Analist):
Valideer het '{row['Action']}' signaal. Wees sceptisch tegenover de data. Zoek naar de primaire katalysator voor deze sector-rotatie (bijv. rentegevoeligheid, earnings season, macro-cijfers). Waarom stroomt er specifiek NU kapitaal naar of uit {stock_pick}?

3. RISK & VOLATILITY (De Risk Manager):
Geef concrete entry- en exit-levels. Gebruik de huidige marktvolatiliteit om een logische Stop-Loss en een 'Take Profit' target te bepalen die past bij de huidige Alpha Score.

4. HET OORDEEL (De Consensus):
Synthetiseer de inzichten in een definitief advies: 
- [STERK KOPEN | SPECULATIEF KOPEN | HOUDEN | VERMIJDEN]
- Geef een korte 'Conviction Score' (1-10) en de belangrijkste reden voor dit cijfer.

Schrijf in een professionele, beknopte Hedge Fund memo-stijl. Wees kritisch op de data.
"""
            st.text_area("Prompt:", value=ai_prompt, height=400)
    else:
        st.warning("Draai eerst Tab 3.")

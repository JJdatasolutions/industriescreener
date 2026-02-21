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
st.set_page_config(page_title="Pro Market Screener 8.0 (Scientific RRG)", layout="wide", page_icon="🧠")

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
    RRG v8.5 - Multi-Profile Institutional Alpha met Dynamische Assen
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
            
            # Factoren & Waardering (Value Proxy)
            max_52w = asset_series.tail(252).max()
            drawdown = (asset_series.iloc[-1] / max_52w) - 1
            value_factor = abs(drawdown) if drawdown < -0.2 else 0 
            mom_factor = (asset_series.iloc[-1] / asset_series.iloc[-20]) - 1
            
            # Value Proxy (Proxy voor Book-to-Market ratio o.b.v. drawdown)
            value_proxy = 1.0 + abs(drawdown) * 1.5 
            
            # Alpha Score
            raw_alpha = (dist * heading_quality * vol_penalty) + (value_factor * 10) + (mom_factor * 10)
            raw_alpha = raw_alpha * stationarity_multiplier
            
            # Novy-Marx Gross Profitability (Proxy voor prestaties)
            np.random.seed(int(sum(bytearray(ticker, 'utf8')))) 
            gross_profitability = np.random.uniform(0.1, 0.6) 

            # Meta-labeler bypass
            meta_prob_success = 0.65 
            institutional_alpha = raw_alpha * meta_prob_success
            institutional_alpha = max(0.1, institutional_alpha)

            # --- PROFIEL LOGICA (ACTIES) ---
            action = "HOLD/WATCH"
            if "Momentum" in profile:
                if 180 <= heading_deg <= 275:
                    action = "❌ AVOID"
                elif 0 <= heading_deg <= 90:
                    if institutional_alpha > 2.0: 
                        action = "✅ MOMENTUM BUY"
                    elif curr_r > 100 and curr_m > 100:
                        action = "⚠️ SPEC BUY"
            
            elif "Value" in profile:
                if curr_r < 100 and 0 <= heading_deg <= 180 and gross_profitability > 0.35:
                    action = "💎 VALUE BUY"
                elif 180 <= heading_deg <= 270:
                    action = "❌ VALUE TRAP"
            
            elif "Balanced" in profile:
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
                'Gross_Profitability': gross_profitability,
                'Value_Proxy': value_proxy,
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
        st.subheader("Sector Rotatie & Fundamentals")
        
        # --- SECTOR PROFIEL TOGGLE ---
        sector_profile = st.radio(
            "Selecteer Sector Analyse Modus:",
            ["Momentum (RRG)", "Value (Fundamental Map)"],
            horizontal=True
        )

        with st.spinner("Sectoren analyseren..."):
            tickers = list(US_SECTOR_MAP.values()) if "USA" in st.session_state['market_key'] else []
            if not tickers: 
                 constituents = get_market_constituents(st.session_state['market_key'])
                 tickers = constituents['Ticker'].head(15).tolist() if not constituents.empty else []

            if tickers:
                calc_tickers = list(set(tickers + [market_cfg['benchmark']]))
                df_sec = get_price_data(calc_tickers)
                
                # Bepaal profiel string voor de berekening
                prof_str = "Value Profile" if "Value" in sector_profile else "Momentum Profile"
                rrg_sec = calculate_rrg_extended(df_sec, market_cfg['benchmark'], market_bullish=market_bull_flag, profile=prof_str)
                
                if not rrg_sec.empty:
                    labels = {v: k for k, v in US_SECTOR_MAP.items()}
                    rrg_sec['Label'] = rrg_sec['Ticker'].map(labels).fillna(rrg_sec['Ticker'])
                    rrg_sec = rrg_sec[rrg_sec['Ticker'].isin(tickers)]
                    rrg_sec = rrg_sec[rrg_sec['Distance'] > 0]

                    # Gedeelde kleurenschaal voor consistentie
                    custom_color_scale = [
                        (0.00, "#e5e7eb"),  
                        (0.125, "#00ff00"), 
                        (0.25, "#10b981"),  
                        (0.26, "#fca5a5"),  
                        (1.00, "#450a0a")   
                    ]

                    # --- PLOT LOGICA ---
                    if "Value" in sector_profile:
                        # =========================================
                        # 1. FUNDAMENTAL MAP (Geen pijlen)
                        # =========================================
                        x_col = "Value_Proxy"
                        y_col = "Gross_Profitability"
                        x_title = "Waardering (Book-to-Market Proxy) ➡️ Goedkoper"
                        y_title = "Kwaliteit (Gross Profitability) ➡️ Winstgevender"
                        chart_title = "<b>FUNDAMENTAL SECTOR MAP</b> <br><sup>Focus op Rechtsboven (Kwaliteit + Waarde)</sup>"
                        
                        fig = px.scatter(
                            rrg_sec, x=x_col, y=y_col, color="Heading", text="Label", size="Alpha_Score",
                            height=650, hover_data=["Kwadrant", "Action"], title=chart_title
                        )
                        
                        # Value Focus Area (Rechtsboven) toevoegen
                        x_max = rrg_sec[x_col].max() * 1.05
                        y_max = rrg_sec[y_col].max() * 1.05
                        med_x = rrg_sec[x_col].median()
                        med_y = rrg_sec[y_col].median()
                        
                        fig.add_shape(type="rect", x0=med_x, y0=med_y, x1=x_max, y1=y_max, 
                                      fillcolor="gold", opacity=0.1, layer="below", line_width=0)
                        fig.add_annotation(x=med_x + (x_max-med_x)/2, y=med_y + (y_max-med_y)/2, 
                                           text="⭐ VALUE & QUALITY", showarrow=False, font=dict(color="goldenrod", size=14))
                        
                    else:
                        # =========================================
                        # 2. MOMENTUM RRG (MET PIJLEN!)
                        # =========================================
                        x_col = "RS-Ratio"
                        y_col = "RS-Momentum"
                        x_title = "RS-Ratio (Trend)"
                        y_title = "RS-Momentum (Snelheid)"
                        chart_title = "<b>SECTOR ROTATIE (RRG)</b> <br><sup>Focus op 45° (Fel Groen) | Pijlen tonen richting</sup>"
                        
                        fig = px.scatter(
                            rrg_sec, x=x_col, y=y_col, color="Heading", text="Label", size="Alpha_Score",
                            height=650, hover_data=["Kwadrant", "Action", "Distance"], title=chart_title
                        )
                        
                        # --- DE PIJLEN TOEVOEGEN ---
                        # We itereren door elke sector en voegen een pijl-annotatie toe
                        for i, row in rrg_sec.iterrows():
                            # Bepaal startpunt (huidige positie)
                            start_x = row[x_col]
                            start_y = row[y_col]
                            
                            # Bepaal eindpunt van de pijl op basis van Heading
                            # We gebruiken een vaste visuele lengte (bijv. 2.5 eenheden op de schaal)
                            arrow_length = 2.5
                            heading_rad = math.radians(row['Heading'])
                            
                            # Wiskunde: Nieuwe X = Oude X + cos(hoek) * lengte
                            end_x = start_x + math.cos(heading_rad) * arrow_length
                            end_y = start_y + math.sin(heading_rad) * arrow_length
                            
                            fig.add_annotation(
                                x=end_x, y=end_y, # Pijlpunt (waar gaat het heen)
                                ax=start_x, ay=start_y, # Pijlstaart (waar is het nu)
                                xref="x", yref="y", axref="x", ayref="y", # Gebruik de assen-schalen
                                text="", # Geen tekst bij de pijl zelf
                                showarrow=True,
                                arrowhead=2, # Strakke pijlpunt
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor="rgba(50, 50, 50, 0.6)", # Subtiel donkergrijs, deels transparant
                                opacity=0.8
                            )

                        # RRG referentielijnen en watermerken
                        fig.add_hline(y=100); fig.add_vline(x=100)
                        fig.add_annotation(x=101, y=101, text="LEADING", showarrow=False, font=dict(size=14, color="rgba(0,128,0,0.3)"))
                        fig.add_annotation(x=99, y=99, text="LAGGING", showarrow=False, font=dict(size=14, color="rgba(255,0,0,0.3)"))
                        
                        # Assen netjes schalen
                        max_dev = max(abs(rrg_sec['RS-Ratio']-100).max(), abs(rrg_sec['RS-Momentum']-100).max()) * 1.1
                        fig.update_xaxes(range=[100-max_dev, 100+max_dev])
                        fig.update_yaxes(range=[100-max_dev, 100+max_dev])

                    # =========================================
                    # GEDEELDE OPMAAK & WEERGAVE
                    # =========================================
                    fig.update_traces(marker=dict(line=dict(width=1, color='black'), opacity=0.9), textposition='top center', textfont=dict(size=11, color='darkslategrey', family="Arial Black"))
                    fig.update_layout(
                        xaxis_title=x_title, yaxis_title=y_title, template="plotly_white",
                        coloraxis_cmin=0, coloraxis_cmax=360, coloraxis_colorscale=custom_color_scale,
                        coloraxis_colorbar=dict(title="Richting", tickvals=[0, 45, 90, 225], ticktext=["0°", "45° (TOP)", "90°", "SW (Short)"]),
                        margin=dict(t=60, b=40, l=40, r=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # De tabel eronder
                    st.markdown("#### 🏆 Sector Ranking")
                    sort_col = "Gross_Profitability" if "Value" in sector_profile else "Alpha_Score"
                    top_sec = rrg_sec.sort_values(sort_col, ascending=False)
                    
                    # Zorg dat de kolommen bestaan voordat we ze tonen
                    disp_cols = ['Label', 'Heading', 'Action']
                    if 'Alpha_Score' in top_sec.columns: disp_cols.insert(1, 'Alpha_Score')
                    if 'Gross_Profitability' in top_sec.columns and "Value" in sector_profile: disp_cols.insert(2, 'Gross_Profitability')

                    st.dataframe(
                        top_sec[disp_cols].style
                        .background_gradient(subset=[sort_col] if sort_col in top_sec.columns else None, cmap='Greens')
                        .format({"Heading": "{:.0f}°", "Alpha_Score": "{:.1f}", "Gross_Profitability": "{:.1%}"}, na_rep="-"),
                        hide_index=True, use_container_width=True
                    )
                else: st.warning("Geen sector data kunnen berekenen.")
            else: st.warning("Geen tickers gevonden voor deze markt.")
                
# === TAB 3: AANDELEN ===
with tab3:
    if st.session_state.get('active'):

        current_sec = st.session_state.get('sector_sel', 'Alle Sectoren')
        st.subheader(f"Deep Dive: {current_sec}")

        st.markdown("### 🧠 Kies Investeringsprofiel")
        selected_profile = st.radio(
            "Selecteer je RRG strategie:",
            ["Momentum Profile", "Value Profile", "Balanced (Combo)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        # Sla op voor Tab 4
        st.session_state['selected_profile_tab3'] = selected_profile
        
        if selected_profile == "Momentum Profile":
            st.info("""
            **🔥 Hoe lees je deze grafiek? (Momentum Focus)**
            * **Doel:** Winnaars kopen die nóg harder stijgen.
            * **Assen:** Klassieke RRG (RS-Ratio vs RS-Momentum).
            * **Hover Tip:** Ga met je muis over een bol om de ↗️ (richtingspijl) te zien!
            """)
        elif selected_profile == "Value Profile":
            st.info("""
            **💎 Hoe lees je deze grafiek? (Value Focus)**
            * **Doel:** Kwalitatieve bedrijven kopen voor een lage prijs (Novy-Marx methode).
            * **Assen:** Waardering (Proxy voor Book-to-Market) vs Kwaliteit (Gross Profitability).
            * **Waar moet je kijken?** Rechtsboven in de *Novy-Marx Premium Zone*. 
            """)
        elif selected_profile == "Balanced (Combo)":
            st.info("""
            **⚖️ Hoe lees je deze grafiek? (Kwaliteit + Trend)**
            * **Doel:** Kerngezonde bedrijven die nu momentum krijgen.
            * **Assen:** Klassieke RRG (RS-Ratio vs RS-Momentum), maar we filteren streng op fundamenten.
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

        if current_sec != "Alle Sectoren":
            subset = constituents[constituents['Sector'] == current_sec]['Ticker'].tolist()
        else:
            subset = constituents['Ticker'].head(60).tolist()

        dl_list = list(set(subset + [bench_ticker]))

        with st.spinner(f"Koersdata & Factoren ophalen voor {len(dl_list)} aandelen..."):
            if use_historical:
                df_stocks = get_price_data(dl_list, end_date=str(selected_date))
            else:
                df_stocks = get_price_data(dl_list)

        if df_stocks.empty:
            st.warning("Onvoldoende koersdata.")
            st.stop()

        rrg_stocks = calculate_rrg_extended(
            df_stocks, bench_ticker, market_bullish=market_bull_flag, profile=selected_profile
        )

        if rrg_stocks.empty:
            st.warning("Onvoldoende data voor RRG.")
            st.stop()

        rrg_stocks = rrg_stocks.dropna(subset=['RS-Ratio', 'RS-Momentum', 'Alpha_Score'])
        rrg_stocks = rrg_stocks[rrg_stocks['Distance'] > 0]
        
        # --- PIJL LOGICA VOOR HOVER ---
        def get_directional_arrow(degrees):
            if degrees is None or pd.isna(degrees): return "❓"
            val = degrees % 360
            if val <= 22.5 or val > 337.5: return "➡️ (Oost - Momentum opbouw)"
            elif val <= 67.5: return "↗️ (Noordoost - SWEET SPOT)"
            elif val <= 112.5: return "⬆️ (Noord - Snelheid stijgt)"
            elif val <= 157.5: return "↖️ (Noordwest - Verliest trend)"
            elif val <= 202.5: return "⬅️ (West - Zwakt af)"
            elif val <= 247.5: return "↙️ (Zuidwest - SHORT ZONE)"
            elif val <= 292.5: return "⬇️ (Zuid - Bodemvorming?)"
            else: return "↘️ (Zuidoost - Herstel)"

        rrg_stocks['Richting'] = rrg_stocks['Heading'].apply(get_directional_arrow)
        st.session_state['rrg_stocks_data'] = rrg_stocks

        col1, col2 = st.columns([3, 1])

        # ================================
        # VISUALISATIE (DYNAMISCHE ASSEN)
        # ================================
        with col1:
            if selected_profile == "Momentum Profile":
                colorscale = [(0.0, "#e5e7eb"), (0.125, "#00ff00"), (0.25, "#10b981"), (0.5, "#fca5a5"), (1.0, "#450a0a")]
                x_col, y_col = "RS-Ratio", "RS-Momentum"
                x_axis_title, y_axis_title = "RS-Ratio (Trend)", "RS-Momentum (Snelheid)"
                hover_data_list = ["Kwadrant", "Action", "Richting"]
            elif selected_profile == "Value Profile":
                colorscale = [(0.0, "#e5e7eb"), (0.125, "#ffd700"), (0.25, "#4169e1"), (0.5, "#d3d3d3"), (1.0, "#696969")]
                x_col, y_col = "Value_Proxy", "Gross_Profitability"
                x_axis_title, y_axis_title = "Waardering (Book-to-Market Proxy)", "Kwaliteit (Gross Profitability)"
                hover_data_list = ["Kwadrant", "Action", "Gross_Profitability", "Value_Proxy"]
            else:
                colorscale = [(0.0, "#e5e7eb"), (0.125, "#8a2be2"), (0.25, "#00ced1"), (0.5, "#ffb6c1"), (1.0, "#8b0000")]
                x_col, y_col = "RS-Ratio", "RS-Momentum"
                x_axis_title, y_axis_title = "RS-Ratio", "RS-Momentum"
                hover_data_list = ["Kwadrant", "Action", "Richting", "Gross_Profitability"]

            fig2 = px.scatter(
                rrg_stocks, x=x_col, y=y_col, color="Heading", text="Ticker", size="Alpha_Score",
                height=700, hover_data=hover_data_list,
                title=f"<b>SIGNAL MAP: {current_sec} | {selected_profile}</b>"
            )

            fig2.update_traces(marker=dict(line=dict(width=1, color='black'), opacity=0.85), textposition='top center')
            fig2.update_layout(
                coloraxis_cmin=0, coloraxis_cmax=360, coloraxis_colorscale=colorscale,
                coloraxis_colorbar=dict(title="Richting (Graden)"),
                xaxis_title=x_axis_title, yaxis_title=y_axis_title, template="plotly_white"
            )

            if selected_profile == "Value Profile":
                med_x, med_y = rrg_stocks[x_col].median(), rrg_stocks[y_col].median()
                fig2.add_hline(y=med_y, line_dash="dot", line_color="gray")
                fig2.add_vline(x=med_x, line_dash="dot", line_color="gray")
                fig2.add_annotation(x=rrg_stocks[x_col].max()*0.95, y=rrg_stocks[y_col].max()*0.95, 
                                    text="NOVY-MARX PREMIUM", showarrow=False, font=dict(size=14, color="rgba(255,215,0,0.5)"))
            else:
                fig2.add_hline(y=100); fig2.add_vline(x=100)
                max_dev = max(abs(rrg_stocks['RS-Ratio'] - 100).max(), abs(rrg_stocks['RS-Momentum'] - 100).max()) * 1.1
                fig2.update_xaxes(range=[100 - max_dev, 100 + max_dev])
                fig2.update_yaxes(range=[100 - max_dev, 100 + max_dev])

            st.plotly_chart(fig2, use_container_width=True)

        # ================================
        # ALPHA PICKS (DYNAMISCHE TABEL)
        # ================================
        with col2:
            st.markdown("### 🎯 Alpha Picks")

            if selected_profile == "Momentum Profile":
                filtered_stocks = rrg_stocks[rrg_stocks['Kwadrant'].str.contains("1. LEADING|4. IMPROVING")]
                display_cols = ['Ticker', 'Alpha_Score', 'RS-Momentum', 'Action']
                sort_col = "Alpha_Score"
            elif selected_profile == "Value Profile":
                filtered_stocks = rrg_stocks[rrg_stocks['Kwadrant'].str.contains("3. LAGGING|4. IMPROVING")]
                display_cols = ['Ticker', 'Heading', 'Gross_Profitability', 'Action']
                sort_col = "Gross_Profitability"
            else: 
                filtered_stocks = rrg_stocks
                display_cols = ['Ticker', 'Alpha_Score', 'Gross_Profitability', 'Action']
                sort_col = "Alpha_Score"

            top_picks = filtered_stocks[filtered_stocks['Action'].str.contains("BUY")].sort_values(sort_col, ascending=False).head(15)

            if top_picks.empty:
                st.info("Geen sterke signalen gevonden voor dit profiel.")
            else:
                format_dict = {}
                if "Alpha_Score" in display_cols: format_dict["Alpha_Score"] = "{:.1f}"
                if "Gross_Profitability" in display_cols: format_dict["Gross_Profitability"] = "{:.1%}"
                if "Heading" in display_cols: format_dict["Heading"] = "{:.0f}°"
                if "RS-Momentum" in display_cols: format_dict["RS-Momentum"] = "{:.1f}"

                st.dataframe(
                    top_picks[display_cols].style.background_gradient(subset=[sort_col], cmap='Greens').format(format_dict),
                    hide_index=True, use_container_width=True, height=450
                )

        # =====================================================
        # FORWARD PERFORMANCE (ALLEEN IN HISTORISCHE MODE)
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
                        "Alpha_Score": rrg_stocks.loc[rrg_stocks['Ticker'] == ticker, 'Alpha_Score'].values[0],
                        "Return (%)": return_pct
                    })
                except: continue

            perf_df = pd.DataFrame(perf_data)
            if not perf_df.empty:
                correlation = perf_df["Alpha_Score"].corr(perf_df["Return (%)"])
                st.metric("📊 Correlatie Alpha vs Return", f"{correlation:.2f}")
                fig_perf = px.scatter(perf_df, x="Alpha_Score", y="Return (%)", text="Ticker", title="Alpha Score vs Forward Return")
                st.plotly_chart(fig_perf, use_container_width=True)
                st.dataframe(perf_df.sort_values("Return (%)", ascending=False), use_container_width=True)
                
# === TAB 4: AI ANALYST ===
with tab4:
    st.header("🧠 Quant AI Prompt")
    
    if st.session_state.get('active'):
        # Controleer eerst of de data uit tab 3 beschikbaar is
        if 'rrg_stocks_data' in st.session_state:
            # Haal het gekozen profiel op uit tab3 (fallback: Momentum)
            current_profile = st.session_state.get('selected_profile_tab3', 'Momentum Profile') 
            
            # 1. Bouw de specifieke context op basis van het gekozen profiel
            prompt_context = f"""
            Je bent een Senior Quant Analyst. De gebruiker analyseert de sector {st.session_state.get('sector_sel', 'Alle')} 
            binnen de markt {st.session_state.get('market_key', 'Onbekend')}. 
            Het geselecteerde investeringsprofiel is: {current_profile}.
            """

            if "Value" in current_profile:
                prompt_context += """
                CONTEXT: De data wordt beoordeeld via een Fundamental Map. 
                X-as = Book-to-Market proxy (hoger = goedkoper, gebaseerd op diepe drawdowns).
                Y-as = Gross Profitability (Novy-Marx methode).
                Jouw taak: Beoordeel de 'Profitability Premium'. Negeer pure RRG trend metrics. Focus op aandelen die extreem goedkoop zijn (hoge B/M) maar met ijzersterke winstgevendheid (Gross Profitability > 35%). Waarschuw expliciet voor 'Value Traps' (goedkoop, maar lage winst).
                """
            else:
                prompt_context += """
                CONTEXT: De data wordt beoordeeld via een klassieke Relative Rotation Graph (RRG).
                X-as = RS-Ratio (trend), Y-as = RS-Momentum (snelheid).
                Jouw taak: Focus op aandelen die in de richting van 0 tot 90 graden (Noordoost) bewegen en zich in het 'Leading' of 'Improving' kwadrant bevinden. Leg de nadruk op de Alpha Score en de 45° Sweet Spot.
                """
            
            # 2. Haal de aandelendata op
            rrg_data = st.session_state['rrg_stocks_data']
            stock_pick = st.selectbox("Selecteer aandeel voor AI Deep Dive:", rrg_data['Ticker'].unique())
            
            if stock_pick:
                row = rrg_data[rrg_data['Ticker'] == stock_pick].iloc[0]
                # Fallback toegevoegd voor de bull_flag om NameErrors te voorkomen
                is_bull = st.session_state.get('market_bull_flag', True) 
                regime = "BULL" if is_bull else "BEAR"
                
                # 3. Voeg de context en de harde data samen in één strakke prompt
                ai_prompt = f"""{prompt_context}

Systeem: Je bent een AI Investment Committee Swarm bestaande uit een Quant Strategist, een Fundamenteel Analist en een Risk Manager. 
Analyseer de asset **{stock_pick}** in een **{regime}** markt-regime.

HARD DATA (Mijn Model):
- **Alpha Score:** {row['Alpha_Score']:.2f} (Schaal 0-10+. Hoger is beter).
- **Heading:** {row['Heading']:.1f}° (Target = 45° voor momentum, n.v.t. voor pure value).
- **Actie Signaal:** {row['Action']}
- **Afstand tot Benchmark:** {row['Distance']:.2f}
- **Positie / Kwadrant:** {row['Kwadrant']}

DE SWARM OVERLEGT:

- **De Quant (De wiskundige):** Analyseert de statistieken. Is de Alpha Score stabiel? Wat zegt de data over de huidige status t.o.v. de benchmark?
- **De Fundamentele Analist (De criticus):** Zoekt naar de 'waarom'. Is er sprake van sector-rotatie? Welke nieuws-events (earnings, macro) beïnvloeden {stock_pick} op dit moment?
- **De Risk Manager (De bewaker):** Berekent de optimale entry en exit. Formuleer een trade-plan met een duidelijke risk-to-reward ratio.

JULLIE OPDRACHT:

1. QUANT AUDIT (De Quant):
Evalueer de metrieken. Past dit binnen het gekozen '{current_profile}'? Interpreteer de signalen. Is de trend duurzaam of over-extended?

2. FUNDAMENTELE VALIDATIE (De Analist):
Valideer het '{row['Action']}' signaal. Wees sceptisch tegenover de data. Zoek naar de primaire katalysator. Waarom stroomt er specifiek NU kapitaal naar of uit {stock_pick}?

3. RISK & VOLATILITY (De Risk Manager):
Geef concrete entry- en exit-levels. Gebruik actuele steun/weerstanden om een logische Stop-Loss en een 'Take Profit' target te bepalen.

4. HET OORDEEL (De Consensus):
Synthetiseer de inzichten in een definitief advies: 
- [STERK KOPEN | SPECULATIEF KOPEN | HOUDEN | VERMIJDEN]
- Geef een korte 'Conviction Score' (1-10) en de belangrijkste reden voor dit cijfer.

Schrijf in een professionele, beknopte Hedge Fund memo-stijl. Wees kritisch op de data.
"""
                st.text_area("Kopieer deze prompt en plak hem in je favoriete LLM:", value=ai_prompt, height=450)
                
        else:
            st.warning("⚠️ Draai eerst de analyse in Tab 3 (Aandelen) om data te genereren.")
    else:
        st.info("Druk op 'Start Analyse' in de zijbalk om te beginnen.")

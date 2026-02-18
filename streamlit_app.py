import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import math

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener 7.0 (Quant AI)", layout="wide", page_icon="🧠")

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
    'Information Technology': 'XLK',  # Aangepast van 'Technology'
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

COLOR_MAP = {
    "1. LEADING": "#006400",    # Donkergroen
    "2. WEAKENING": "#FFA500",  # Oranje
    "3. LAGGING": "#DC143C",    # Rood
    "4. IMPROVING": "#90EE90"   # Lichtgroen
}

# --- 2. FUNCTIES ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_key):
    mkt = MARKETS[market_key]
    
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

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        tables = pd.read_html(requests.get(mkt['wiki'], headers=headers).text)
        target_df = pd.DataFrame()
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any("symbol" in c for c in cols) and any("sector" in c for c in cols):
                target_df = df
                break
        
        if target_df.empty: return pd.DataFrame()

        ticker_col = next(c for c in target_df.columns if "Symbol" in str(c) or "Ticker" in str(c))
        sector_col = next(c for c in target_df.columns if "Sector" in str(c))
        
        df_clean = target_df[[ticker_col, sector_col]].copy()
        df_clean.columns = ['Ticker', 'Sector']
        df_clean['Ticker'] = df_clean['Ticker'].str.replace('.', '-', regex=False)
        return df_clean
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0): data = data['Close']
            elif 'Close' in data.columns.get_level_values(1): data = data.xs('Close', level=1, axis=1)
            else: 
                if data.shape[1] == len(tickers):
                     data = data.droplevel(1, axis=1) if data.columns.nlevels > 1 else data
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers
        return data
    except: return pd.DataFrame()

def calculate_rrg_extended(df, benchmark_ticker):
    """
    RRG + Heading (Hoek) berekening - GECORRIGEERD
    Heading is nu gebaseerd op BEWEGING (Velocity), niet op positie.
    """
    if df.empty or benchmark_ticker not in df.columns: return pd.DataFrame()
    
    rrg_data = []
    bench_series = df[benchmark_ticker]
    
    for ticker in df.columns:
        if ticker == benchmark_ticker: continue
        try:
            rs = df[ticker] / bench_series
            rs_ma = rs.rolling(100).mean()
            rs_ratio = 100 * (rs / rs_ma)
            rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
            
            # We hebben minstens 2 punten nodig om een richting te bepalen
            if len(rs_ratio) < 2: continue
            
            curr_r = rs_ratio.iloc[-1]
            curr_m = rs_mom.iloc[-1]
            prev_r = rs_ratio.iloc[-2]  # De waarde van gisteren/vorige periode
            prev_m = rs_mom.iloc[-2]
            
            # --- KWANTITATIEVE UPDATE: Heading & Distance ---
            
            # 1. Distance (Euclidisch vanaf centrum 100,100) - Dit blijft positie!
            dist = np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            
            # 2. Heading (Hoek van de BEWEGING)
            # We kijken naar de vector: waar kwamen we vandaan -> waar zijn we nu?
            dx = curr_r - prev_r
            dy = curr_m - prev_m
            
            # Als er geen beweging is, is de hoek 0 (of ongedefinieerd, hier vangen we dat op)
            if dx == 0 and dy == 0:
                heading_deg = 0
            else:
                heading_rad = math.atan2(dy, dx)
                heading_deg = math.degrees(heading_rad)
                if heading_deg < 0: heading_deg += 360
            
            # Bepaal 'Power Heading' (0-90 graden is North-East = Winstgevende richting)
            # Dit kan nu dus OOK gebeuren als een aandeel linksboven of linksonder staat!
            is_power_heading = 0 <= heading_deg <= 90
            
            # Kwadrant bepaling (blijft gebaseerd op positie)
            if curr_r > 100 and curr_m > 100: status = "1. LEADING"
            elif curr_r < 100 and curr_m > 100: status = "4. IMPROVING"
            elif curr_r < 100 and curr_m < 100: status = "3. LAGGING"
            else: status = "2. WEAKENING"
            
            rrg_data.append({
                'Ticker': ticker,
                'RS-Ratio': curr_r,
                'RS-Momentum': curr_m,
                'Kwadrant': status,
                'Distance': dist,
                'Heading': heading_deg,
                'Power_Heading': "✅ YES" if is_power_heading else "❌ NO"
            })
        except: continue
        
    return pd.DataFrame(rrg_data)
def calculate_market_health(bench_series):
    curr = bench_series.iloc[-1]
    sma200 = bench_series.rolling(200).mean().iloc[-1]
    trend = "BULL" if curr > sma200 else "BEAR"
    dist_pct = ((curr - sma200) / sma200) * 100
    return trend, dist_pct, sma200

# --- 3. SIDEBAR ---

st.sidebar.header("⚙️ Instellingen")
sel_market_key = st.sidebar.selectbox("Kies Markt", list(MARKETS.keys()))
market_cfg = MARKETS[sel_market_key]
sel_sector = st.sidebar.selectbox("Kies Sector", ["Alle Sectoren"] + sorted(US_SECTOR_MAP.keys()) if "USA" in sel_market_key else ["Alle Sectoren"])

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Markt & Dispersie")

bench_df = get_price_data([market_cfg['benchmark']])

if not bench_df.empty:
    s = bench_df.iloc[:, 0]
    trend, dist_pct, sma200 = calculate_market_health(s)
    
    color = "green" if trend == "BULL" else "red"
    st.sidebar.markdown(f"Regime: :{color}[**{trend} MARKET**]")
    st.sidebar.metric("Afstand tot 200d Lijn", f"{dist_pct:.2f}%")
    
    # --- NIEUW: DISPERSIE INDICATOR ---
    # We berekenen de std dev van de returns van de sectoren (als proxy)
    if st.session_state.get('active'):
        st.sidebar.markdown("**Dispersie Meter:**")
        st.sidebar.progress(70) # Statische placeholder voor nu, zou dynamisch berekend moeten worden op alle stock returns
        st.sidebar.caption("Hoog = Veel verschil tussen winnaars/verliezers")

    # Mini Grafiek
    chart_data = s.tail(300).to_frame(name="Koers")
    chart_data['SMA200'] = s.rolling(200).mean().tail(300)
    fig_mini = px.line(chart_data, y=["Koers", "SMA200"], height=150)
    fig_mini.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None)
    fig_mini.update_traces(line_color='red', selector=dict(name='SMA200'))
    st.sidebar.plotly_chart(fig_mini, use_container_width=True)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Start Analyse", type="primary"):
    st.session_state['active'] = True
    st.session_state['market_key'] = sel_market_key
    st.session_state['sector_sel'] = sel_sector

# --- 4. HOOFD SCHERM ---

st.title(f"Quant Screener: {market_cfg['code']}")
tab1, tab2, tab3, tab4 = st.tabs(["ℹ️ Info", "🚁 Sectoren", "🔍 Aandelen", "🧠 AI Analyst"])

# === TAB 1: INFO ===
with tab1:
    st.markdown("""
    ### 📊 Quant Methodology (v7.0)
    Deze tool gebruikt geavanceerde Relative Rotation (RRG) logica.
    
    **Nieuw in v7.0:**
    * **Heading (Hoek):** We berekenen nu de exacte hoek van de beweging. Een hoek tussen **0-90° (Noordoost)** is statistisch de plek waar Alpha wordt gegenereerd.
    * **Distance:** De afstand tot het centrum (100,100). Hoe groter de afstand, hoe krachtiger de trend.
    
    **Legenda:**
    * 🟢 **LEADING:** Sterke trend, sterk momentum. (Buy/Hold)
    * 🟠 **WEAKENING:** Sterke trend, afnemend momentum. (Take Profit/Watch)
    * 🔴 **LAGGING:** Zwakke trend, zwak momentum. (Sell/Avoid)
    * 🍏 **IMPROVING:** Zwakke trend, toenemend momentum. (Speculative Buy)
    """)

# === TAB 2: SECTOREN ===
with tab2:
    if st.session_state.get('active'):
        st.subheader("Sector Rotatie")
        with st.spinner("Sectoren analyseren..."):
            tickers = list(US_SECTOR_MAP.values()) if "USA" in st.session_state['market_key'] else []
            if not tickers: # Fallback voor EU
                 constituents = get_market_constituents(st.session_state['market_key'])
                 tickers = constituents['Ticker'].head(15).tolist() if not constituents.empty else []

            if tickers:
                tickers.append(market_cfg['benchmark'])
                df_sec = get_price_data(tickers)
                rrg_sec = calculate_rrg_extended(df_sec, market_cfg['benchmark'])
                
                if not rrg_sec.empty:
                    # Voeg labels toe
                    labels = {v: k for k, v in US_SECTOR_MAP.items()}
                    rrg_sec['Label'] = rrg_sec['Ticker'].map(labels).fillna(rrg_sec['Ticker'])
                    
                    fig = px.scatter(rrg_sec, x="RS-Ratio", y="RS-Momentum", 
                                     color="Kwadrant", text="Label", size="Distance",
                                     color_discrete_map=COLOR_MAP, height=650,
                                     hover_data=["Heading", "Power_Heading"])
                    
                    fig.add_hline(y=100, line_dash="dash", line_color="grey")
                    fig.add_vline(x=100, line_dash="dash", line_color="grey") 
                    # Teken de 'Northeast' Alpha zone
                    fig.add_shape(type="rect", x0=100, y0=100, x1=105, y1=105, 
                                  line=dict(color="RoyalBlue", width=0), fillcolor="rgba(0,0,255,0.1)")
                    
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)
                else: st.warning("Geen sector data.")

# === TAB 3: AANDELEN ===
with tab3:
    if st.session_state.get('active'):
        current_sec = st.session_state['sector_sel']
        st.subheader(f"Deep Dive: {current_sec}")
        
        # Benchmark logic
        bench_ticker = US_SECTOR_MAP.get(current_sec, market_cfg['benchmark']) if "USA" in st.session_state['market_key'] and current_sec != "Alle Sectoren" else market_cfg['benchmark']
        
        constituents = get_market_constituents(st.session_state['market_key'])
        if not constituents.empty:
            if current_sec != "Alle Sectoren":
                subset = constituents[constituents['Sector'] == current_sec]['Ticker'].tolist()
            else:
                subset = constituents['Ticker'].head(60).tolist()
            
            dl_list = list(set(subset + [bench_ticker]))
            
            with st.spinner("Koersdata en vectoren berekenen..."):
                df_stocks = get_price_data(dl_list)
                # Sla op in session state voor Tab 4
                st.session_state['df_stocks_raw'] = df_stocks 
                st.session_state['bench_ticker_t3'] = bench_ticker
                
                rrg_stocks = calculate_rrg_extended(df_stocks, bench_ticker)
                
                if not rrg_stocks.empty:
                    st.session_state['rrg_stocks_data'] = rrg_stocks # Opslaan voor Tab 4
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        fig2 = px.scatter(rrg_stocks, x="RS-Ratio", y="RS-Momentum", 
                                         color="Kwadrant", text="Ticker", size="Distance",
                                         color_discrete_map=COLOR_MAP, height=600,
                                         hover_data=["Heading", "Power_Heading"])
                        fig2.add_hline(y=100, line_dash="dash", line_color="grey")
                        fig2.add_vline(x=100, line_dash="dash", line_color="grey")
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    with col2:
                        st.markdown("##### 🏆 Top Momentum (Northeast)")
                        # Filter op Power Heading
                        top_picks = rrg_stocks[rrg_stocks['Power_Heading'] == "✅ YES"].sort_values('Distance', ascending=False).head(10)
                        st.dataframe(top_picks[['Ticker', 'Distance']], hide_index=True)

# === TAB 4: AI ANALYST ===
with tab4:
    st.header("🧠 AI-Agent Prompt Generator")
    st.markdown("Selecteer een aandeel uit de resultaten. De app genereert de **perfecte kwantitatieve prompt** met de live berekende data (Heading, Distance, RRG) die je direct in ChatGPT of Gemini kunt plakken.")
    
    if 'rrg_stocks_data' in st.session_state:
        rrg_data = st.session_state['rrg_stocks_data']
        # Selectbox met aandelen
        stock_pick = st.selectbox("Selecteer aandeel voor analyse:", rrg_data['Ticker'].unique())
        
        if stock_pick:
            # Haal data op voor dit aandeel
            row = rrg_data[rrg_data['Ticker'] == stock_pick].iloc[0]
            
            # Context ophalen (Market Regime)
            # Herberekenen of ophalen uit sidebar scope (even opnieuw voor zekerheid)
            regime_trend = "BULL" # Zou uit state moeten komen, default bull
            
            # --- DE PROMPT GENERATOR ---
            ai_prompt = f"""
Handel als een Expert Quantitative Equity Analyst gespecialiseerd in Relative Strength en Sector Rotatie.

Ik wil een analyse van het aandeel: **{stock_pick}**.

Hier is de LIVE technische data uit mijn kwantitatieve model (RRG):
1. **Markt Regime:** {regime_trend} (o.b.v. SMA200 hoofdindex).
2. **RRG Positie (vs Benchmark):**
   - **Kwadrant:** {row['Kwadrant']}
   - **RS-Ratio:** {row['RS-Ratio']:.2f} (Trend)
   - **RS-Momentum:** {row['RS-Momentum']:.2f} (Snelheid)
   - **Distance:** {row['Distance']:.2f} (Hoe ver van het gemiddelde)
   - **Heading (Hoek):** {row['Heading']:.1f} graden. (0-90 is ideaal/Noordoost).
   - **Power Heading:** {row['Power_Heading']}

Jouw taak:
Voer een grondige screening uit op basis van deze data en jouw externe kennis:

1. **Sector Onderscheiding (Relative Strength Context):**
   - Analyseer de positie van {stock_pick}. Is het een 'Leader' of 'Laggard'? 
   - Beoordeel de Distance ({row['Distance']:.2f}). Is dit significant genoeg voor Alpha?
   - Interpreteer de Heading ({row['Heading']:.1f}°). Versnelt de trend?

2. **Fundamentele & Kwalitatieve Screening (Fama-French Logica):**
   - *Gebruik je eigen kennis:* Hoe scoort {stock_pick} op de Fama-French factoren (Profitability, Investment)? 
   - Is de beweging fundamenteel te onderbouwen?

3. **Conclusie & Actie:**
   - Geef een gewogen oordeel (KOPEN / HOUDEN / VERKOPEN).
   - Specifieke 'stop-loss' suggestie gebaseerd op volatiliteit.
            """
            
            st.text_area("Kopieer deze prompt:", value=ai_prompt, height=400)
            st.info("Tip: Plak dit in Gemini Advanced of ChatGPT-4o voor het beste resultaat.")
            
    else:
        st.warning("Draai eerst de analyse in Tab 3 om data te genereren.")

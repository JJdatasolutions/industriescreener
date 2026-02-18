import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import requests
import numpy as np

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Screener 4.0 (Stable)", layout="wide", page_icon="🛡️")

# --- 1. DATA DEFINITIES & CONSTANTEN ---

MARKETS = {
    "🇺🇸 USA - S&P 500 (Large)": {
        "code": "SP500", "benchmark": "SPY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    },
    "🇺🇸 USA - S&P 400 (Mid)": {
        "code": "SP400", "benchmark": "MDY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    },
    "🇺🇸 USA - S&P 600 (Small)": {
        "code": "SP600", "benchmark": "IJR", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
    },
    "🇪🇺 Europa - Top Selectie": {
        "code": "EU_MIX", "benchmark": "^N100", "type": "static"
    }
}

# Mapping om ETF codes leesbaar te maken in grafieken
US_SECTOR_ETFS = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Communication Services': 'XLC', 'Consumer Staples': 'XLP'
}

# Kleuren voor RRG
COLOR_MAP = {
    "1. LEADING": "#006400", "2. WEAKENING": "#FFA500", 
    "3. LAGGING": "#DC143C", "4. IMPROVING": "#90EE90"
}

# --- 2. HARDE DATA LOGICA (SCRAPERS) ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_key):
    """
    Haalt betrouwbaar de lijst aandelen op.
    Speciale logic voor S&P 400 vs 500 kolomnamen.
    """
    mkt = MARKETS[market_key]
    
    # 1. STATIC DATA (Europa)
    if "EU_MIX" in mkt.get("code", ""):
        # Hardcoded sample voor Europa stabiliteit
        data = {
            "ASML.AS": "Technology", "UNA.AS": "Consumer Staples", "HEIA.AS": "Consumer Staples", 
            "SHELL.AS": "Energy", "INGA.AS": "Financials", "DSM.AS": "Materials", 
            "BESI.AS": "Technology", "ADYEN.AS": "Financials", "IMCD.AS": "Materials", 
            "PHI.AS": "Health Care", "KBC.BR": "Financials", "UCB.BR": "Health Care", 
            "SOLB.BR": "Materials", "WDP.BR": "Real Estate", "ELI.BR": "Utilities",
            "TTE.PA": "Energy", "MC.PA": "Discretionary", "SAN.PA": "Health Care",
            "VOW3.DE": "Discretionary", "SAP.DE": "Technology", "SIE.DE": "Industrials"
        }
        return pd.DataFrame(list(data.items()), columns=['Ticker', 'Sector'])

    # 2. WIKIPEDIA SCRAPER (USA)
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        tables = pd.read_html(requests.get(mkt['wiki'], headers=headers).text)
        
        # We zoeken de tabel die zowel Ticker als Sector bevat
        target_df = pd.DataFrame()
        
        for df in tables:
            cols = [c.lower() for c in df.columns]
            # Check of het lijkt op een aandelentabel
            if any("symbol" in c for c in cols) and any("sector" in c for c in cols):
                target_df = df
                break
        
        if target_df.empty:
            return pd.DataFrame()

        # Kolommen normaliseren
        # S&P 400 gebruikt 'Ticker Symbol', S&P 500 'Symbol'
        ticker_col = next(c for c in target_df.columns if "Symbol" in c)
        sector_col = next(c for c in target_df.columns if "Sector" in c)
        
        df_clean = target_df[[ticker_col, sector_col]].copy()
        df_clean.columns = ['Ticker', 'Sector']
        
        # Cleaning: Punten naar streepjes (BRK.B -> BRK-B) en Sector namen opschonen
        df_clean['Ticker'] = df_clean['Ticker'].str.replace('.', '-', regex=False)
        
        # Zorg dat sector namen matchen met onze keys (GICS standaard)
        # Bv: "Consumer Discretionary" -> matcht
        return df_clean

    except Exception as e:
        st.error(f"Fout bij ophalen data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=True)
        # Flatten MultiIndex indien nodig
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            elif 'Close' in data.columns.get_level_values(1):
                 # Soms gooit yfinance het andersom
                try:
                    data = data.xs('Close', level=1, axis=1)
                except:
                    data = data.xs('Close', level=0, axis=1)
        return data
    except:
        return pd.DataFrame()

# --- 3. BEREKENINGEN ---

def calculate_rrg_base(df, benchmark):
    """Basis RRG berekening voor elk aandeel in df t.o.v. benchmark"""
    if benchmark not in df.columns: return pd.DataFrame()
    
    rrg_data = []
    
    # Benchmark Returns voor RS berekening
    bench_series = df[benchmark]
    
    for ticker in df.columns:
        if ticker == benchmark: continue
        
        try:
            # RS Ratio (100 dagen)
            rs = df[ticker] / bench_series
            rs_ma = rs.rolling(100).mean()
            rs_ratio = 100 * (rs / rs_ma)
            
            # RS Momentum (Rate of Change van de Ratio)
            rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
            
            if len(rs_ratio) < 1: continue
            
            curr_r = rs_ratio.iloc[-1]
            curr_m = rs_mom.iloc[-1]
            
            # Kwadrant Bepaling
            if curr_r > 100 and curr_m > 100: status = "1. LEADING"
            elif curr_r < 100 and curr_m > 100: status = "4. IMPROVING"
            elif curr_r < 100 and curr_m < 100: status = "3. LAGGING"
            else: status = "2. WEAKENING"
            
            rrg_data.append({
                'Ticker': ticker,
                'RS-Ratio': curr_r,
                'RS-Momentum': curr_m,
                'Kwadrant': status,
                'Distance': np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            })
        except:
            continue
            
    return pd.DataFrame(rrg_data)

def calculate_matrix(df):
    """Toernooi model"""
    tickers = df.columns.tolist()
    if len(tickers) < 2: return pd.DataFrame()
    
    # 3-Maands Momentum als proxy voor kracht
    mom = (df.iloc[-1] / df.shift(63).iloc[-1]) - 1
    
    results = []
    for t in tickers:
        wins = 0
        for opponent in tickers:
            if t == opponent: continue
            if mom[t] > mom[opponent]: wins += 1
        
        power = (wins / (len(tickers)-1)) * 100
        results.append({'Ticker': t, 'Matrix_Power': power})
        
    return pd.DataFrame(results).sort_values('Matrix_Power', ascending=False)

def get_ai_advice(ticker, key, regime, rank, score):
    if not key: return "Voer API Key in."
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        p = f"Analyseer aandeel {ticker}. Markt Regime: {regime}. Matrix Score: {score}/100. Rank: {rank}. Advies (Kopen/Verkopen/Houden)? Kort."
        return model.generate_content(p).text
    except: return "AI Error."

# --- 4. UI OPBOUW (SIDEBAR EERST VOOR STABILITEIT) ---

st.sidebar.header("⚙️ Instellingen")
sel_market = st.sidebar.selectbox("1. Kies Markt", list(MARKETS.keys()))
market_cfg = MARKETS[sel_market]

# API Key input
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# LOAD MARKET DATA (Cached)
with st.spinner("Markt data laden..."):
    df_constituents = get_market_constituents(sel_market)

if df_constituents.empty:
    st.error("Kon geen data ophalen voor deze markt. Probeer een andere.")
    st.stop()

# SECTOR SELECTOR
sectors = sorted(df_constituents['Sector'].astype(str).unique())
sel_sector = st.sidebar.selectbox("2. Kies Sector", ["Alle Sectoren"] + sectors)

# DISPERSIE (Altijd zichtbaar)
st.sidebar.markdown("---")
st.sidebar.write("📊 **Markt Status**")

# Regime check (Benchmark)
regime_ticker = market_cfg['benchmark']
regime_df = get_price_data([regime_ticker])
if not regime_df.empty:
    sma = regime_df.rolling(200).mean().iloc[-1]
    curr = regime_df.iloc[-1]
    regime = "BULL" if curr > sma else "BEAR"
    color = "green" if regime == "BULL" else "red"
    st.sidebar.markdown(f"Regime: :{color}[**{regime}**]")
else:
    regime = "UNKNOWN"

# KNOP: START ANALYSE (Essentieel voor stabiliteit!)
st.sidebar.markdown("---")
start_btn = st.sidebar.button("🚀 Start Analyse", type="primary")

# --- 5. LOGICA VOOR SESSIE STATE ---

if start_btn:
    # We slaan alles op in session state zodat het blijft staan bij tab-wissels
    st.session_state['active_analysis'] = True
    st.session_state['selected_sector'] = sel_sector
    st.session_state['selected_market'] = sel_market
    
    with st.spinner("Koersen ophalen en rekenen..."):
        # 1. Bepaal welke tickers we nodig hebben
        if sel_sector == "Alle Sectoren":
            # Als "Alle", pakken we de sector ETF's (indien US) of top 30 stocks
            if "USA" in sel_market:
                tickers_to_fetch = list(US_SECTOR_ETFS.values())
            else:
                tickers_to_fetch = df_constituents['Ticker'].head(30).tolist()
        else:
            # SPECIFIEK: Filter op de gekozen sector
            tickers_to_fetch = df_constituents[df_constituents['Sector'] == sel_sector]['Ticker'].tolist()
            
        # Voeg benchmark toe
        tickers_to_fetch.append(market_cfg['benchmark'])
        
        # 2. Download Data
        # Beperk tot max 100 tickers om yfinance niet te choken
        tickers_to_fetch = tickers_to_fetch[:100] 
        df_prices = get_price_data(tickers_to_fetch)
        
        # 3. Bereken RRG & Matrix
        if not df_prices.empty:
            rrg_df = calculate_rrg_base(df_prices, market_cfg['benchmark'])
            
            # Matrix alleen zinnig als we >2 aandelen hebben
            matrix_df = pd.DataFrame()
            if len(tickers_to_fetch) > 2:
                matrix_df = calculate_matrix(df_prices)
                # Voeg prijs toe
                last_p = df_prices.iloc[-1].to_frame('Prijs')
                matrix_df = matrix_df.merge(last_p, left_on='Ticker', right_index=True)
                
            st.session_state['rrg_result'] = rrg_df
            st.session_state['matrix_result'] = matrix_df
            st.session_state['last_update'] = pd.Timestamp.now()
        else:
            st.error("Geen prijsdata gevonden.")

# --- 6. HOOFD SCHERM WEERGAVE ---

st.title(f"Screener: {sel_market}")

if not st.session_state.get('active_analysis'):
    st.info("👈 Selecteer een sector in de zijbalk en klik op 'Start Analyse'.")
else:
    # Data ophalen uit sessie
    rrg_df = st.session_state.get('rrg_result', pd.DataFrame())
    matrix_df = st.session_state.get('matrix_result', pd.DataFrame())
    current_sector = st.session_state.get('selected_sector')

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🚁 RRG Scatter", "📋 Matrix & Ranking", "🤖 AI Advies"])

    # TAB 1: SCATTERPLOT
    with tab1:
        st.subheader(f"Rotatie binnen: {current_sector}")
        if not rrg_df.empty:
            # Dynamische titel
            count = len(rrg_df)
            st.caption(f"Toont {count} instrumenten.")
            
            fig = px.scatter(
                rrg_df, x="RS-Ratio", y="RS-Momentum", 
                color="Kwadrant", text="Ticker", size="Distance",
                color_discrete_map=COLOR_MAP,
                title=f"RRG: {current_sector}", height=700
            )
            # Kruis in het midden
            fig.add_hline(y=100, line_dash="dash", line_color="grey")
            fig.add_vline(x=100, line_dash="dash", line_color="grey") 
            fig.update_traces(textposition='top center', textfont=dict(size=11, family="Arial Black"))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Geen data om te plotten.")

    # TAB 2: MATRIX
    with tab2:
        st.subheader("Sterkte Ranglijst")
        if not matrix_df.empty:
            # Styling functie
            def style_rank(val):
                if val <= 3: return "🟢 BUY"
                elif val <= 6: return "🟡 HOLD"
                else: return "🔴 AVOID"

            matrix_df['Rank'] = range(1, len(matrix_df)+1)
            matrix_df['Advies'] = matrix_df['Rank'].apply(style_rank)
            
            # Opslaan voor AI
            st.session_state['top_pick'] = matrix_df.iloc[0]['Ticker']
            st.session_state['top_score'] = matrix_df.iloc[0]['Matrix_Power']
            
            st.dataframe(
                matrix_df[['Rank', 'Advies', 'Ticker', 'Matrix_Power', 'Prijs']]
                .style.map(lambda x: 'color: green' if "BUY" in x else ('color: red' if "AVOID" in x else 'color: orange'), subset=['Advies'])
                .format({'Matrix_Power': '{:.1f}', 'Prijs': '{:.2f}'}),
                use_container_width=True, height=600
            )
        else:
            st.warning("Onvoldoende data voor matrix berekening.")

    # TAB 3: AI
    with tab3:
        st.subheader("AI Oordeel")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            def_ticker = st.session_state.get('top_pick', '')
            t_in = st.text_input("Ticker:", value=def_ticker)
            if st.button("Vraag Gemini"):
                with st.spinner("Analyseren..."):
                    advies = get_ai_advice(
                        t_in, api_key, regime, 
                        st.session_state.get('top_pick', 0) == t_in, # Is het nummer 1?
                        st.session_state.get('top_score', 0)
                    )
                    st.markdown(f"**Conclusie:**\n\n{advies}")

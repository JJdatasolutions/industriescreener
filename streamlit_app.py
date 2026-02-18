import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import requests
import numpy as np

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener 5.0", layout="wide", page_icon="🧭")

# --- 1. DATA DEFINITIES ---

MARKETS = {
    "🇺🇸 USA - S&P 500 (Large)": {
        "code": "SP500", "benchmark": "SPY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    },
    "🇺🇸 USA - S&P 400 (Mid)": {
        "code": "SP400", "benchmark": "MDY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    },
    "🇪🇺 Europa - Top Selectie": {
        "code": "EU_MIX", "benchmark": "^N100", "type": "static"
    }
}

US_SECTOR_ETFS = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Communication Services': 'XLC', 'Consumer Staples': 'XLP'
}

COLOR_MAP = {
    "1. LEADING": "#006400", "2. WEAKENING": "#FFA500", 
    "3. LAGGING": "#DC143C", "4. IMPROVING": "#90EE90"
}

# --- 2. DATA FUNCTIES ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_key):
    mkt = MARKETS[market_key]
    
    # Europa Static
    if "EU_MIX" in mkt.get("code", ""):
        data = {
            "ASML.AS": "Technology", "UNA.AS": "Staples", "HEIA.AS": "Staples", 
            "SHELL.AS": "Energy", "INGA.AS": "Financials", "DSM.AS": "Materials", 
            "BESI.AS": "Technology", "ADYEN.AS": "Financials", "IMCD.AS": "Materials", 
            "PHI.AS": "Health Care", "KBC.BR": "Financials", "UCB.BR": "Health Care", 
            "SOLB.BR": "Materials", "WDP.BR": "Real Estate", "ELI.BR": "Utilities",
            "MELE.BR": "Industrials", "XIOR.BR": "Real Estate", "ACKB.BR": "Financials"
        }
        return pd.DataFrame(list(data.items()), columns=['Ticker', 'Sector'])

    # USA Wikipedia Scraper (Robust)
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

        # Kolommen normaliseren
        ticker_col = next(c for c in target_df.columns if "Symbol" in str(c))
        sector_col = next(c for c in target_df.columns if "Sector" in str(c))
        
        df_clean = target_df[[ticker_col, sector_col]].copy()
        df_clean.columns = ['Ticker', 'Sector']
        df_clean['Ticker'] = df_clean['Ticker'].str.replace('.', '-', regex=False)
        return df_clean

    except Exception as e:
        st.error(f"Wiki Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', level=1, axis=1)
            else:
                data = data.droplevel(1, axis=1) 
        return data
    except:
        return pd.DataFrame()

# --- 3. BEREKENINGEN ---

def calculate_rrg_metrics(df, benchmark):
    """Berekent RRG data voor visualisatie"""
    if benchmark not in df.columns: return pd.DataFrame()
    rrg_data = []
    bench_series = df[benchmark]
    
    for ticker in df.columns:
        if ticker == benchmark: continue
        try:
            rs = df[ticker] / bench_series
            rs_ma = rs.rolling(100).mean()
            rs_ratio = 100 * (rs / rs_ma)
            rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
            
            if len(rs_ratio) < 1: continue
            curr_r = rs_ratio.iloc[-1]
            curr_m = rs_mom.iloc[-1]
            
            # Naamgeving voor plot
            label = ticker
            for k, v in US_SECTOR_ETFS.items():
                if v == ticker: label = k # Toon 'Technology' ipv 'XLK'
            
            if curr_r > 100 and curr_m > 100: status = "1. LEADING"
            elif curr_r < 100 and curr_m > 100: status = "4. IMPROVING"
            elif curr_r < 100 and curr_m < 100: status = "3. LAGGING"
            else: status = "2. WEAKENING"
            
            rrg_data.append({
                'Ticker': ticker, 'Label': label,
                'RS-Ratio': curr_r, 'RS-Momentum': curr_m,
                'Kwadrant': status, 
                'Distance': np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            })
        except: continue
    return pd.DataFrame(rrg_data)

def calculate_ranking_score(df):
    """Jouw Ranking Script Logica: 1m/3m/6m returns"""
    if df.empty or len(df) < 130: return pd.DataFrame()
    try:
        curr = df.iloc[-1]
        r1 = (curr / df.shift(21).iloc[-1]) - 1
        r3 = (curr / df.shift(63).iloc[-1]) - 1
        r6 = (curr / df.shift(126).iloc[-1]) - 1
        
        # Gewogen score
        score = (r1 * 0.2) + (r3 * 0.4) + (r6 * 0.4)
        
        ranking_data = pd.DataFrame({
            'Prijs': curr,
            '1M %': r1 * 100,
            '3M %': r3 * 100,
            '6M %': r6 * 100,
            'Score': score * 100
        })
        return ranking_data.sort_values('Score', ascending=False).dropna()
    except:
        return pd.DataFrame()

def get_ai_advice(ticker, key, regime, score):
    if not key: return "Voer API Key in."
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        p = f"Aandeel: {ticker}. Regime: {regime}. Technische Score (0-100): {score:.1f}. Geef kort en krachtig koop/verkoop advies. Nederlands."
        return model.generate_content(p).text
    except: return "AI Error."

# --- 4. UI EN LOGICA ---

# SIDEBAR
st.sidebar.header("⚙️ Instellingen")
sel_market = st.sidebar.selectbox("1. Kies Markt", list(MARKETS.keys()))
market_cfg = MARKETS[sel_market]
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# Data laden
with st.spinner("Laden marktdefinitie..."):
    df_constituents = get_market_constituents(sel_market)

if df_constituents.empty:
    st.error("Kon marktdata niet laden.")
    st.stop()

sectors = sorted(df_constituents['Sector'].astype(str).unique())
sel_sector = st.sidebar.selectbox("2. Kies Sector (voor Detail)", sectors)

st.sidebar.markdown("---")
# Regime Check
regime_df = get_price_data([market_cfg['benchmark']])
regime = "UNKNOWN"
if not regime_df.empty:
    try:
        # Scalar fix
        curr = float(regime_df.iloc[-1].item()) if hasattr(regime_df.iloc[-1], 'item') else float(regime_df.iloc[-1])
        sma = float(regime_df.rolling(200).mean().iloc[-1].item()) if hasattr(regime_df.rolling(200).mean().iloc[-1], 'item') else float(regime_df.rolling(200).mean().iloc[-1])
        regime = "BULL" if curr > sma else "BEAR"
        color = "green" if regime == "BULL" else "red"
        st.sidebar.markdown(f"Regime: :{color}[**{regime}**]")
    except: pass

# DE GROTE KNOP
start_btn = st.sidebar.button("🚀 Start Analyse", type="primary")

# --- 5. DATA PROCES ---

if start_btn:
    st.session_state['active'] = True
    st.session_state['sector_name'] = sel_sector
    
    with st.spinner("Bezig met ophalen data..."):
        # STAP 1: SECTOR ROTATIE DATA (HELICOPTER VIEW)
        if "USA" in sel_market:
            # Voor USA gebruiken we de ETF's
            sector_tickers = list(US_SECTOR_ETFS.values())
        else:
            # Voor Europa de top 30 aandelen (als proxy)
            sector_tickers = df_constituents['Ticker'].head(30).tolist()
            
        sector_tickers.append(market_cfg['benchmark'])
        df_sector_prices = get_price_data(sector_tickers)
        
        if not df_sector_prices.empty:
            st.session_state['rrg_market'] = calculate_rrg_metrics(df_sector_prices, market_cfg['benchmark'])
        
        # STAP 2: STOCK DETAIL DATA (DEEP DIVE)
        stock_tickers = df_constituents[df_constituents['Sector'] == sel_sector]['Ticker'].tolist()
        # Voeg benchmark toe voor RRG berekening
        stock_tickers.append(market_cfg['benchmark'])
        stock_tickers = list(set(stock_tickers))[:100] # Limiet
        
        df_stock_prices = get_price_data(stock_tickers)
        
        if not df_stock_prices.empty:
            # A. RRG voor aandelen
            st.session_state['rrg_stocks'] = calculate_rrg_metrics(df_stock_prices, market_cfg['benchmark'])
            # B. Ranking Lijst
            st.session_state['ranking_table'] = calculate_ranking_score(df_stock_prices)
            
            st.session_state['last_update'] = pd.Timestamp.now()

# --- 6. WEERGAVE ---

st.title(f"Market Screener: {sel_market}")

if st.session_state.get('active'):
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🚁 Sector Rotatie", "🔍 Aandelen Detail", "🤖 AI Advies"])
    
    # TAB 1: HELICOPTER VIEW
    with tab1:
        rrg_mkt = st.session_state.get('rrg_market', pd.DataFrame())
        if not rrg_mkt.empty:
            st.subheader("Markt Rotatie")
            st.caption("Waar stroomt het geld heen in de markt?")
            fig = px.scatter(rrg_mkt, x="RS-Ratio", y="RS-Momentum", 
                             color="Kwadrant", text="Label", size="Distance",
                             color_discrete_map=COLOR_MAP, height=650)
            fig.add_hline(y=100, line_dash="dash", line_color="grey")
            fig.add_vline(x=100, line_dash="dash", line_color="grey") 
            fig.update_traces(textposition='top center', textfont=dict(size=12, family="Arial Black"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Geen sector data beschikbaar.")

    # TAB 2: DETAIL VIEW
    with tab2:
        sec_name = st.session_state.get('sector_name')
        st.subheader(f"Deep Dive: {sec_name}")
        
        rrg_stk = st.session_state.get('rrg_stocks', pd.DataFrame())
        rank_tbl = st.session_state.get('ranking_table', pd.DataFrame())
        
        col_graph, col_table = st.columns([3, 2])
        
        with col_graph:
            st.markdown("**1. RRG Scatter**")
            if not rrg_stk.empty:
                fig2 = px.scatter(rrg_stk, x="RS-Ratio", y="RS-Momentum", 
                                 color="Kwadrant", text="Label", size="Distance",
                                 color_discrete_map=COLOR_MAP, height=600)
                fig2.add_hline(y=100, line_dash="dash", line_color="grey")
                fig2.add_vline(x=100, line_dash="dash", line_color="grey")
                fig2.update_traces(textposition='top center', textfont=dict(size=11))
                st.plotly_chart(fig2, use_container_width=True)
        
        with col_table:
            st.markdown("**2. Score Lijst**")
            if not rank_tbl.empty:
                # Opslaan voor AI
                st.session_state['top_pick'] = rank_tbl.index[0]
                st.session_state['top_score'] = rank_tbl.iloc[0]['Score']
                
                st.dataframe(
                    rank_tbl[['Score', 'Prijs', '3M %']]
                    .style.background_gradient(subset=['Score'], cmap='RdYlGn')
                    .format("{:.1f}"),
                    use_container_width=True, height=600
                )
    
    # TAB 3: AI
    with tab3:
        st.subheader("Portfolio Manager")
        col1, col2 = st.columns([1, 2])
        with col1:
            pick = st.session_state.get('top_pick', '')
            score = st.session_state.get('top_score', 0)
            st.info(f"Top Pick uit lijst: **{pick}** (Score: {score:.1f})")
            
            if st.button("Vraag AI Advies"):
                with st.spinner("Analyseren..."):
                    advies = get_ai_advice(pick, api_key, regime, score)
                    st.markdown(advies)

else:
    st.info("👈 Selecteer een markt en sector in de zijbalk en klik op 'Start Analyse'.")

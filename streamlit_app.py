import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import requests
import numpy as np

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Screener 4.1 (Fixed)", layout="wide", page_icon="🛡️")

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
    "🇺🇸 USA - S&P 600 (Small)": {
        "code": "SP600", "benchmark": "IJR", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
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

# --- 2. DATA OPHALEN ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_key):
    mkt = MARKETS[market_key]
    
    if "EU_MIX" in mkt.get("code", ""):
        data = {
            "ASML.AS": "Technology", "UNA.AS": "Consumer Staples", "HEIA.AS": "Consumer Staples", 
            "SHELL.AS": "Energy", "INGA.AS": "Financials", "DSM.AS": "Materials", 
            "BESI.AS": "Technology", "ADYEN.AS": "Financials", "IMCD.AS": "Materials", 
            "PHI.AS": "Health Care", "KBC.BR": "Financials", "UCB.BR": "Health Care", 
            "SOLB.BR": "Materials", "WDP.BR": "Real Estate", "ELI.BR": "Utilities"
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
            # Probeer 'Close' te pakken op verschillende niveaus
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', level=1, axis=1)
            else:
                # Fallback: pak level 0 als er geen 'Close' is (soms stuurt yf alleen data)
                data = data.droplevel(1, axis=1) 
        return data
    except:
        return pd.DataFrame()

# --- 3. BEREKENINGEN ---

def calculate_rrg_base(df, benchmark):
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
            
            if curr_r > 100 and curr_m > 100: status = "1. LEADING"
            elif curr_r < 100 and curr_m > 100: status = "4. IMPROVING"
            elif curr_r < 100 and curr_m < 100: status = "3. LAGGING"
            else: status = "2. WEAKENING"
            
            rrg_data.append({
                'Ticker': ticker, 'RS-Ratio': curr_r, 'RS-Momentum': curr_m,
                'Kwadrant': status, 'Distance': np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            })
        except: continue
    return pd.DataFrame(rrg_data)

def calculate_matrix(df):
    tickers = df.columns.tolist()
    if len(tickers) < 2: return pd.DataFrame()
    
    mom = (df.iloc[-1] / df.shift(63).iloc[-1]) - 1
    results = []
    for t in tickers:
        wins = 0
        for opponent in tickers:
            if t == opponent: continue
            if mom[t] > mom[opponent]: wins += 1
        
        power = (wins / (len(tickers)-1)) * 100 if len(tickers) > 1 else 0
        results.append({'Ticker': t, 'Matrix_Power': power})
        
    return pd.DataFrame(results).sort_values('Matrix_Power', ascending=False)

def get_ai_advice(ticker, key, regime, is_top, score):
    if not key: return "Voer API Key in."
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        p = f"Analyseer aandeel {ticker}. Markt Regime: {regime}. Matrix Score: {score}/100. Is Top Pick: {is_top}. Advies (Kopen/Verkopen/Houden)? Max 3 zinnen."
        return model.generate_content(p).text
    except: return "AI Error."

# --- 4. SIDEBAR ---

st.sidebar.header("⚙️ Instellingen")
sel_market = st.sidebar.selectbox("1. Kies Markt", list(MARKETS.keys()))
market_cfg = MARKETS[sel_market]
api_key = st.sidebar.text_input("Gemini API Key", type="password")

with st.spinner("Markt data laden..."):
    df_constituents = get_market_constituents(sel_market)

if df_constituents.empty:
    st.error("⚠️ Kon marktdata niet laden.")
    st.stop()

sectors = sorted(df_constituents['Sector'].astype(str).unique())
sel_sector = st.sidebar.selectbox("2. Kies Sector", ["Alle Sectoren"] + sectors)

st.sidebar.markdown("---")
st.sidebar.write("📊 **Markt Status**")

# --- FIX: REGIME BEREKENING ---
regime_ticker = market_cfg['benchmark']
regime_df = get_price_data([regime_ticker])

regime = "UNKNOWN"
if not regime_df.empty:
    try:
        # Hier zat de fout. We forceren nu expliciet scalars (enkele getallen) met .item()
        # Als regime_df 1 kolom heeft, is .iloc[-1] een Series. .item() maakt er een float van.
        if isinstance(regime_df, pd.Series):
             curr = regime_df.iloc[-1]
             sma = regime_df.rolling(200).mean().iloc[-1]
        else:
             curr = regime_df.iloc[-1].item()
             sma = regime_df.rolling(200).mean().iloc[-1].item()
             
        regime = "BULL" if curr > sma else "BEAR"
        color = "green" if regime == "BULL" else "red"
        st.sidebar.markdown(f"Regime: :{color}[**{regime}**]")
    except Exception as e:
        st.sidebar.error(f"Regime error: {e}")

# KNOP
st.sidebar.markdown("---")
start_btn = st.sidebar.button("🚀 Start Analyse", type="primary")

# --- 5. LOGICA ---

if start_btn:
    st.session_state['active_analysis'] = True
    st.session_state['selected_sector'] = sel_sector
    
    with st.spinner("Rekenmachine loopt..."):
        if sel_sector == "Alle Sectoren":
             if "USA" in sel_market:
                tickers = list(US_SECTOR_ETFS.values())
             else:
                tickers = df_constituents['Ticker'].head(30).tolist()
        else:
            tickers = df_constituents[df_constituents['Sector'] == sel_sector]['Ticker'].tolist()
            
        tickers.append(market_cfg['benchmark'])
        tickers = list(set(tickers))[:100] # Max 100 unieke tickers voor snelheid
        
        df_prices = get_price_data(tickers)
        
        if not df_prices.empty and len(df_prices.columns) > 1:
            rrg_df = calculate_rrg_base(df_prices, market_cfg['benchmark'])
            
            matrix_df = pd.DataFrame()
            if len(tickers) > 2:
                matrix_df = calculate_matrix(df_prices)
                last_p = df_prices.iloc[-1].to_frame('Prijs')
                matrix_df = matrix_df.merge(last_p, left_on='Ticker', right_index=True)
                
            st.session_state['rrg_result'] = rrg_df
            st.session_state['matrix_result'] = matrix_df
        else:
            st.error("Onvoldoende prijsdata opgehaald.")

# --- 6. WEERGAVE ---

st.title(f"Screener: {sel_market}")

if st.session_state.get('active_analysis'):
    rrg_df = st.session_state.get('rrg_result', pd.DataFrame())
    matrix_df = st.session_state.get('matrix_result', pd.DataFrame())
    sec = st.session_state.get('selected_sector')

    tab1, tab2, tab3 = st.tabs(["🚁 RRG Scatter", "📋 Matrix", "🤖 AI"])

    with tab1:
        st.subheader(f"Rotatie: {sec}")
        if not rrg_df.empty:
            fig = px.scatter(rrg_df, x="RS-Ratio", y="RS-Momentum", 
                             color="Kwadrant", text="Ticker", size="Distance",
                             color_discrete_map=COLOR_MAP, height=650, title=f"RRG ({len(rrg_df)} assets)")
            fig.add_hline(y=100, line_dash="dash", line_color="grey")
            fig.add_vline(x=100, line_dash="dash", line_color="grey") 
            fig.update_traces(textposition='top center', textfont=dict(size=11, family="Arial Black"))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Ranking")
        if not matrix_df.empty:
            matrix_df['Rank'] = range(1, len(matrix_df)+1)
            matrix_df['Advies'] = matrix_df['Rank'].apply(lambda x: "🟢 BUY" if x<=3 else ("🔴 SELL" if x>6 else "🟡 HOLD"))
            
            st.session_state['top_pick'] = matrix_df.iloc[0]['Ticker']
            st.session_state['top_score'] = matrix_df.iloc[0]['Matrix_Power']
            
            st.dataframe(matrix_df[['Rank', 'Advies', 'Ticker', 'Matrix_Power', 'Prijs']], use_container_width=True, height=600)

    with tab3:
        st.subheader("AI Manager")
        col1, col2 = st.columns([1,2])
        with col1:
            if st.button("Vraag Advies"):
                with st.spinner("Thinking..."):
                    advies = get_ai_advice(st.session_state.get('top_pick'), api_key, regime, True, st.session_state.get('top_score'))
                    st.markdown(advies)
else:
    st.info("👈 Kies een sector en klik op Start Analyse.")

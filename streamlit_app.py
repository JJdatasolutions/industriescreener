import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import requests
import numpy as np

# --- CONFIGURATIE ---
st.set_page_config(page_title="Industry Screener 3.1 (Fixed)", layout="wide", page_icon="⚔️")

# --- 1. DATA DEFINITIES ---

MARKETS = {
    "🇺🇸 USA - S&P 500": {
        "code": "SP500", "benchmark": "SPY", 
        "wiki_url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    },
    "🇺🇸 USA - S&P 400 (MidCap)": {
        "code": "SP400", "benchmark": "MDY", 
        "wiki_url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    },
    "🇪🇺 Europa - AEX/BEL20 Mix": {
        "code": "EU_MIX", "benchmark": "^N100", "type": "static"
    }
}

US_SECTOR_ETFS = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Comm Services': 'XLC', 'Staples': 'XLP'
}

# Omgekeerde mapping voor de grafiek labels (Code -> Naam)
ETF_TO_NAME = {v: k for k, v in US_SECTOR_ETFS.items()}

STATIC_EU_DATA = {
    "ASML.AS": "Technology", "UNA.AS": "Staples", "HEIA.AS": "Staples", "SHELL.AS": "Energy", 
    "AD.AS": "Staples", "INGA.AS": "Financials", "DSM.AS": "Materials", "BESI.AS": "Technology", 
    "ADYEN.AS": "Financials", "IMCD.AS": "Materials", "ASM.AS": "Technology", "PHI.AS": "Health Care", 
    "KBC.BR": "Financials", "UCB.BR": "Health Care", "SOLB.BR": "Materials", "ACKB.BR": "Financials", 
    "ARGX.BR": "Health Care", "UMI.BR": "Materials", "GBL.BR": "Financials", "WDP.BR": "Real Estate",
    "ELI.BR": "Utilities", "VGP.BR": "Real Estate", "MELE.BR": "Industrials", "XIOR.BR": "Real Estate"
}

COLOR_MAP = {
    "1. LEADING": "#006400", "2. WEAKENING": "#FFA500", 
    "3. LAGGING": "#DC143C", "4. IMPROVING": "#90EE90"
}

# --- 2. DATA FUNCTIES ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_code):
    if "USA" in market_code:
        current_market = next(v for k, v in MARKETS.items() if v['code'] == MARKETS[k]['code'] and k.startswith("🇺🇸"))
        for k, v in MARKETS.items():
            if v['code'] == market_code: 
                url = v['wiki_url']
                break
        
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            df = pd.read_html(requests.get(url, headers=headers).text)[0]
            sym_col = 'Symbol' if 'Symbol' in df.columns else 'Ticker Symbol'
            sec_col = 'GICS Sector' if 'GICS Sector' in df.columns else 'Sector'
            df_res = df[[sym_col, sec_col]].copy()
            df_res.columns = ['Ticker', 'Sector']
            df_res['Ticker'] = df_res['Ticker'].str.replace('.', '-', regex=False)
            return df_res
        except:
            return pd.DataFrame()
    else:
        return pd.DataFrame(list(STATIC_EU_DATA.items()), columns=['Ticker', 'Sector'])

def clean_yfinance_data(df):
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            df = df['Close']
        elif 'Close' in df.columns.get_level_values(1):
            df = df.xs('Close', level=1, axis=1)
    return df

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=True)
        return clean_yfinance_data(data)
    except:
        return pd.DataFrame()

def calculate_market_regime(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        df = clean_yfinance_data(df)
        if isinstance(df, pd.DataFrame): df = df.iloc[:, 0]
        
        curr = df.iloc[-1]
        sma = df.rolling(200).mean().iloc[-1]
        return "BULL" if curr > sma else "BEAR", df
    except:
        return "UNKNOWN", pd.Series()

def calculate_dispersion(tickers):
    try:
        data = get_price_data(tickers)
        if data.empty: return 0, "Onbekend"
        returns = (data.iloc[-1] / data.shift(63).iloc[-1]) - 1
        std_dev = returns.std() * 100 
        
        if std_dev < 5: status = "Laag (Kuddegedrag)"
        elif std_dev < 10: status = "Neutraal"
        else: status = "Hoog (Stock Pickers Markt)"
        return std_dev, status
    except:
        return 0, "Error"

def calculate_rs_matrix_score(df_prices):
    tickers = df_prices.columns.tolist()
    n = len(tickers)
    if n < 2: return pd.DataFrame()

    momentum = (df_prices.iloc[-1] / df_prices.shift(63).iloc[-1]) - 1
    scores = {t: 0 for t in tickers}
    
    for t1 in tickers:
        for t2 in tickers:
            if t1 == t2: continue
            if momentum[t1] > momentum[t2]:
                scores[t1] += 1
                
    max_score = n - 1
    results = []
    for t, score in scores.items():
        results.append({
            'Ticker': t,
            'Matrix_Wins': score,
            'Matrix_Power': (score / max_score) * 100 if max_score > 0 else 0
        })
        
    return pd.DataFrame(results).sort_values('Matrix_Power', ascending=False)

def calculate_rrg_metrics(df, benchmark):
    if benchmark not in df.columns: return pd.DataFrame()
    rrg_list = []
    for col in df.columns:
        if col == benchmark: continue
        try:
            rs = df[col] / df[benchmark]
            rs_ma = rs.rolling(100).mean()
            rs_ratio = 100 * (rs / rs_ma)
            rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
            if len(rs_ratio) < 2: continue
            
            curr_r = rs_ratio.iloc[-1]
            curr_m = rs_mom.iloc[-1]
            
            if curr_r > 100 and curr_m > 100: status = "1. LEADING"
            elif curr_r < 100 and curr_m > 100: status = "4. IMPROVING"
            elif curr_r < 100 and curr_m < 100: status = "3. LAGGING"
            else: status = "2. WEAKENING"
            
            # Label bepalen (Naam ipv Ticker indien mogelijk)
            label_name = ETF_TO_NAME.get(col, col)
            
            rrg_list.append({
                'Ticker': col, 
                'Naam': label_name, # Hier pakken we de volledige naam
                'RS-Ratio': curr_r, 'RS-Momentum': curr_m,
                'Kwadrant': status, 'Distance': np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            })
        except: continue
    return pd.DataFrame(rrg_list)

def get_actionable_advice(ticker, key, market_info, metrics):
    if not key: return "⚠️ API Key vereist."
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-pro') 
        prompt = f"""
        Je bent een Hedge Fund Manager. Geef een BINDEND advies voor {ticker}.
        MARKTDATA: Regime: {market_info['regime']}, Dispersie: {market_info['dispersion']}
        AANDEEL DATA: Matrix Score (0-100): {metrics['matrix_score']:.1f}, Rank: #{metrics['rank']}
        LOGICA: BEAR Market = Terughoudend. Top 3 Rank = Kopen. Rank 6+ = Verkopen.
        VRAAG: Geef eindoordeel: "STERK KOPEN", "KOPEN", "HOUDEN", "VERKOPEN". Max 5 regels.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

# --- 3. UI LAYOUT ---

# SIDEBAR
st.sidebar.title("⚔️ Screener 3.1")
sel_market_key = st.sidebar.selectbox("Market", list(MARKETS.keys()))
cur_mkt = MARKETS[sel_market_key]
api_key = st.sidebar.text_input("Gemini API Key", type="password")

st.sidebar.markdown("---")
regime, regime_data = calculate_market_regime(cur_mkt['benchmark'])
if regime == "BULL":
    st.sidebar.success(f"🚦 FILTER: {regime}\n(Long posities toegestaan)")
else:
    st.sidebar.error(f"🚦 FILTER: {regime}\n(Cash is King / Hedging)")

st.sidebar.markdown("---")
st.sidebar.write("📊 **Dispersie Meter**")
if "USA" in sel_market_key:
    disp_tickers = list(US_SECTOR_ETFS.values())
else:
    disp_tickers = list(STATIC_EU_DATA.keys())[:10]
    
disp_val, disp_status = calculate_dispersion(disp_tickers)
st.sidebar.progress(min(disp_val/20, 1.0))
st.sidebar.caption(f"Score: {disp_val:.1f}% - {disp_status}")

# MAIN CONTENT
st.title(f"Tactical Industry Screener: {cur_mkt['code']}")

with st.spinner("Loading Assets..."):
    constituents = get_market_constituents(cur_mkt['code'])
    all_tickers = constituents['Ticker'].tolist()

# TABBLADEN: Info is terug!
tab0, tab1, tab2, tab3 = st.tabs(["ℹ️ Info", "🚁 Helicopter View (RRG)", "⚔️ Matrix & Signals", "🤖 AI Final Call"])

# TAB 0: INFO
with tab0:
    st.markdown("""
    ### ℹ️ Hoe werkt dit dashboard?
    
    **1. Markt Regime (Sidebar)**
    * 🟢 **BULL:** De markt zit boven de 200-dagen lijn. Veilig om te kopen.
    * 🔴 **BEAR:** De markt zit eronder. Pas op, verhoog cash positie.

    **2. RRG (Relative Rotation Graph)**
    * Dit toont de rotatie van sectoren of aandelen t.o.v. de benchmark.
    * 🟢 **LEADING:** Sterke trend + Momentum (Kopen)
    * 🟠 **WEAKENING:** Trend zwakt af (Winst nemen?)
    * 🔴 **LAGGING:** Slechte trend + Momentum (Afblijven/Short)
    * 🍏 **IMPROVING:** Trend draait bij (Kansen zoeken)
    
    **3. Matrix Score (Tab 2)**
    * Elk aandeel vecht 1-tegen-1 met alle andere aandelen in de sector.
    * Een score van **100** betekent dat het aandeel sterker is dan *alle* andere.
    * **Strategie:** Koop de Top 3, verkoop alles buiten de Top 6.
    """)

# TAB 1: RRG
with tab1:
    st.subheader("Sector Rotatie")
    if "USA" in sel_market_key:
        plot_tickers = list(US_SECTOR_ETFS.values())
    else:
        plot_tickers = all_tickers[:30] 
        
    prices = get_price_data(plot_tickers + [cur_mkt['benchmark']])
    if not prices.empty:
        rrg = calculate_rrg_metrics(prices, cur_mkt['benchmark'])
        if not rrg.empty:
            fig = px.scatter(rrg, x="RS-Ratio", y="RS-Momentum", color="Kwadrant", 
                             text="Naam", # AANGEPAST: Nu de Naam ipv Ticker
                             size="Distance", color_discrete_map=COLOR_MAP,
                             title="Relative Rotation Graph", height=650)
            
            # Betere styling voor leesbaarheid
            fig.update_traces(textposition='top center', textfont=dict(family="Arial Black", size=12))
            fig.add_hline(y=100, line_dash="dash", line_color="grey")
            fig.add_vline(x=100, line_dash="dash", line_color="grey")
            
            st.plotly_chart(fig, use_container_width=True)

# TAB 2: MATRIX
with tab2:
    st.subheader("🎯 Sector Drill-down & Buy/Sell Logic")
    sectors = sorted(constituents['Sector'].unique().tolist())
    sel_sector = st.selectbox("Selecteer Sector voor Analyse:", sectors)
    
    if st.button("Start Toernooi Analyse"):
        sector_tickers = constituents[constituents['Sector'] == sel_sector]['Ticker'].tolist()
        with st.spinner(f"vechten {len(sector_tickers)} aandelen het uit..."):
            sec_prices = get_price_data(sector_tickers)
            if not sec_prices.empty and len(sector_tickers) > 1:
                matrix_df = calculate_rs_matrix_score(sec_prices)
                last_prices = sec_prices.iloc[-1].to_frame(name="Prijs")
                final_df = matrix_df.merge(last_prices, left_on="Ticker", right_index=True)
                
                def determine_signal(rank):
                    if rank <= 3: return "🟢 BUY"
                    elif rank <= 5: return "🟡 HOLD"
                    else: return "🔴 SELL / AVOID"
                
                final_df['Rank'] = range(1, len(final_df) + 1)
                final_df['Actie'] = final_df['Rank'].apply(determine_signal)
                
                if not final_df.empty:
                    st.session_state['top_stock'] = final_df.iloc[0]['Ticker']
                    st.session_state['top_score'] = final_df.iloc[0]['Matrix_Power']
                    st.session_state['top_rank'] = 1
                    st.session_state['total_peers'] = len(final_df)
                
                st.markdown(f"### 🏆 Winnaars in {sel_sector}")
                st.dataframe(
                    final_df[['Rank', 'Actie', 'Ticker', 'Matrix_Power', 'Prijs']]
                    .style.map(lambda x: 'color: green; font-weight: bold' if x == "🟢 BUY" else 
                               ('color: orange' if x == "🟡 HOLD" else 'color: red'), subset=['Actie'])
                    .format({'Matrix_Power': '{:.1f}', 'Prijs': '{:.2f}'}),
                    use_container_width=True, height=600
                )

# TAB 3: AI
with tab3:
    st.subheader("👨‍💼 De Portfolio Manager")
    col_in, col_res = st.columns([1, 2])
    with col_in:
        t_input = st.text_input("Ticker om te beoordelen:", value=st.session_state.get('top_stock', ''))
        if st.button("Vraag Oordeel"):
            metrics = {
                'matrix_score': st.session_state.get('top_score', 50),
                'rank': st.session_state.get('top_rank', 5)
            }
            mkt_info = {'regime': regime, 'dispersion': disp_status}
            with st.spinner("Analyseren..."):
                advice = get_actionable_advice(t_input, api_key, mkt_info, metrics)
                st.markdown(advice)

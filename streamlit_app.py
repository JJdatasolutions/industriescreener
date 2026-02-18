import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import requests
import numpy as np

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Industry Screener 3.3", layout="wide", page_icon="⚔️")

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

ETF_TO_NAME = {v: k for k, v in US_SECTOR_ETFS.items()}

STATIC_EU_DATA = {
    "ASML.AS": "Technology", "UNA.AS": "Staples", "HEIA.AS": "Staples", "SHELL.AS": "Energy", 
    "AD.AS": "Staples", "INGA.AS": "Financials", "DSM.AS": "Materials", "BESI.AS": "Technology", 
    "ADYEN.AS": "Financials", "IMCD.AS": "Materials", "ASM.AS": "Technology", "PHI.AS": "Health Care", 
    "KBC.BR": "Financials", "UCB.BR": "Health Care", "SOLB.BR": "Materials", "ACKB.BR": "Financials", 
    "ARGX.BR": "Health Care", "UMI.BR": "Materials", "GBL.BR": "Financials", "WDP.BR": "Real Estate",
    "ELI.BR": "Utilities", "VGP.BR": "Real Estate", "MELE.BR": "Industrials", "XIOR.BR": "Real Estate",
    "TNET.BR": "Comm Services", "PROX.BR": "Comm Services", "AED.BR": "Real Estate", "COFB.BR": "Real Estate"
}

COLOR_MAP = {
    "1. LEADING": "#006400", "2. WEAKENING": "#FFA500", 
    "3. LAGGING": "#DC143C", "4. IMPROVING": "#90EE90"
}

# --- 2. DATA FUNCTIES ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_code):
    """
    SLIMME SCRAPER UPDATE:
    Zoekt door alle tabellen op de wiki pagina naar de juiste tabel 
    met 'Symbol' en 'Sector' kolommen, in plaats van blind tabel [0] te pakken.
    """
    if "USA" in market_code:
        # Zoek juiste URL
        current_market = next(v for k, v in MARKETS.items() if v['code'] == MARKETS[k]['code'] and k.startswith("🇺🇸"))
        for k, v in MARKETS.items():
            if v['code'] == market_code: 
                url = v['wiki_url']
                break
        
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            # Haal ALLE tabellen op
            dfs = pd.read_html(requests.get(url, headers=headers).text)
            
            df_target = pd.DataFrame()
            
            # Loop door tabellen om de juiste te vinden
            for df in dfs:
                # S&P 500/400 hebben meestal 'Symbol' en 'GICS Sector'
                if ('Symbol' in df.columns or 'Ticker Symbol' in df.columns) and \
                   ('GICS Sector' in df.columns or 'Sector' in df.columns):
                    df_target = df
                    break
            
            if df_target.empty:
                st.error("Kon geen aandelentabel vinden op Wikipedia.")
                return pd.DataFrame()

            # Normaliseer kolomnamen
            sym_col = 'Symbol' if 'Symbol' in df_target.columns else 'Ticker Symbol'
            sec_col = 'GICS Sector' if 'GICS Sector' in df_target.columns else 'Sector'
            
            df_res = df_target[[sym_col, sec_col]].copy()
            df_res.columns = ['Ticker', 'Sector']
            
            # Clean tickers (BRK.B -> BRK-B)
            df_res['Ticker'] = df_res['Ticker'].str.replace('.', '-', regex=False)
            return df_res
            
        except Exception as e:
            st.error(f"Fout bij ophalen Wikipedia: {e}")
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
        # Batch download
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
            
            label_name = ETF_TO_NAME.get(col, col)
            rrg_list.append({
                'Ticker': col, 'Naam': label_name,
                'RS-Ratio': curr_r, 'RS-Momentum': curr_m,
                'Kwadrant': status, 'Distance': np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            })
        except: continue
    return pd.DataFrame(rrg_list)

def calculate_sector_rrg_metrics(df_prices):
    if df_prices.empty or len(df_prices) < 65: return pd.DataFrame()
    try:
        ret_long = (df_prices.iloc[-1] - df_prices.shift(60).iloc[-1]) / df_prices.shift(60).iloc[-1] * 100
        ret_short = (df_prices.iloc[-1] - df_prices.shift(10).iloc[-1]) / df_prices.shift(10).iloc[-1] * 100

        avg_long = ret_long.mean()
        avg_short = ret_short.mean()

        metrics = pd.DataFrame({'Ret_Long': ret_long, 'Ret_Short': ret_short})
        metrics['X_Trend'] = metrics['Ret_Long'] - avg_long
        metrics['Y_Momentum'] = metrics['Ret_Short'] - avg_short

        def get_quadrant(row):
            x, y = row['X_Trend'], row['Y_Momentum']
            if x > 0 and y > 0: return "1. LEADING"
            if x > 0 and y < 0: return "2. WEAKENING"
            if x < 0 and y < 0: return "3. LAGGING"
            if x < 0 and y > 0: return "4. IMPROVING"
            return "Unknown"

        metrics['Kwadrant'] = metrics.apply(get_quadrant, axis=1)
        metrics['Ticker'] = metrics.index
        metrics['Distance'] = np.sqrt(metrics['X_Trend']**2 + metrics['Y_Momentum']**2) * 3
        return metrics.dropna()
    except:
        return pd.DataFrame()

def calculate_rs_matrix_score(df_prices):
    tickers = df_prices.columns.tolist()
    n = len(tickers)
    if n < 2: return pd.DataFrame()

    momentum = (df_prices.iloc[-1] / df_prices.shift(63).iloc[-1]) - 1
    scores = {t: 0 for t in tickers}
    for t1 in tickers:
        for t2 in tickers:
            if t1 == t2: continue
            if momentum[t1] > momentum[t2]: scores[t1] += 1
                
    max_score = n - 1
    results = []
    for t, score in scores.items():
        results.append({
            'Ticker': t, 'Matrix_Wins': score,
            'Matrix_Power': (score / max_score) * 100 if max_score > 0 else 0
        })
    return pd.DataFrame(results).sort_values('Matrix_Power', ascending=False)

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
st.sidebar.title("⚔️ Screener 3.3")
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
with st.sidebar.expander("❓ Wat is dit?"):
    st.write("Dispersie meet het verschil tussen winnaars en verliezers. Hoger is beter voor stock picking.")

if "USA" in sel_market_key:
    disp_tickers = list(US_SECTOR_ETFS.values())
else:
    disp_tickers = list(STATIC_EU_DATA.keys())[:10]
    
disp_val, disp_status = calculate_dispersion(disp_tickers)
st.sidebar.progress(min(disp_val/20, 1.0))
st.sidebar.caption(f"Score: {disp_val:.1f}% - {disp_status}")

# MAIN CONTENT
st.title(f"Tactical Industry Screener: {cur_mkt['code']}")

with st.spinner("Laden van markt data..."):
    constituents = get_market_constituents(cur_mkt['code'])
    
    # EXTRA CHECK: Toon hoeveel aandelen we hebben gevonden
    if constituents.empty:
        st.error("⚠️ Fout: Geen aandelen gevonden. Check de Wikipedia verbinding.")
        all_tickers = []
    else:
        all_tickers = constituents['Ticker'].tolist()

# TABBLADEN
tab0, tab1, tab2, tab3 = st.tabs(["ℹ️ Info", "🚁 Helicopter View (RRG)", "⚔️ Matrix & Signals", "🤖 AI Final Call"])

# TAB 0: INFO
with tab0:
    st.markdown("""
    ### ℹ️ Handleiding
    **1. Markt Regime (Sidebar):** Groen = Kopen mag. Rood = Pas op.
    **2. RRG (Tab 1):** Waar stroomt het geld heen? Zoek naar 'Leading' (Groen) of 'Improving' (Blauw).
    **3. Matrix (Tab 2):** De ultieme stock-picker. Aandelen vechten 1-tegen-1.
    **4. AI (Tab 3):** Twijfel je? Vraag de AI om een eindoordeel.
    """)

# TAB 1: RRG (MARKET LEVEL)
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
                             text="Naam", size="Distance", color_discrete_map=COLOR_MAP,
                             title="Relative Rotation Graph (Markt)", height=650)
            fig.update_traces(textposition='top center', textfont=dict(family="Arial Black", size=12))
            fig.add_hline(y=100, line_dash="dash", line_color="grey")
            fig.add_vline(x=100, line_dash="dash", line_color="grey") 
            st.plotly_chart(fig, use_container_width=True)

# TAB 2: MATRIX & SCATTER (STOCK LEVEL)
with tab2:
    st.subheader("🎯 Sector Analyse")
    
    # 1. Selectie
    if not constituents.empty:
        sectors = sorted(constituents['Sector'].unique().tolist())
        sel_sector = st.selectbox("Selecteer Sector:", sectors)
        
        # 2. Knop & Session State
        if st.button("Start Analyse"):
            # RESET data bij nieuwe klik om "Something else" te voorkomen
            if 'data_matrix' in st.session_state: del st.session_state['data_matrix']
            if 'data_scatter' in st.session_state: del st.session_state['data_scatter']

            sector_tickers = constituents[constituents['Sector'] == sel_sector]['Ticker'].tolist()
            
            st.info(f"⏳ Data ophalen voor {len(sector_tickers)} aandelen in {sel_sector}...")
            
            with st.spinner(f"Bezig met berekenen..."):
                sec_prices = get_price_data(sector_tickers)
                
                if not sec_prices.empty and len(sector_tickers) > 1:
                    # A. Scatter Data
                    scatter_df = calculate_sector_rrg_metrics(sec_prices)
                    
                    # B. Matrix Data
                    matrix_df = calculate_rs_matrix_score(sec_prices)
                    last_prices = sec_prices.iloc[-1].to_frame(name="Prijs")
                    final_matrix = matrix_df.merge(last_prices, left_on="Ticker", right_index=True)
                    
                    # Signalen
                    final_matrix['Rank'] = range(1, len(final_matrix) + 1)
                    final_matrix['Actie'] = final_matrix['Rank'].apply(lambda r: "🟢 BUY" if r <= 3 else ("🟡 HOLD" if r <= 5 else "🔴 SELL"))
                    
                    # OPSLAAN
                    st.session_state['data_scatter'] = scatter_df
                    st.session_state['data_matrix'] = final_matrix
                    st.session_state['data_sector'] = sel_sector
                else:
                    st.error("Te weinig data beschikbaar voor deze sector.")
    else:
        st.warning("Geen sectoren geladen.")

    # 3. Weergave (Persistent)
    if 'data_matrix' in st.session_state and st.session_state.get('data_sector') == sel_sector:
        
        # Weergave Check
        aantal = len(st.session_state['data_matrix'])
        st.success(f"✅ Analyse voltooid: {aantal} aandelen geanalyseerd.")

        # A. SCATTER
        st.markdown(f"#### 1. Visueel: {sel_sector} Scatter")
        scat_data = st.session_state.get('data_scatter', pd.DataFrame())
        if not scat_data.empty:
            fig_sec = px.scatter(
                scat_data, x="X_Trend", y="Y_Momentum",
                color="Kwadrant", text="Ticker", size="Distance",
                color_discrete_map=COLOR_MAP,
                title=f"Positionering binnen {sel_sector}", height=600
            )
            fig_sec.update_traces(textposition='top center', textfont=dict(size=12, family="Arial Black"))
            fig_sec.add_hline(y=0, line_color="black")
            fig_sec.add_vline(x=0, line_color="black")
            st.plotly_chart(fig_sec, use_container_width=True)
            
        # B. MATRIX
        st.markdown(f"#### 2. Lijst: {sel_sector} Matrix Score")
        mat_data = st.session_state['data_matrix']
        
        st.session_state['top_stock'] = mat_data.iloc[0]['Ticker']
        st.session_state['top_score'] = mat_data.iloc[0]['Matrix_Power']
        st.session_state['top_rank'] = 1
        
        st.dataframe(
            mat_data[['Rank', 'Actie', 'Ticker', 'Matrix_Power', 'Prijs']]
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
        def_ticker = st.session_state.get('top_stock', '')
        t_input = st.text_input("Ticker:", value=def_ticker)
        
        if st.button("Vraag Advies"):
            metrics = {
                'matrix_score': st.session_state.get('top_score', 50),
                'rank': st.session_state.get('top_rank', 5)
            }
            mkt_info = {'regime': regime, 'dispersion': disp_status}
            with st.spinner("Analyseren..."):
                advice = get_actionable_advice(t_input, api_key, mkt_info, metrics)
                st.markdown(advice)

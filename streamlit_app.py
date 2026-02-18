import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import requests
import numpy as np

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener (Ultimate)", layout="wide", page_icon="🧭")

# --- 1. DATA DEFINITIES ---

MARKETS = {
    "🇺🇸 USA - S&P 500 (LargeCap)": {
        "code": "SP500", 
        "benchmark": "SPY", 
        "wiki_url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    },
    "🇺🇸 USA - S&P 400 (MidCap)": {
        "code": "SP400", 
        "benchmark": "MDY", 
        "wiki_url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    },
    "🇳🇱 Nederland - AEX/AMX": {
        "code": "NL", 
        "benchmark": "^AEX", 
        "type": "static"
    },
    "🇧🇪 België - BEL 20": {
        "code": "BE", 
        "benchmark": "^BFX", 
        "type": "static"
    }
}

US_SECTOR_ETFS = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Comm Services': 'XLC', 'Staples': 'XLP'
}

STATIC_EU_DATA = {
    "ASML.AS": "Technology", "UNA.AS": "Staples", "HEIA.AS": "Staples", "SHELL.AS": "Energy", 
    "AD.AS": "Staples", "INGA.AS": "Financials", "DSM.AS": "Materials", "ABN.AS": "Financials", 
    "KPN.AS": "Comm Services", "WKL.AS": "Industrials", "RAND.AS": "Industrials", "NN.AS": "Financials", 
    "BESI.AS": "Technology", "ADYEN.AS": "Financials", "IMCD.AS": "Materials", "ASM.AS": "Technology",
    "PHI.AS": "Health Care", "MT.AS": "Materials", "WDP.AS": "Real Estate", "JDEP.AS": "Staples",
    "KBC.BR": "Financials", "UCB.BR": "Health Care", "SOLB.BR": "Materials", "ACKB.BR": "Financials", 
    "ARGX.BR": "Health Care", "UMI.BR": "Materials", "GBL.BR": "Financials", "COFB.BR": "Real Estate", 
    "WDP.BR": "Real Estate", "ELI.BR": "Utilities", "AED.BR": "Real Estate", "ABI.BR": "Staples",
    "LOTB.BR": "Staples", "APAM.BR": "Materials", "VGP.BR": "Real Estate", "MELE.BR": "Industrials",
    "XIOR.BR": "Real Estate", "TNET.BR": "Comm Services", "PROX.BR": "Comm Services"
}

COLOR_MAP = {
    "1. LEADING": "#006400",   
    "2. WEAKENING": "#FFA500", 
    "3. LAGGING": "#DC143C",   
    "4. IMPROVING": "#90EE90"  
}

# --- 2. DATA FUNCTIES ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_code):
    if market_code in ["SP500", "SP400"]:
        current_market = next(v for k, v in MARKETS.items() if v['code'] == market_code)
        url = current_market['wiki_url']
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            df = pd.read_html(requests.get(url, headers=headers).text)[0]
            
            sym_col = 'Symbol' if 'Symbol' in df.columns else 'Ticker Symbol'
            sec_col = 'GICS Sector' if 'GICS Sector' in df.columns else 'Sector'
            
            df_res = df[[sym_col, sec_col]].copy()
            df_res.columns = ['Ticker', 'Sector']
            df_res['Ticker'] = df_res['Ticker'].str.replace('.', '-', regex=False)
            return df_res
        except Exception as e:
            st.error(f"Fout bij ophalen Wikipedia: {e}")
            return pd.DataFrame()
    else:
        suffix = ".AS" if market_code == "NL" else ".BR"
        filtered = {k: v for k, v in STATIC_EU_DATA.items() if k.endswith(suffix)}
        return pd.DataFrame(list(filtered.items()), columns=['Ticker', 'Sector'])

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
        # Fix voor nieuwe yfinance die soms MultiIndex teruggeeft bij 1 ticker
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                 return data['Close']
        if 'Close' in data.columns:
             return data['Close']
        return data
    except:
        return pd.DataFrame()

def calculate_market_regime_data(ticker):
    """Haalt data op voor de grafiek in de sidebar."""
    try:
        data = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if 'Close' in data.columns:
            df = data[['Close']].copy()
        else:
            df = data.copy() # Fallback
            
        # Rename column for clarity if needed
        if isinstance(df, pd.Series): df = df.to_frame(name='Close')
        if df.shape[1] > 1 and 'Close' not in df.columns: df = df.iloc[:, 0].to_frame(name='Close') # Pak eerste kolom
            
        df['SMA200'] = df['Close'].rolling(200).mean()
        return df
    except Exception as e:
        return pd.DataFrame()

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
            
            if pd.notna(curr_r) and pd.notna(curr_m):
                if curr_r > 100 and curr_m > 100: status = "1. LEADING"
                elif curr_r < 100 and curr_m > 100: status = "4. IMPROVING"
                elif curr_r < 100 and curr_m < 100: status = "3. LAGGING"
                else: status = "2. WEAKENING"
                
                distance = np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
                
                name = col
                for k, v in US_SECTOR_ETFS.items():
                    if v == col: name = k
                    
                rrg_list.append({
                    'Ticker': col, 'Naam': name,
                    'RS-Ratio': curr_r, 'RS-Momentum': curr_m,
                    'Kwadrant': status, 'Distance': distance
                })
        except:
            continue
            
    return pd.DataFrame(rrg_list)

def calculate_sector_rrg_metrics(df_prices):
    if df_prices.empty: return pd.DataFrame()

    try:
        ret_long = (df_prices.iloc[-1] - df_prices.shift(60).iloc[-1]) / df_prices.shift(60).iloc[-1] * 100
        ret_short = (df_prices.iloc[-1] - df_prices.shift(10).iloc[-1]) / df_prices.shift(10).iloc[-1] * 100

        avg_long = ret_long.mean()
        avg_short = ret_short.mean()

        metrics = pd.DataFrame({
            'Ret_Long': ret_long,
            'Ret_Short': ret_short
        })

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
        metrics['Distance'] = np.sqrt(metrics['X_Trend']**2 + metrics['Y_Momentum']**2)

        return metrics.dropna()
    except:
        return pd.DataFrame()

def calculate_ranking(df):
    """
    Berekent ranking en zorgt voor een schoon DataFrame om merge errors te voorkomen.
    """
    if df.empty or len(df) < 130: return pd.DataFrame()
    
    try:
        curr = df.iloc[-1]
        
        r1 = (curr / df.shift(21).iloc[-1]) - 1
        r3 = (curr / df.shift(63).iloc[-1]) - 1
        r6 = (curr / df.shift(126).iloc[-1]) - 1
        
        score = (r1 * 0.2) + (r3 * 0.4) + (r6 * 0.4)
        
        ranking_data = pd.DataFrame({
            'Prijs': curr,
            '1M %': r1 * 100,
            '3M %': r3 * 100,
            '6M %': r6 * 100,
            'Score': score * 100
        })
        
        # BELANGRIJK: Zorg dat Ticker een kolom is en geen index
        ranking_data.index.name = 'Ticker'
        ranking_data = ranking_data.reset_index()
        
        return ranking_data.sort_values('Score', ascending=False).dropna()
    except Exception as e:
        st.error(f"Fout in ranking berekening: {e}")
        return pd.DataFrame()

def get_gemini_advice(ticker, key, market_status, sector):
    if not key: return "⚠️ Voer eerst een Google Gemini sleutel in."
    
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-pro') 
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        info = stock.info
        
        if hist.empty: return "Geen data."
        trend = "Stijgend" if hist['Close'].iloc[-1] > hist['Close'].iloc[0] else "Dalend"
        
        prompt = f"""
        Jij bent een expert beleggingsanalist.
        Analyseer: {ticker} ({info.get('longName', 'Onbekend')}).
        CONTEXT: Markt: {market_status}. Sector: {sector}. Trend (3m): {trend}.
        VRAAG: Geef kritisch advies (Kopen/Houden/Verkopen) in het Nederlands. Max 100 woorden.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Fout: {e}"

# --- 3. UI OPBOUW ---

# SIDEBAR
st.sidebar.header("⚙️ 1. Kies Universum")
sel_market_key = st.sidebar.selectbox("Beschikbare Markten", list(MARKETS.keys()))
current_market = MARKETS[sel_market_key]
market_code = current_market['code']
benchmark_ticker = current_market['benchmark']

gemini_key = st.sidebar.text_input("Google Gemini Key", type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("🚦 Markt Regime")

# Markt data ophalen en plotten
regime_df = calculate_market_regime_data(benchmark_ticker)
if not regime_df.empty:
    curr_price = regime_df['Close'].iloc[-1]
    sma_200 = regime_df['SMA200'].iloc[-1]
    
    if curr_price > sma_200:
        st.sidebar.success(f"✅ BULL MARKET\n\nKoers ligt {((curr_price/sma_200)-1)*100:.1f}% boven 200 SMA.")
    else:
        st.sidebar.error(f"⛔ BEAR MARKET\n\nKoers ligt {((1-(curr_price/sma_200)))*100:.1f}% onder 200 SMA.")
        
    # Mini chart in sidebar
    st.sidebar.line_chart(regime_df.tail(252), color=["#00FF00" if curr_price > sma_200 else "#FF0000", "#FFFFFF"], height=150)
else:
    st.sidebar.warning("Geen benchmark data beschikbaar.")

st.title(f"🧭 Market Screener: {market_code}")

# DATA LADEN
with st.spinner(f"Laden van {market_code} data..."):
    constituents_df = get_market_constituents(market_code)
    # Zorg voor clean dataframe voor merge later
    constituents_df = constituents_df.reset_index(drop=True)
    all_tickers = constituents_df['Ticker'].tolist()

# TABS
tab0, tab1, tab2, tab3 = st.tabs(["ℹ️ Info", "📊 Market RRG", "🏆 Sector Drill-down", "🤖 AI Advies"])

# TAB 0: INFO
with tab0:
    st.markdown("""
    ### 📊 Legende RRG
    * 🟢 **LEADING:** Sterke trend + Momentum (Kopen)
    * 🟠 **WEAKENING:** Trend zwakt af (Winst nemen?)
    * 🔴 **LAGGING:** Slechte trend + Momentum (Afblijven)
    * 🍏 **IMPROVING:** Trend draait bij (Kansen)
    """)

# TAB 1: RRG (MARKET LEVEL)
with tab1:
    st.subheader(f"Global View: {market_code}")
    
    if "USA" in sel_market_key:
        tickers_to_plot = list(US_SECTOR_ETFS.values())
        st.caption("Overzicht van US Sectoren.")
    else:
        tickers_to_plot = all_tickers
        st.caption(f"Overzicht van individuele aandelen in {market_code}.")

    raw_data = get_price_data(tickers_to_plot + [benchmark_ticker])
    
    if not raw_data.empty:
        rrg_metrics = calculate_rrg_metrics(raw_data, benchmark_ticker)
        
        if not rrg_metrics.empty:
            fig = px.scatter(
                rrg_metrics, x="RS-Ratio", y="RS-Momentum",
                color="Kwadrant", text="Naam", size="Distance",
                color_discrete_map=COLOR_MAP,
                title=f"RRG: {market_code}", height=650,
                hover_data=["Kwadrant"]
            )
            # Layout verbetering voor leesbaarheid
            fig.update_traces(textposition='top center', textfont=dict(size=12, color='black', family="Arial Black"))
            fig.update_layout(shapes=[
                dict(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="gray", dash="dash")),
                dict(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="gray", dash="dash"))
            ])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Te weinig data voor RRG.")

# TAB 2: FILTER & SECTOR RRG (STOCK LEVEL)
with tab2:
    st.subheader(f"🔍 Drill-down binnen {market_code}")
    st.caption("Selecteer hieronder een sector om de specifieke aandelen uit het gekozen universum te analyseren.")
    
    # 1. Selectie
    available_sectors = sorted(constituents_df['Sector'].unique().tolist())
    col_filter, col_dummy = st.columns([1,2])
    with col_filter:
        selected_sector = st.selectbox("📂 Kies Sector:", ["Alle Sectoren"] + available_sectors)
    
    # 2. Filtering
    if selected_sector != "Alle Sectoren":
        filtered_tickers = constituents_df[constituents_df['Sector'] == selected_sector]['Ticker'].tolist()
    else:
        filtered_tickers = all_tickers
        
    if st.button("🚀 Start Sector Analyse"):
        with st.spinner(f"Analyseren van {len(filtered_tickers)} aandelen..."):
            stock_data = get_price_data(filtered_tickers)
            
            if not stock_data.empty:
                # --- SECTOR RRG VISUALISATIE ---
                if selected_sector != "Alle Sectoren" and len(filtered_tickers) > 2:
                    st.markdown("---")
                    st.markdown(f"### 🎯 Positie t.o.v. {selected_sector} sector")
                    
                    sector_rrg_df = calculate_sector_rrg_metrics(stock_data)
                    
                    if not sector_rrg_df.empty:
                        fig_sec = px.scatter(
                            sector_rrg_df, x="X_Trend", y="Y_Momentum",
                            color="Kwadrant", text="Ticker", size="Distance",
                            color_discrete_map=COLOR_MAP,
                            title=f"Relative Rotation: {selected_sector}",
                            labels={"X_Trend": "Trend (vs Sector)", "Y_Momentum": "Momentum (vs Sector)"},
                            height=650
                        )
                        
                        # VERBETERINGEN LEESBAARHEID
                        fig_sec.update_traces(
                            textposition='top center', 
                            marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')), # Randje om bol
                            textfont=dict(size=12, color='white') # Witte tekst in dark mode, of zwart in light
                        )
                        
                        # Assen kruis
                        fig_sec.add_hline(y=0, line_color="white", line_width=1)
                        fig_sec.add_vline(x=0, line_color="white", line_width=1)
                        
                        st.plotly_chart(fig_sec, use_container_width=True)

                # --- STANDAARD TABEL ---
                st.markdown("---")
                st.subheader("🏆 Score Lijst")
                rank_df = calculate_ranking(stock_data)
                
                if not rank_df.empty:
                    st.session_state['top_pick'] = rank_df.iloc[0]['Ticker']
                    
                    # VEILIGE MERGE (Bugfix applied)
                    # We forceren dat 'Ticker' een kolom is in beide
                    display = pd.merge(rank_df, constituents_df, on='Ticker', how='left')
                    
                    st.dataframe(
                        display.style.format({'Prijs': '{:.2f}', 'Score': '{:.1f}'})
                        .background_gradient(subset=['Score'], cmap='RdYlGn'),
                        use_container_width=True, height=600
                    )

# TAB 3: AI
with tab3:
    st.subheader("🤖 Gemini Advies")
    col1, col2 = st.columns([1, 2])
    with col1:
        def_val = st.session_state.get('top_pick', "")
        u_ticker = st.text_input("Ticker", value=def_val)
        
        # Sector ophalen
        u_sector = "Onbekend"
        if not constituents_df.empty:
            row = constituents_df[constituents_df['Ticker'] == u_ticker]
            if not row.empty: u_sector = row.iloc[0]['Sector']
            
        if st.button("Vraag Advies"):
            with st.spinner("Gemini analyseert..."):
                advies = get_gemini_advice(u_ticker, gemini_key, "Zie Sidebar", u_sector)
                st.markdown(advies)

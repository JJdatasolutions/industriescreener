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

# MARKETS
MARKETS = {
    "🇺🇸 USA - S&P 500 (LargeCap)": {
        "code": "SP500", 
        "benchmark": "SPY", 
        "wiki_url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "type": "wiki"
    },
    "🇺🇸 USA - S&P 400 (MidCap)": {
        "code": "SP400", 
        "benchmark": "MDY", 
        "wiki_url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "type": "wiki"
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

# US SECTOR ETFS (Voor US RRG visualisatie)
US_SECTOR_ETFS = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Comm Services': 'XLC', 'Staples': 'XLP'
}

# STATIC DATA EU (Om sector filters mogelijk te maken zonder Yahoo data)
STATIC_EU_DATA = {
    # NL
    "ASML.AS": "Technology", "UNA.AS": "Staples", "HEIA.AS": "Staples", "SHELL.AS": "Energy", 
    "AD.AS": "Staples", "INGA.AS": "Financials", "DSM.AS": "Materials", "ABN.AS": "Financials", 
    "KPN.AS": "Comm Services", "WKL.AS": "Industrials", "RAND.AS": "Industrials", "NN.AS": "Financials", 
    "BESI.AS": "Technology", "ADYEN.AS": "Financials", "IMCD.AS": "Materials", "ASM.AS": "Technology",
    "PHI.AS": "Health Care", "MT.AS": "Materials", "WDP.AS": "Real Estate", "JDEP.AS": "Staples",
    # BE
    "KBC.BR": "Financials", "UCB.BR": "Health Care", "SOLB.BR": "Materials", "ACKB.BR": "Financials", 
    "ARGX.BR": "Health Care", "UMI.BR": "Materials", "GBL.BR": "Financials", "COFB.BR": "Real Estate", 
    "WDP.BR": "Real Estate", "ELI.BR": "Utilities", "AED.BR": "Real Estate", "ABI.BR": "Staples",
    "LOTB.BR": "Staples", "APAM.BR": "Materials", "VGP.BR": "Real Estate", "MELE.BR": "Industrials",
    "XIOR.BR": "Real Estate", "TNET.BR": "Comm Services", "PROX.BR": "Comm Services"
}

# KLEURENPALET (Eenduidig)
COLOR_MAP = {
    "1. LEADING": "#006400",   # Donkergroen
    "2. WEAKENING": "#FFA500", # Oranje
    "3. LAGGING": "#DC143C",   # Rood
    "4. IMPROVING": "#90EE90"  # Lichtgroen
}

# --- 2. DATA FUNCTIES ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_code):
    """Haalt tickers en sectoren op."""
    if market_code in ["SP500", "SP400"]:
        # WIKI SCRAPING VOOR US
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
            st.error(f"Fout bij ophalen Wikipedia data: {e}")
            return pd.DataFrame()
    else:
        # STATIC DATA VOOR EU
        suffix = ".AS" if market_code == "NL" else ".BR"
        filtered = {k: v for k, v in STATIC_EU_DATA.items() if k.endswith(suffix)}
        return pd.DataFrame(list(filtered.items()), columns=['Ticker', 'Sector'])

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        # 2 jaar historie voor lange termijn trends
        return yf.download(tickers, period="2y", progress=False, auto_adjust=True)['Close']
    except:
        return pd.DataFrame()

def calculate_market_regime(ticker):
    """Check of Index boven 200 SMA zit."""
    try:
        data = yf.download(ticker, period="2y", progress=False, auto_adjust=True)['Close']
        if data.empty: return "UNKNOWN"
        curr = data.iloc[-1]
        sma = data.rolling(200).mean().iloc[-1]
        return "BULL" if curr > sma else "BEAR"
    except:
        return "UNKNOWN"

def calculate_rrg_metrics(df, benchmark):
    """Berekent RRG data en wijst de juiste kleuren/fases toe."""
    if benchmark not in df.columns: return pd.DataFrame()
    
    rrg_list = []
    
    for col in df.columns:
        if col == benchmark: continue
        
        rs = df[col] / df[benchmark]
        rs_ma = rs.rolling(100).mean()
        
        rs_ratio = 100 * (rs / rs_ma)
        rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
        
        if len(rs_ratio) < 2: continue
        
        curr_r = rs_ratio.iloc[-1]
        curr_m = rs_mom.iloc[-1]
        
        if pd.notna(curr_r) and pd.notna(curr_m):
            # Fase bepaling met nummers voor sortering in legende
            if curr_r > 100 and curr_m > 100: status = "1. LEADING"
            elif curr_r < 100 and curr_m > 100: status = "4. IMPROVING" # Let op: Improving zit linksboven
            elif curr_r < 100 and curr_m < 100: status = "3. LAGGING"
            else: status = "2. WEAKENING"
            
            # Distance
            distance = np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            
            # Naam
            name = col
            for k, v in US_SECTOR_ETFS.items():
                if v == col: name = k
                
            rrg_list.append({
                'Ticker': col, 'Naam': name,
                'RS-Ratio': curr_r, 'RS-Momentum': curr_m,
                'Kwadrant': status, 'Distance': distance
            })
            
    return pd.DataFrame(rrg_list)

def calculate_ranking(df):
    if df.empty or len(df) < 130: return pd.DataFrame()
    curr = df.iloc[-1]
    
    r1 = (curr / df.shift(21).iloc[-1]) - 1
    r3 = (curr / df.shift(63).iloc[-1]) - 1
    r6 = (curr / df.shift(126).iloc[-1]) - 1
    
    score = (r1 * 0.2) + (r3 * 0.4) + (r6 * 0.4)
    
    return pd.DataFrame({
        'Ticker': df.columns, 'Prijs': curr,
        '1M %': r1 * 100, '3M %': r3 * 100, '6M %': r6 * 100,
        'Score': score * 100
    }).sort_values('Score', ascending=False).dropna()

def get_gemini_advice(ticker, key, market_status, sector):
    """Vraagt advies aan Google Gemini 1.5 Pro (Best available)."""
    if not key: return "⚠️ Voer eerst een Google Gemini sleutel in."
    
    try:
        genai.configure(api_key=key)
        # We gebruiken 1.5 Pro voor de beste kwaliteit ("2.5" ervaring)
        model = genai.GenerativeModel('gemini-1.5-pro') 
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        info = stock.info
        
        if hist.empty: return "Geen data."
        trend = "Stijgend" if hist['Close'].iloc[-1] > hist['Close'].iloc[0] else "Dalend"
        
        prompt = f"""
        Jij bent een expert beleggingsanalist (Quant & Fundamenteel).
        Analyseer: {ticker} ({info.get('longName', 'Onbekend')}).
        
        CONTEXT:
        1. Markt Regime: {market_status}.
        2. Sector: {sector}.
        3. Trend (3m): {trend}.
        
        VRAAG:
        Geef een scherp en kritisch advies in het Nederlands.
        - Is het aandeel fundamenteel koopwaardig?
        - Bevestigt de technische analyse dit?
        - CONCLUSIE: KOPEN, HOUDEN of VERKOPEN?
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Fout: {e}"

# --- 3. UI OPBOUW ---

# SIDEBAR
st.sidebar.header("⚙️ Instellingen")
sel_market_key = st.sidebar.selectbox("Kies Universum", list(MARKETS.keys()))
current_market = MARKETS[sel_market_key]
market_code = current_market['code']
benchmark_ticker = current_market['benchmark']

gemini_key = st.sidebar.text_input("Google Gemini Key", type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("🚦 Markt Regime")
regime_status = calculate_market_regime(benchmark_ticker)
if regime_status == "BULL":
    st.sidebar.success("✅ VEILIG (Bull)\nIndex > 200 SMA")
else:
    st.sidebar.error("⛔ GEVAAR (Bear)\nIndex < 200 SMA")

st.title(f"🧭 Market Screener: {market_code}")

# DATA LADEN
with st.spinner("Laden data..."):
    constituents_df = get_market_constituents(market_code)
    all_tickers = constituents_df['Ticker'].tolist()

# TABS
tab0, tab1, tab2, tab3 = st.tabs(["ℹ️ Info", "📊 Sector RRG", "🏆 Filter & Ranking", "🤖 AI Advies"])

# TAB 0: INFO
with tab0:
    st.markdown("""
    ### 📊 Legende Sector Analyse
    De kleuren in Tabblad 1 (RRG) hebben een vaste betekenis:
    * 🟢 **LEADING (Donkergroen):** Sterke trend + Positief momentum. (Kopen)
    * 🟠 **WEAKENING (Oranje):** Sterke trend, maar verliest snelheid. (Winst nemen?)
    * 🔴 **LAGGING (Rood):** Slechte trend + Negatief momentum. (Vermijden)
    * 🍏 **IMPROVING (Lichtgroen):** Trend draait bij naar positief. (Kansen)
    """)

# TAB 1: RRG
with tab1:
    st.subheader(f"Relative Rotation Graph ({benchmark_ticker})")
    
    # Bepaal wat we plotten
    if "USA" in sel_market_key:
        # Voor USA plotten we de Sector ETFs voor overzicht
        tickers_to_plot = list(US_SECTOR_ETFS.values())
        st.caption("We tonen hier de US Sector ETFs voor een helder overzicht.")
    else:
        # Voor EU plotten we de individuele aandelen
        tickers_to_plot = all_tickers
        st.caption(f"We tonen alle aandelen uit de {market_code}.")

    raw_data = get_price_data(tickers_to_plot + [benchmark_ticker])
    
    if not raw_data.empty:
        rrg_metrics = calculate_rrg_metrics(raw_data, benchmark_ticker)
        
        if not rrg_metrics.empty:
            # PLOT MET VASTE KLEUREN
            fig = px.scatter(
                rrg_metrics, x="RS-Ratio", y="RS-Momentum",
                color="Kwadrant", # Gebruik altijd de Kwadrant voor kleur
                text="Naam", size="Distance",
                color_discrete_map=COLOR_MAP, # Forceer jouw kleuren
                title=f"RRG: {market_code}", height=700,
                hover_data=["Kwadrant"]
            )
            
            # Layout
            fig.add_hline(y=100, line_dash="dash", line_color="gray")
            fig.add_vline(x=100, line_dash="dash", line_color="gray")
            fig.add_shape(type="rect", x0=100, y0=100, x1=115, y1=115, fillcolor="green", opacity=0.1, line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Te weinig data voor RRG.")

# TAB 2: FILTER & RANKING
with tab2:
    st.subheader("🔍 Stock Screener")
    
    # UNIVERSELE SECTOR FILTER
    available_sectors = sorted(constituents_df['Sector'].unique().tolist())
    selected_sector = st.selectbox("📂 Filter op Sector:", ["Alle Sectoren"] + available_sectors)
    
    if selected_sector != "Alle Sectoren":
        filtered_tickers = constituents_df[constituents_df['Sector'] == selected_sector]['Ticker'].tolist()
    else:
        filtered_tickers = all_tickers
        
    if st.button("🚀 Start Analyse"):
        with st.spinner("Analyseren..."):
            stock_data = get_price_data(filtered_tickers)
            if not stock_data.empty:
                rank_df = calculate_ranking(stock_data)
                if not rank_df.empty:
                    st.session_state['top_pick'] = rank_df.iloc[0]['Ticker']
                    
                    # Merge sector info terug
                    display = rank_df.merge(constituents_df, on='Ticker', how='left')
                    
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
            with st.spinner("Gemini 1.5 Pro analyseert..."):
                advies = get_gemini_advice(u_ticker, gemini_key, regime_status, u_sector)
                st.markdown(advies)

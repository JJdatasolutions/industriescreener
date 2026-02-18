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

# STATIC DATA EU
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

# KLEURENPALET
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
        suffix = ".AS" if market_code == "NL" else ".BR"
        filtered = {k: v for k, v in STATIC_EU_DATA.items() if k.endswith(suffix)}
        return pd.DataFrame(list(filtered.items()), columns=['Ticker', 'Sector'])

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        return yf.download(tickers, period="2y", progress=False, auto_adjust=True)['Close']
    except:
        return pd.DataFrame()

def calculate_market_regime(ticker):
    try:
        data = yf.download(ticker, period="2y", progress=False, auto_adjust=True)['Close']
        if data.empty: return "UNKNOWN"
        curr = data.iloc[-1]
        sma = data.rolling(200).mean().iloc[-1]
        return "BULL" if curr > sma else "BEAR"
    except:
        return "UNKNOWN"

def calculate_rrg_metrics(df, benchmark):
    """Berekent klassieke RRG data (voor Tab 1)."""
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
            
    return pd.DataFrame(rrg_list)

def calculate_sector_rrg_metrics(df_prices):
    """
    NIEUW: Berekent RRG op basis van RELATIEVE performance t.o.v. de SECTOR (het gemiddelde).
    """
    if df_prices.empty: return pd.DataFrame()

    # 1. Rendementen berekenen
    # Lange termijn trend (bv. 60 dagen / 3 maanden)
    ret_long = (df_prices.iloc[-1] - df_prices.shift(60).iloc[-1]) / df_prices.shift(60).iloc[-1] * 100
    
    # Korte termijn momentum (bv. 10 dagen / 2 weken)
    ret_short = (df_prices.iloc[-1] - df_prices.shift(10).iloc[-1]) / df_prices.shift(10).iloc[-1] * 100

    # 2. Benchmark bepalen (het gemiddelde van deze groep)
    avg_long = ret_long.mean()
    avg_short = ret_short.mean()

    # 3. Dataframe bouwen
    metrics = pd.DataFrame({
        'Ret_Long': ret_long,
        'Ret_Short': ret_short
    })

    # 4. Relatieve scores berekenen (centreren rond 0)
    metrics['X_Trend'] = metrics['Ret_Long'] - avg_long
    metrics['Y_Momentum'] = metrics['Ret_Short'] - avg_short

    # 5. Kwadranten toewijzen
    def get_quadrant(row):
        x, y = row['X_Trend'], row['Y_Momentum']
        if x > 0 and y > 0: return "1. LEADING"
        if x > 0 and y < 0: return "2. WEAKENING"
        if x < 0 and y < 0: return "3. LAGGING"
        if x < 0 and y > 0: return "4. IMPROVING"
        return "Unknown"

    metrics['Kwadrant'] = metrics.apply(get_quadrant, axis=1)
    metrics['Ticker'] = metrics.index
    metrics['Distance'] = np.sqrt(metrics['X_Trend']**2 + metrics['Y_Momentum']**2) # Voor bolgrootte

    return metrics.dropna()

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
tab0, tab1, tab2, tab3 = st.tabs(["ℹ️ Info", "📊 Market RRG", "🏆 Sector Drill-down", "🤖 AI Advies"])

# TAB 0: INFO
with tab0:
    st.markdown("""
    ### 📊 Legende RRG (Relative Rotation Graph)
    Of je nu naar de hele markt kijkt of naar één sector, de kleuren betekenen altijd hetzelfde:
    
    * 🟢 **LEADING (Rechtsboven):** **Kopen**. Dit aandeel verslaat de rest en het momentum is sterk.
    * 🟠 **WEAKENING (Rechtsonder):** **Oppassen**. Het aandeel is nog steeds sterk, maar verliest snelheid.
    * 🔴 **LAGGING (Linksonder):** **Verkopen**. Dit aandeel doet het slechter dan de groep en zakt verder weg.
    * 🍏 **IMPROVING (Linksboven):** **Opletten**. Het aandeel was slecht, maar begint nu kracht te tonen.
    """)
    

# TAB 1: RRG (MARKET LEVEL)
with tab1:
    st.subheader(f"Market Level RRG ({benchmark_ticker})")
    
    if "USA" in sel_market_key:
        tickers_to_plot = list(US_SECTOR_ETFS.values())
        st.caption("Overzicht van de US Sectoren.")
    else:
        tickers_to_plot = all_tickers
        st.caption(f"Overzicht van alle aandelen in {market_code}.")

    raw_data = get_price_data(tickers_to_plot + [benchmark_ticker])
    
    if not raw_data.empty:
        rrg_metrics = calculate_rrg_metrics(raw_data, benchmark_ticker)
        
        if not rrg_metrics.empty:
            fig = px.scatter(
                rrg_metrics, x="RS-Ratio", y="RS-Momentum",
                color="Kwadrant", text="Naam", size="Distance",
                color_discrete_map=COLOR_MAP,
                title=f"RRG: {market_code}", height=600,
                hover_data=["Kwadrant"]
            )
            # Layout Shapes
            fig.add_hline(y=100, line_dash="dash", line_color="gray")
            fig.add_vline(x=100, line_dash="dash", line_color="gray")
            fig.add_shape(type="rect", x0=100, y0=100, x1=115, y1=115, fillcolor="green", opacity=0.1, line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Te weinig data voor RRG.")

# TAB 2: FILTER & SECTOR RRG (STOCK LEVEL)
with tab2:
    st.subheader("🔍 Stock Screener & Sector Drill-down")
    
    # 1. Selectie
    available_sectors = sorted(constituents_df['Sector'].unique().tolist())
    col_filter, col_dummy = st.columns([1,2])
    with col_filter:
        selected_sector = st.selectbox("📂 Filter op Sector:", ["Alle Sectoren"] + available_sectors)
    
    # 2. Filtering
    if selected_sector != "Alle Sectoren":
        filtered_tickers = constituents_df[constituents_df['Sector'] == selected_sector]['Ticker'].tolist()
        st.success(f"Geselecteerd: {len(filtered_tickers)} aandelen in {selected_sector}")
    else:
        filtered_tickers = all_tickers
        st.info(f"Totaal: {len(filtered_tickers)} aandelen (Kies een sector voor gedetailleerde RRG)")
        
    if st.button("🚀 Start Analyse"):
        with st.spinner(f"Analyseren van {selected_sector}..."):
            stock_data = get_price_data(filtered_tickers)
            
            if not stock_data.empty:
                # --- NIEUW: SECTOR RRG VISUALISATIE ---
                # Alleen tonen als er een specifieke sector is gekozen (anders wordt de plot te druk/onleesbaar)
                if selected_sector != "Alle Sectoren" and len(filtered_tickers) > 2:
                    st.markdown("---")
                    st.subheader(f"📊 Sector RRG: {selected_sector}")
                    
                    # Berekenen van de relatieve data
                    sector_rrg_df = calculate_sector_rrg_metrics(stock_data)
                    
                    if not sector_rrg_df.empty:
                        # Uitleg uitklapper
                        with st.expander("ℹ️ Hoe lees ik deze grafiek?"):
                            st.markdown(f"""
                            Deze grafiek toont hoe elk aandeel presteert **ten opzichte van het gemiddelde van de {selected_sector} sector**.
                            * **Centrum (0,0):** Het sectorgemiddelde.
                            * **Rechts:** Sterker dan de sectorgenoten.
                            * **Boven:** Momentum neemt toe.
                            """)

                        # Plotten
                        fig_sec = px.scatter(
                            sector_rrg_df, 
                            x="X_Trend", 
                            y="Y_Momentum",
                            color="Kwadrant", 
                            text="Ticker", 
                            size="Distance",
                            color_discrete_map=COLOR_MAP,
                            title=f"Relative Rotation binnen {selected_sector}",
                            labels={"X_Trend": "Trend (vs Sector)", "Y_Momentum": "Momentum (vs Sector)"},
                            height=600
                        )
                        
                        # Assen kruis
                        fig_sec.add_hline(y=0, line_color="black", line_width=1)
                        fig_sec.add_vline(x=0, line_color="black", line_width=1)
                        # Groene zone
                        max_x = sector_rrg_df['X_Trend'].max()
                        max_y = sector_rrg_df['Y_Momentum'].max()
                        fig_sec.add_shape(type="rect", x0=0, y0=0, x1=max_x, y1=max_y, fillcolor="green", opacity=0.1, line_width=0)
                        
                        st.plotly_chart(fig_sec, use_container_width=True)

                # --- STANDAARD TABEL ---
                st.markdown("---")
                st.subheader(f"🏆 Ranking Lijst ({selected_sector})")
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

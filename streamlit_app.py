import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import requests
import numpy as np

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener (Ultimate)", layout="wide", page_icon="🧭")

# --- DATA CONSTANTEN & MAPPINGS ---

# 1. MARKETS DEFINITIE
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

# 2. US SECTOR ETFS (Alleen voor US RRG)
US_SECTOR_ETFS = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Comm Services': 'XLC', 'Staples': 'XLP'
}

# 3. STATIC DATA VOOR NL/BE (Zzodat sector filter werkt zonder Wikipedia)
# Dit is nodig omdat Yahoo Finance vaak geen sector info geeft voor Europese tickers
STATIC_EU_DATA = {
    # NEDERLAND
    "ASML.AS": "Technology", "UNA.AS": "Staples", "HEIA.AS": "Staples", "SHELL.AS": "Energy", 
    "AD.AS": "Staples", "INGA.AS": "Financials", "DSM.AS": "Materials", "ABN.AS": "Financials", 
    "KPN.AS": "Comm Services", "WKL.AS": "Industrials", "RAND.AS": "Industrials", "NN.AS": "Financials", 
    "BESI.AS": "Technology", "ADYEN.AS": "Financials", "IMCD.AS": "Materials", "ASM.AS": "Technology",
    "PHI.AS": "Health Care", "MT.AS": "Materials",
    # BELGIE
    "KBC.BR": "Financials", "UCB.BR": "Health Care", "SOLB.BR": "Materials", "ACKB.BR": "Financials", 
    "ARGX.BR": "Health Care", "UMI.BR": "Materials", "GBL.BR": "Financials", "COFB.BR": "Real Estate", 
    "WDP.BR": "Real Estate", "ELI.BR": "Utilities", "AED.BR": "Real Estate", "ABI.BR": "Staples",
    "LOTB.BR": "Staples", "APAM.BR": "Materials", "VGP.BR": "Real Estate", "MELE.BR": "Industrials"
}

# --- FUNCTIES ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_code):
    """
    Haalt de lijst van aandelen op.
    - Voor US: Scrapet Wikipedia.
    - Voor NL/BE: Gebruikt de statische lijst met hardcoded sectoren.
    """
    tickers = []
    df_result = pd.DataFrame(columns=['Ticker', 'Sector'])

    if market_code in ["SP500", "SP400"]:
        # WIKI SCRAPING
        url = MARKETS[next(k for k, v in MARKETS.items() if v['code'] == market_code)]['wiki_url']
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            df = pd.read_html(requests.get(url, headers=headers).text)[0]
            
            # Kolomnamen normaliseren
            sym_col = 'Symbol' if 'Symbol' in df.columns else 'Ticker Symbol'
            sec_col = 'GICS Sector' if 'GICS Sector' in df.columns else 'Sector'
            
            df_result = df[[sym_col, sec_col]].copy()
            df_result.columns = ['Ticker', 'Sector']
            df_result['Ticker'] = df_result['Ticker'].str.replace('.', '-', regex=False)
            
        except Exception as e:
            st.error(f"Wikipedia error: {e}")
            return pd.DataFrame() # Return empty on error
            
    else:
        # STATIC LIST (NL/BE)
        # Filter de static dict op basis van suffix (.AS of .BR)
        suffix = ".AS" if market_code == "NL" else ".BR"
        filtered_data = {k: v for k, v in STATIC_EU_DATA.items() if k.endswith(suffix)}
        
        df_result = pd.DataFrame(list(filtered_data.items()), columns=['Ticker', 'Sector'])

    return df_result

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        # We halen 2 jaar data op voor de 200 SMA en lange termijn momentum
        data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)['Close']
        return data
    except Exception:
        return pd.DataFrame()

def calculate_market_regime(ticker):
    """Bepaalt of de markt veilig is (Bull/Bear) obv 200 SMA."""
    try:
        data = yf.download(ticker, period="2y", progress=False, auto_adjust=True)['Close']
        if data.empty: return "UNKNOWN", 0, 0
        current = data.iloc[-1]
        sma = data.rolling(200).mean().iloc[-1]
        return ("BULL" if current > sma else "BEAR"), current, sma
    except:
        return "UNKNOWN", 0, 0

def calculate_rrg_metrics(df, benchmark):
    """Berekent RS-Ratio, RS-Momentum, Heading en Distance."""
    if benchmark not in df.columns: return pd.DataFrame()
    
    rrg_list = []
    
    for col in df.columns:
        if col == benchmark: continue
        
        # 1. Relative Strength
        rs = df[col] / df[benchmark]
        rs_ma = rs.rolling(100).mean()
        
        # 2. RRG Indicatoren
        rs_ratio = 100 * (rs / rs_ma)
        rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
        
        if len(rs_ratio) < 2: continue

        curr_r = rs_ratio.iloc[-1]
        curr_m = rs_mom.iloc[-1]
        prev_r = rs_ratio.iloc[-2]
        prev_m = rs_mom.iloc[-2]
        
        if pd.notna(curr_r) and pd.notna(curr_m):
            # Kwadrant Bepaling
            if curr_r > 100 and curr_m > 100: status = "LEADING 🟢"
            elif curr_r < 100 and curr_m > 100: status = "IMPROVING 🔵"
            elif curr_r < 100 and curr_m < 100: status = "LAGGING 🔴"
            else: status = "WEAKENING 🟡"
            
            # Distance (Alpha potentieel)
            distance = np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            
            # Heading (0-360 graden)
            dx = curr_r - prev_r
            dy = curr_m - prev_m
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 0: angle += 360
            
            # Naamgeving
            name = col
            # Als het een US Sector ETF is, geef de mooie naam
            for k, v in US_SECTOR_ETFS.items():
                if v == col: name = k
            
            rrg_list.append({
                'Ticker': col,
                'Naam': name,
                'RS-Ratio': curr_r,
                'RS-Momentum': curr_m,
                'Kwadrant': status,
                'Distance': distance,
                'Heading': angle
            })
            
    return pd.DataFrame(rrg_list)

def calculate_ranking(df):
    if df.empty or len(df) < 130: return pd.DataFrame()
    
    curr = df.iloc[-1]
    # Returns 1m, 3m, 6m
    r1 = (curr / df.shift(21).iloc[-1]) - 1
    r3 = (curr / df.shift(63).iloc[-1]) - 1
    r6 = (curr / df.shift(126).iloc[-1]) - 1
    
    score = (r1 * 0.2) + (r3 * 0.4) + (r6 * 0.4)
    
    return pd.DataFrame({
        'Ticker': df.columns,
        'Prijs': curr,
        '1M %': r1 * 100,
        '3M %': r3 * 100,
        '6M %': r6 * 100,
        'Score': score * 100
    }).sort_values('Score', ascending=False).dropna()

def get_gemini_advice(ticker, key, market_status, sector):
    """Vraagt advies aan Google Gemini 1.5 Flash."""
    if not key: return "⚠️ Voer eerst een Google Gemini sleutel in."
    
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-flash') # Snel en stabiel
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        info = stock.info
        
        if hist.empty: return "Geen koersdata gevonden."
        
        trend = "Stijgend" if hist['Close'].iloc[-1] > hist['Close'].iloc[0] else "Dalend"
        
        prompt = f"""
        Jij bent een strenge aandelenanalist. Analyseer {ticker} ({info.get('longName', 'Onbekend')}).
        
        HARDE DATA:
        1. Markt Regime (Hoofdindex): {market_status}. (Als BEAR: Wees zeer negatief).
        2. Sector: {sector}.
        3. Korte termijn trend: {trend}.
        
        GEEF ADVIES IN HET NEDERLANDS:
        - Analyseer kort de fundamentele waardering (K/W, groei).
        - Analyseer de Relative Strength t.o.v. de markt.
        - CONCLUSIE: KOPEN / HOUDEN / VERKOPEN.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

# --- UI OPBOUW ---

# SIDEBAR
st.sidebar.header("⚙️ Instellingen")
sel_market_key = st.sidebar.selectbox("Kies Universum", list(MARKETS.keys()))
current_market = MARKETS[sel_market_key]
market_code = current_market['code']
benchmark_ticker = current_market['benchmark']

gemini_key = st.sidebar.text_input("Google Gemini Key", type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("🚦 Markt Regime")
regime_status, r_price, r_sma = calculate_market_regime(benchmark_ticker)
if regime_status == "BULL":
    st.sidebar.success(f"✅ VEILIG (Bull)\nIndex > 200 SMA")
else:
    st.sidebar.error(f"⛔ GEVAAR (Bear)\nIndex < 200 SMA")

# HOOFD SCHERM
st.title(f"🧭 Market Screener: {market_code}")

# 1. Data ophalen (Tickers + Sectoren)
with st.spinner(f"Laden van {sel_market_key}..."):
    constituents_df = get_market_constituents(market_code)
    
    if constituents_df.empty:
        st.error("Kon lijst met aandelen niet laden. Check je internet of bron.")
        st.stop()
        
    all_tickers = constituents_df['Ticker'].tolist()

# 2. Tabs
tab0, tab1, tab2, tab3 = st.tabs(["ℹ️ Info", "📊 Sector Analyse", "🏆 Ranking & Filter", "🤖 AI Advies"])

# TAB 0: INFO
with tab0:
    st.markdown("""
    ### 📚 Handleiding
    1.  **Universum:** Kies links je markt (S&P 500, BEL 20, etc.).
    2.  **Markt Regime:** Check het stoplicht links. Rood = Cash is King (niet kopen).
    3.  **Sector Analyse:** Kijk in Tab 1 welke sectoren 'Leading' zijn (Rechtsboven).
    4.  **Filter:** Ga naar Tab 2, filter op de sterke sector en kies de sterkste aandelen.
    
    **Disclaimer:** Gebaseerd op momentum strategieën. Resultaten uit het verleden bieden geen garantie.
    """)

# TAB 1: RRG (SECTOR ANALYSE)
with tab1:
    st.subheader("Relative Rotation Graph (RRG)")
    
    rrg_tickers = []
    
    # LOGICA: 
    # Voor US Markten -> Toon Sector ETFs (Overzichtelijker)
    # Voor NL/BE Markten -> Toon individuele aandelen ingekleurd per sector (Want geen ETFs)
    if market_code in ["SP500", "SP400"]:
        st.info("We tonen hier de US Sector ETF's t.o.v. de index.")
        rrg_tickers = list(US_SECTOR_ETFS.values())
        color_col = "Kwadrant" # Kleur op basis van fase
    else:
        st.info(f"We tonen alle {market_code} aandelen t.o.v. de index.")
        rrg_tickers = all_tickers
        color_col = "Sector" # Kleur op basis van sector (zodat je clusters ziet)

    # Data ophalen
    rrg_data_raw = get_price_data(rrg_tickers + [benchmark_ticker])
    
    if not rrg_data_raw.empty:
        rrg_metrics = calculate_rrg_metrics(rrg_data_raw, benchmark_ticker)
        
        if not rrg_metrics.empty:
            # Als we in NL/BE mode zijn, moeten we de Sector kolom toevoegen aan de RRG data
            if market_code in ["NL", "BE"]:
                rrg_metrics = rrg_metrics.merge(constituents_df, on='Ticker', how='left')
                # Vul lege sectoren
                rrg_metrics['Sector'] = rrg_metrics['Sector'].fillna('Overig')
            
            
            fig = px.scatter(
                rrg_metrics, 
                x="RS-Ratio", 
                y="RS-Momentum", 
                color=color_col if market_code in ["SP500", "SP400"] else "Sector", # Wisselende kleurstrategie
                text="Naam" if market_code in ["SP500", "SP400"] else "Ticker",
                size="Distance",
                hover_data=["Kwadrant", "Heading"],
                title=f"RRG: {market_code} vs {benchmark_ticker}",
                height=700
            )
            
            # Opmaak RRG
            fig.add_hline(y=100, line_dash="dash", line_color="gray")
            fig.add_vline(x=100, line_dash="dash", line_color="gray")
            fig.add_shape(type="rect", x0=100, y0=100, x1=115, y1=115, fillcolor="green", opacity=0.1, line_width=0)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Niet genoeg data voor RRG berekening.")

# TAB 2: RANKING & FILTER
with tab2:
    st.subheader("🔍 Stock Screener")
    
    # 1. SECTOR FILTER (Werkt nu voor ALLE markten)
    available_sectors = sorted(constituents_df['Sector'].dropna().unique().tolist())
    selected_sector = st.selectbox("📂 Filter op Sector:", ["Alle Sectoren"] + available_sectors)
    
    # Filter toepassen
    if selected_sector != "Alle Sectoren":
        filtered_tickers = constituents_df[constituents_df['Sector'] == selected_sector]['Ticker'].tolist()
        st.success(f"Geselecteerd: {selected_sector} ({len(filtered_tickers)} aandelen)")
    else:
        filtered_tickers = all_tickers

    # 2. RUN KNOP
    if st.button("🚀 Start Analyse"):
        if regime_status == "BEAR":
            st.warning("⚠️ Let op: De markt is in een downtrend (Bear).")
            
        with st.spinner("Koersdata ophalen en rankings berekenen..."):
            stock_data = get_price_data(filtered_tickers)
            
            if not stock_data.empty:
                rank_df = calculate_ranking(stock_data)
                
                if not rank_df.empty:
                    # Top aandeel opslaan voor AI
                    st.session_state['top_pick'] = rank_df.iloc[0]['Ticker']
                    
                    # Merge met sector info voor de tabel
                    display_df = rank_df.merge(constituents_df, on='Ticker', how='left')
                    
                    # Tabel tonen
                    st.dataframe(
                        display_df.style.format({
                            'Prijs': '{:.2f}', '1M %': '{:+.1f}%', 
                            '3M %': '{:+.1f}%', '6M %': '{:+.1f}%', 'Score': '{:.1f}'
                        }).background_gradient(subset=['Score'], cmap='RdYlGn'),
                        use_container_width=True, height=600
                    )
                else:
                    st.error("Geen rankings kunnen genereren.")
            else:
                st.error("Geen data gevonden voor deze selectie.")

# TAB 3: AI
with tab3:
    st.subheader("🤖 Gemini 1.5 Second Opinion")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        default_val = st.session_state.get('top_pick', "")
        user_ticker = st.text_input("Ticker Symbool", value=default_val)
        
        # Sector ophalen voor context
        ticker_sector = "Onbekend"
        if user_ticker:
            row = constituents_df[constituents_df['Ticker'] == user_ticker]
            if not row.empty:
                ticker_sector = row.iloc[0]['Sector']
        
        if st.button("Vraag Advies"):
            with st.spinner("Gemini analyseert..."):
                advies = get_gemini_advice(user_ticker, gemini_key, regime_status, ticker_sector)
                st.markdown(advies)
                st.caption("Gegenereerd door Google Gemini 1.5 Flash")

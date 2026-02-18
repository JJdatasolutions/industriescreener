import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import openai
import requests
import matplotlib.pyplot as plt # Nodig voor de background_gradient

# --- CONFIGURATIE ---
st.set_page_config(page_title="MidCap Market Screener", layout="wide", page_icon="📈")

# Constanten
BENCHMARK_US = 'MDY'  # S&P 400 MidCap ETF
BENCHMARK_NAME = "S&P 400 (MidCap)"

# Sector ETFs mapping (voor de RRG in Tab 1)
SECTOR_MAP = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Comm Services': 'XLC', 'Staples': 'XLP'
}

# --- DATA FUNCTIES ---

@st.cache_data(ttl=24*3600)
def get_sp400_data():
    """
    Haalt S&P 400 tickers EN sectoren op van Wikipedia.
    Geeft een DataFrame terug: [Ticker, Sector]
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    
    try:
        # User-Agent vermomming om 403 error te voorkomen
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        dfs = pd.read_html(response.text)
        df = dfs[0]
        
        # Kolomnamen normaliseren (soms heet het 'Symbol', soms 'Ticker Symbol')
        sym_col = 'Symbol' if 'Symbol' in df.columns else 'Ticker Symbol'
        sec_col = 'GICS Sector' if 'GICS Sector' in df.columns else 'Sector' # Soms heet het anders
        
        # Maak een schone dataframe
        clean_df = df[[sym_col, sec_col]].copy()
        clean_df.columns = ['Ticker', 'Sector']
        
        # Yahoo format fix (punten naar streepjes)
        clean_df['Ticker'] = clean_df['Ticker'].str.replace('.', '-', regex=False)
        
        return clean_df

    except Exception as e:
        st.error(f"⚠️ Wikipedia data niet beschikbaar: {e}")
        # Fallback data als Wikipedia faalt
        return pd.DataFrame({
            'Ticker': ["JBL", "DECK", "RPM", "TTC", "EMN"],
            'Sector': ["Technology", "Discretionary", "Materials", "Industrials", "Materials"]
        })

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        # Batch download
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=True)['Close']
        return data
    except Exception:
        return pd.DataFrame()

def calculate_rrg(df, benchmark):
    if benchmark not in df.columns: return pd.DataFrame()
    
    rrg_data = []
    
    for col in df.columns:
        if col == benchmark: continue
        
        # 1. RS = Prijs / Benchmark
        rs = df[col] / df[benchmark]
        
        # 2. RS-Ratio (Trend) = 100 * (RS / Gemiddelde RS van 100 dagen)
        # Is de trend beter (>100) of slechter (<100) dan de index?
        rs_ma = rs.rolling(window=100).mean()
        rs_ratio = 100 * (rs / rs_ma)
        
        # 3. RS-Momentum (Snelheid) = 100 * (Ratio / Ratio van 10 dagen geleden)
        # Versnelt de trend (>100) of vertraagt hij (<100)?
        rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
        
        curr_r = rs_ratio.iloc[-1]
        curr_m = rs_mom.iloc[-1]
        
        if pd.notna(curr_r) and pd.notna(curr_m):
            if curr_r > 100 and curr_m > 100: status = "LEADING 🟢"
            elif curr_r < 100 and curr_m > 100: status = "IMPROVING 🔵"
            elif curr_r < 100 and curr_m < 100: status = "LAGGING 🔴"
            else: status = "WEAKENING 🟡"
            
            # Zoek leesbare naam voor sectoren
            sector_name = [k for k, v in SECTOR_MAP.items() if v == col]
            label = sector_name[0] if sector_name else col
            
            rrg_data.append({
                'Ticker': col,
                'Naam': label,
                'RS-Ratio': curr_r,
                'RS-Momentum': curr_m,
                'Kwadrant': status
            })
            
    return pd.DataFrame(rrg_data)

def calculate_ranking(df):
    if df.empty or len(df) < 130: return pd.DataFrame()
    
    current = df.iloc[-1]
    p1m = df.shift(21).iloc[-1]
    p3m = df.shift(63).iloc[-1]
    p6m = df.shift(126).iloc[-1]
    
    r1 = (current / p1m) - 1
    r3 = (current / p3m) - 1
    r6 = (current / p6m) - 1
    
    score = (r1 * 0.2) + (r3 * 0.4) + (r6 * 0.4)
    
    rank_df = pd.DataFrame({
        'Ticker': df.columns,
        'Prijs': current,
        '1M %': r1 * 100,
        '3M %': r3 * 100,
        '6M %': r6 * 100,
        'Score': score * 100
    })
    
    return rank_df.sort_values('Score', ascending=False).dropna()

def get_ai_advice(ticker, key):
    if not key: return "⚠️ Voer eerst een OpenAI API sleutel in."
    try:
        client = openai.OpenAI(api_key=key)
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        info = stock.info
        
        if hist.empty: return "Geen data."
        
        trend = "Stijgend" if hist['Close'].iloc[-1] > hist['Close'].iloc[0] else "Dalend"
        
        prompt = f"""
        Analyseer aandeel {ticker}. Bedrijf: {info.get('longName', ticker)}.
        Sector: {info.get('sector', 'Unknown')}. Huidige Trend (1m): {trend}.
        Geef beknopt advies in het Nederlands:
        1. Fundamenteel oordeel.
        2. Technisch oordeel.
        3. Conclusie (Kopen/Houden/Verkopen).
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Fout: {str(e)}"

# --- UI OPBOUW ---

st.sidebar.header("⚙️ Instellingen")

# Markt Keuze
market_options = {
    "🇺🇸 USA (S&P 400 MidCap)": "SP400",
    "🇳🇱 Nederland (AEX/AMX)": "NL",
    "🇧🇪 België (BEL 20)": "BE"
}
selected_market_label = st.sidebar.selectbox("Kies Markt", list(market_options.keys()))
market_code = market_options[selected_market_label]

# API Key
api_key = st.sidebar.text_input("OpenAI Key (Optioneel)", type="password")

st.title(f"🚀 Screener: {selected_market_label}")

# --- DATA VOORBEREIDING ---
tickers_to_scan = []
sector_df = pd.DataFrame()

if market_code == "SP400":
    # Haal de master lijst op (Ticker + Sector)
    sp400_df = get_sp400_data()
    if not sp400_df.empty:
        tickers_to_scan = sp400_df['Ticker'].tolist()
        sector_df = sp400_df # Bewaar voor filtering later
    else:
        st.warning("Kon S&P 400 lijst niet laden.")
elif market_code == "NL":
    tickers_to_scan = ["ASML.AS", "UNA.AS", "HEIA.AS", "SHELL.AS", "AD.AS", "INGA.AS", "DSM.AS", "ABN.AS", "KPN.AS", "WKL.AS", "RAND.AS", "NN.AS", "BESI.AS", "ADYEN.AS", "IMCD.AS"]
elif market_code == "BE":
    tickers_to_scan = ["KBC.BR", "UCB.BR", "SOLB.BR", "ACKB.BR", "ARGX.BR", "UMI.BR", "GBL.BR", "COFB.BR", "WDP.BR", "ELI.BR", "AED.BR"]

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Sector Rotatie", "🏆 Aandelen Ranking", "🤖 AI Analist"])

# TAB 1: RRG (Sectoren)
with tab1:
    st.subheader(f"Sector Rotatie vs {BENCHMARK_NAME}")
    
    # UITLEG EXPANDER
    with st.expander("ℹ️ Hoe lees ik deze grafiek? (Klik voor uitleg)"):
        st.markdown("""
        Deze grafiek (RRG) toont de rotatie van sectoren ten opzichte van de index.
        
        * **X-As (RS-Ratio):** Is de trend beter (>100) of slechter (<100) dan de index?
        * **Y-As (RS-Momentum):** Neemt de kracht toe (>100) of af (<100)?
        
        **De 4 Fases:**
        1.  🟢 **LEADING (Rechtsboven):** Sterke trend + Positief momentum. **(Kopen/Vasthouden)**
        2.  🟡 **WEAKENING (Rechtsonder):** Sterke trend, maar verliest snelheid. **(Winst nemen?)**
        3.  🔴 **LAGGING (Linksonder):** Slechte trend + Negatief momentum. **(Vermijden/Short)**
        4.  🔵 **IMPROVING (Linksboven):** Trend is nog slecht, maar draait bij. **(Kansen zoeken)**
        
        *Tip: De klok mee is de natuurlijke rotatie. Van Improving -> Leading -> Weakening -> Lagging.*
        """)

    if market_code == "SP400":
        sector_tickers = list(SECTOR_MAP.values())
        with st.spinner('Sector data ophalen...'):
            sector_data = get_price_data(sector_tickers + [BENCHMARK_US])
        
        if not sector_data.empty:
            rrg_df = calculate_rrg(sector_data, BENCHMARK_US)
            if not rrg_df.empty:
                
                fig = px.scatter(rrg_df, x="RS-Ratio", y="RS-Momentum", 
                                 color="Kwadrant", text="Naam", hover_data=["Ticker"],
                                 title=f"RRG: Sectoren vs {BENCHMARK_NAME}", height=600,
                                 color_discrete_map={
                                     "LEADING 🟢": "green", "WEAKENING 🟡": "orange",
                                     "LAGGING 🔴": "red", "IMPROVING 🔵": "blue"
                                 })
                
                fig.add_hline(y=100, line_dash="dash", line_color="gray")
                fig.add_vline(x=100, line_dash="dash", line_color="gray")
                # Groen vlak arceren
                fig.add_shape(type="rect", x0=100, y0=100, x1=115, y1=115, fillcolor="green", opacity=0.1, line_width=0)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Onvoldoende data voor RRG.")
    else:
        st.info("Sector RRG is momenteel alleen beschikbaar voor de US markt.")

# TAB 2: Ranking
with tab2:
    st.subheader(f"Momentum Ranking: {selected_market_label}")
    
    # FILTER LOGICA
    active_tickers = tickers_to_scan
    
    if market_code == "SP400" and not sector_df.empty:
        # Haal unieke sectoren op
        unique_sectors = sorted(sector_df['Sector'].unique().tolist())
        # Voeg 'Alle' optie toe
        selected_sector = st.selectbox("🔍 Filter op Industrie:", ["Alle Sectoren"] + unique_sectors)
        
        if selected_sector != "Alle Sectoren":
            # Filter de dataframe
            filtered_df = sector_df[sector_df['Sector'] == selected_sector]
            active_tickers = filtered_df['Ticker'].tolist()
            st.success(f"Gefilterd: {len(active_tickers)} aandelen in sector '{selected_sector}'")
    
    if st.button("🚀 Start Scan"):
        with st.spinner(f"Koersen ophalen voor {len(active_tickers)} aandelen..."):
            
            stock_data = get_price_data(active_tickers)
            
            if not stock_data.empty:
                rank_df = calculate_ranking(stock_data)
                
                if not rank_df.empty:
                    st.session_state['top_stock'] = rank_df.iloc[0]['Ticker']
                    
                    st.dataframe(
                        rank_df.style.format({
                            'Prijs': '{:.2f}', 
                            '1M %': '{:+.1f}%', 
                            '3M %': '{:+.1f}%', 
                            '6M %': '{:+.1f}%', 
                            'Score': '{:.1f}'
                        }).background_gradient(subset=['Score'], cmap='RdYlGn'),
                        use_container_width=True, 
                        height=600
                    )
                else:
                    st.warning("Geen rankings kunnen berekenen.")
            else:
                st.error("Data download mislukt.")

# TAB 3: AI
with tab3:
    st.subheader("🤖 AI Second Opinion")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        default_ticker = st.session_state.get('top_stock', "")
        user_ticker = st.text_input("Ticker Symbool", value=default_ticker)
        analyze_btn = st.button("Vraag AI Advies")
    
    with col2:
        if analyze_btn:
            with st.spinner("AI is aan het analyseren..."):
                advies = get_ai_advice(user_ticker, api_key)
                st.markdown(advies)
                st.caption("Gegenereerd door AI (OpenAI GPT-4o). Geen financieel advies.")

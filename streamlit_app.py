import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import openai
import requests

# --- CONFIGURATIE ---
st.set_page_config(page_title="MidCap Market Screener", layout="wide", page_icon="📈")

# Constanten
BENCHMARK_US = 'MDY'  # S&P 400 MidCap ETF
BENCHMARK_NAME = "S&P 400 (MidCap)"

# Sector ETFs (We gebruiken de standaard SPDRs omdat die het meest liquide zijn, 
# maar we meten ze nu tegen de MidCap index)
SECTOR_MAP = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Comm Services': 'XLC', 'Staples': 'XLP', 
    'Benchmark (MidCap)': BENCHMARK_US
}

# --- DATA FUNCTIES ---

# --- VERVANG DE OUDE get_sp400_tickers FUNCTIE DOOR DEZE ---

@st.cache_data(ttl=24*3600)
def get_sp400_tickers():
    """
    Haalt S&P 400 tickers op met een 'User-Agent' vermomming om
    de Wikipedia blokkade te omzeilen.
    """
    # Noodlijst (Fallback) voor als Wikipedia echt plat ligt
    fallback_list = [
        "JBL", "DECK", "RPM", "TTC", "EMN", "GNTX", "WSO", "FICO", "LECO", "ATR",
        "MANH", "RGLD", "NDSN", "WST", "TECH", "SAIA", "PFGC", "EXP", "CNM", "CASY"
    ]

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        
        # Dit is de truc: We doen alsof we een Windows computer met Chrome zijn
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Gebruik requests om de HTML op te halen
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check of het gelukt is (geen 403 meer)
        
        # Lees de tabellen uit de HTML tekst
        dfs = pd.read_html(response.text)
        df = dfs[0]
        
        # Wikipedia verandert kolomnamen soms. Check welke bestaat.
        if 'Symbol' in df.columns:
            tickers = df['Symbol'].tolist()
        elif 'Ticker Symbol' in df.columns:
            tickers = df['Ticker Symbol'].tolist()
        else:
            # Als we de kolom niet vinden, pak de eerste kolom
            tickers = df.iloc[:, 0].tolist()
            
        # Clean tickers (vervang punten door streepjes voor Yahoo)
        clean_tickers = [str(t).replace('.', '-') for t in tickers]
        
        return clean_tickers

    except Exception as e:
        st.error(f"⚠️ Wikipedia blokkade actief. We tonen een beperkte noodlijst. (Fout: {e})")
        return fallback_list

@st.cache_data(ttl=3600)
def get_data(tickers):
    if not tickers: return pd.DataFrame()
    # Download data
    try:
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=True)['Close']
        return data
    except Exception:
        return pd.DataFrame()

def calculate_rrg(df, benchmark):
    if benchmark not in df.columns: return pd.DataFrame()
    
    rrg_data = []
    
    # Gebruik alleen de laatste datum
    current_date = df.index[-1]
    
    for col in df.columns:
        if col == benchmark: continue
        
        # JdK RS-Ratio & Momentum Berekening
        # 1. RS = Prijs / Benchmark
        rs = df[col] / df[benchmark]
        
        # 2. Ratio = 100 * (RS / Gemiddelde RS van 100 dagen)
        rs_ma = rs.rolling(window=100).mean()
        rs_ratio = 100 * (rs / rs_ma)
        
        # 3. Momentum = 100 * (Ratio / Ratio van 10 dagen geleden)
        rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
        
        curr_r = rs_ratio.iloc[-1]
        curr_m = rs_mom.iloc[-1]
        
        if pd.notna(curr_r) and pd.notna(curr_m):
            if curr_r > 100 and curr_m > 100: status = "LEADING 🟢"
            elif curr_r < 100 and curr_m > 100: status = "IMPROVING 🔵"
            elif curr_r < 100 and curr_m < 100: status = "LAGGING 🔴"
            else: status = "WEAKENING 🟡"
            
            # Zoek leesbare naam
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
    if df.empty: return pd.DataFrame()
    
    # We hebben minstens 6 maanden data nodig
    if len(df) < 130: return pd.DataFrame()
    
    current = df.iloc[-1]
    p1m = df.shift(21).iloc[-1]
    p3m = df.shift(63).iloc[-1]
    p6m = df.shift(126).iloc[-1]
    
    # Rendementen
    r1 = (current / p1m) - 1
    r3 = (current / p3m) - 1
    r6 = (current / p6m) - 1
    
    # Score: Recente prestaties wegen mee, maar trend over 6m is leidend
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
        Analyseer aandeel {ticker} (MidCap VS). Bedrijf: {info.get('longName', ticker)}.
        Sector: {info.get('sector', 'Unknown')}. Huidige Trend (1m): {trend}.
        Geef beknopt advies in het Nederlands:
        1. Fundamenteel oordeel (Kort).
        2. Technisch oordeel (Kort).
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

# Data Logic op basis van keuze
tickers = []
if market_code == "SP400":
    st.info(f"Ophalen van S&P 400 MidCap componenten... (Benchmark: {BENCHMARK_US})")
    tickers = get_sp400_tickers()
elif market_code == "NL":
    tickers = ["ASML.AS", "UNA.AS", "HEIA.AS", "SHELL.AS", "AD.AS", "INGA.AS", "DSM.AS", "ABN.AS", "KPN.AS", "WKL.AS", "RAND.AS", "NN.AS", "BESI.AS", "ADYEN.AS", "IMCD.AS"]
elif market_code == "BE":
    tickers = ["KBC.BR", "UCB.BR", "SOLB.BR", "ACKB.BR", "ARGX.BR", "UMI.BR", "GBL.BR", "COFB.BR", "WDP.BR", "ELI.BR", "AED.BR"]

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Sector Rotatie", "🏆 Aandelen Ranking", "🤖 AI Analist"])

# TAB 1: RRG (Sectoren)
with tab1:
    st.subheader(f"Sector Rotatie vs {BENCHMARK_NAME}")
    if market_code == "SP400":
        # Voor USA gebruiken we de Sector ETFs
        sector_tickers = list(SECTOR_MAP.values())
        with st.spinner('Sector data ophalen...'):
            sector_data = get_data(sector_tickers)
        
        if not sector_data.empty:
            rrg_df = calculate_rrg(sector_data, BENCHMARK_US)
            if not rrg_df.empty:
                # Plot
                
                fig = px.scatter(rrg_df, x="RS-Ratio", y="RS-Momentum", 
                                 color="Kwadrant", text="Naam", hover_data=["Ticker"],
                                 title=f"RRG: Sectoren vs {BENCHMARK_NAME}", height=600)
                
                # Crosshair
                fig.add_hline(y=100, line_dash="dash", line_color="gray")
                fig.add_vline(x=100, line_dash="dash", line_color="gray")
                fig.add_shape(type="rect", x0=100, y0=100, x1=105, y1=105, fillcolor="green", opacity=0.1, line_width=0)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Onvoldoende data voor RRG.")
    else:
        st.info("Sector RRG is momenteel alleen beschikbaar voor de US markt (ivm ETF beschikbaarheid).")

# TAB 2: Ranking
with tab2:
    st.subheader(f"Momentum Ranking: {selected_market_label}")
    
    if st.button("🚀 Start Scan"):
        with st.spinner(f"Koersen ophalen voor {len(tickers)} aandelen..."):
            # Batch download is sneller
            stock_data = get_data(tickers)
            
            if not stock_data.empty:
                rank_df = calculate_ranking(stock_data)
                
                if not rank_df.empty:
                    # Sla op voor AI tab
                    st.session_state['top_stock'] = rank_df.iloc[0]['Ticker']
                    
                    # DE TABEL (Hier zat de fout, nu opgelost door matplotlib in requirements.txt)
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
                    st.warning("Geen rankings kunnen berekenen (data te kort of incompleet).")
            else:
                st.error("Data download mislukt.")

# TAB 3: AI
with tab3:
    st.subheader("🤖 AI Second Opinion")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        # Pak de winnaar uit tab 2 als die er is
        default_ticker = st.session_state.get('top_stock', "")
        user_ticker = st.text_input("Ticker Symbool", value=default_ticker)
        analyze_btn = st.button("Vraag AI Advies")
    
    with col2:
        if analyze_btn:
            with st.spinner("AI is aan het analyseren..."):
                advies = get_ai_advice(user_ticker, api_key)
                st.markdown(advies)
                st.caption("Gegenereerd door AI (OpenAI GPT-4o). Geen financieel advies.")

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import openai

# --- CONFIGURATIE ---
st.set_page_config(page_title="Market Screener Pro", layout="wide", page_icon="📈")

# Markten Configuratie
# We gebruiken vaste lijsten voor NL/BE omdat Wikipedia tabellen vaak veranderen van formaat
MARKETS = {
    "🇺🇸 USA (S&P 500 Sector ETFs)": {
        "type": "etf",
        "tickers": ["XLK", "XLF", "XLV", "XLE", "XLY", "XLI", "XLU", "XLB", "XLRE", "XLC", "XLP", "SPY"]
    },
    "🇳🇱 Nederland (AEX/AMX Selectie)": {
        "type": "stock",
        "tickers": [
            "ASML.AS", "UNA.AS", "HEIA.AS", "SHELL.AS", "AD.AS", "INGA.AS", "DSM.AS", 
            "ABN.AS", "KPN.AS", "WKL.AS", "RAND.AS", "NN.AS", "IMCD.AS", "ASM.AS", 
            "BESI.AS", "MT.AS", "TKWY.AS", "PHI.AS", "UMG.AS", "ADYEN.AS", "WDP.AS"
        ]
    },
    "🇧🇪 België (BEL 20)": {
        "type": "stock",
        "tickers": [
            "ABI.BR", "ACKB.BR", "AED.BR", "AGS.BR", "APAM.BR", "ARGX.BR", "COFB.BR",
            "ELI.BR", "GBL.BR", "KBC.BR", "LOTB.BR", "MELE.BR", "PROX.BR", "SOF.BR",
            "SOLB.BR", "TNET.BR", "UCB.BR", "UMI.BR", "VGP.BR", "WDP.BR", "XIOR.BR"
        ]
    }
}

# Mapping voor leesbare namen
SECTOR_NAMES = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Health Care',
    'XLE': 'Energy', 'XLY': 'Discretionary', 'XLI': 'Industrials',
    'XLU': 'Utilities', 'XLB': 'Materials', 'XLRE': 'Real Estate',
    'XLC': 'Comm Services', 'XLP': 'Staples', 'SPY': 'Benchmark'
}

# --- FUNCTIES ---

@st.cache_data(ttl=3600)
def get_data(tickers):
    if not tickers: return pd.DataFrame()
    # Download data, auto_adjust zorgt voor splits/dividends correctie
    data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)['Close']
    return data

def calculate_rrg(df, benchmark='SPY'):
    # Relatieve Rotatie Grafiek Logic
    if benchmark not in df.columns: return pd.DataFrame()
    
    rrg_data = []
    
    for col in df.columns:
        if col == benchmark: continue
        
        # RS = Stock / Benchmark
        rs = df[col] / df[benchmark]
        
        # RS-Ratio = 100 * (RS / Average(RS, 100))
        rs_ratio = 100 * (rs / rs.rolling(100).mean())
        
        # RS-Momentum = 100 * (Ratio / Ratio_10_days_ago)
        rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
        
        curr_r = rs_ratio.iloc[-1]
        curr_m = rs_mom.iloc[-1]
        
        if pd.notna(curr_r) and pd.notna(curr_m):
            if curr_r > 100 and curr_m > 100: status = "LEADING 🟢"
            elif curr_r < 100 and curr_m > 100: status = "IMPROVING 🔵"
            elif curr_r < 100 and curr_m < 100: status = "LAGGING 🔴"
            else: status = "WEAKENING 🟡"
            
            # Naam ophalen
            name = SECTOR_NAMES.get(col, col)
            
            rrg_data.append({
                'Ticker': col,
                'Naam': name,
                'RS-Ratio': curr_r,
                'RS-Momentum': curr_m,
                'Kwadrant': status
            })
            
    return pd.DataFrame(rrg_data)

def calculate_ranking(df):
    # Momentum Ranking Logic
    if df.empty: return pd.DataFrame()
    
    # Check of we genoeg data hebben (minimaal 6 maanden + beetje extra)
    if len(df) < 130: return pd.DataFrame()
    
    current = df.iloc[-1]
    p1m = df.shift(21).iloc[-1]
    p3m = df.shift(63).iloc[-1]
    p6m = df.shift(126).iloc[-1]
    
    # Returns
    r1 = (current / p1m) - 1
    r3 = (current / p3m) - 1
    r6 = (current / p6m) - 1
    
    # Score: 20% 1m + 40% 3m + 40% 6m
    score = (r1 * 0.2) + (r3 * 0.4) + (r6 * 0.4)
    
    rank_df = pd.DataFrame({
        'Ticker': df.columns,
        'Prijs': current,
        '1M %': r1 * 100,
        '3M %': r3 * 100,
        '6M %': r6 * 100,
        'Score': score * 100
    })
    
    # Verwijder kolommen met NaN score
    rank_df = rank_df.dropna(subset=['Score'])
    
    return rank_df.sort_values('Score', ascending=False)

def get_ai_advice(ticker, key):
    if not key: return "⚠️ Voer eerst een OpenAI API sleutel in."
    
    try:
        client = openai.OpenAI(api_key=key)
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        info = stock.info
        
        if hist.empty: return "Geen data gevonden."
        
        trend = "Stijgend" if hist['Close'].iloc[-1] > hist['Close'].iloc[0] else "Dalend"
        
        prompt = f"""
        Analyseer {ticker}. Bedrijf: {info.get('longName', ticker)}.
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
market_choice = st.sidebar.selectbox("Kies Markt", list(MARKETS.keys()))
api_key = st.sidebar.text_input("OpenAI Key (Optioneel)", type="password")
st.sidebar.info("Zonder key werken de grafieken wel, maar de AI niet.")

st.title(f"🚀 Market Screener: {market_choice}")

# Haal data op
tickers = MARKETS[market_choice]["tickers"]
with st.spinner('Data ophalen...'):
    df_prices = get_data(tickers)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Sector/RRG", "🏆 Ranking", "🤖 AI Analist"])

with tab1:
    st.subheader("Relative Rotation Graph (RRG)")
    # Alleen RRG tonen als we de USA ETF lijst hebben (of een index)
    if "USA" in market_choice and not df_prices.empty:
        rrg_df = calculate_rrg(df_prices)
        if not rrg_df.empty:
            
            fig = px.scatter(rrg_df, x="RS-Ratio", y="RS-Momentum", 
                             color="Kwadrant", text="Naam", hover_data=["Ticker"],
                             title="Sector Rotatie vs S&P 500", height=600)
            fig.add_hline(y=100, line_dash="dash", line_color="gray")
            fig.add_vline(x=100, line_dash="dash", line_color="gray")
            fig.add_shape(type="rect", x0=100, y0=100, x1=110, y1=110, fillcolor="green", opacity=0.1, line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Niet genoeg data voor RRG.")
    else:
        st.info("RRG grafiek is geoptimaliseerd voor de US Sectoren (ETF's). Voor lokale markten, zie de Ranking tab.")

with tab2:
    st.subheader("Momentum Ranking")
    if not df_prices.empty:
        rank_df = calculate_ranking(df_prices)
        if not rank_df.empty:
            st.dataframe(
                rank_df.style.format({
                    'Prijs': '{:.2f}', '1M %': '{:+.1f}%', 
                    '3M %': '{:+.1f}%', '6M %': '{:+.1f}%', 'Score': '{:.1f}'
                }).background_gradient(subset=['Score'], cmap='RdYlGn'),
                use_container_width=True, height=600
            )
            top_pick = rank_df.iloc[0]['Ticker']
        else:
            st.warning("Onvoldoende historie voor ranking.")
            top_pick = ""

with tab3:
    st.subheader("🤖 AI Analyse")
    col1, col2 = st.columns([1, 2])
    with col1:
        # Default waarde is de top pick uit tab 2
        def_val = top_pick if 'top_pick' in locals() else ""
        symbol = st.text_input("Ticker", value=def_val)
        if st.button("Vraag AI"):
            with st.spinner("AI denkt na..."):
                advies = get_ai_advice(symbol, api_key)
                st.session_state['ai_result'] = advies
    
    with col2:
        if 'ai_result' in st.session_state:
            st.markdown(st.session_state['ai_result'])

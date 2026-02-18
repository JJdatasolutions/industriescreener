import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import openai
import requests
import numpy as np

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener (Scientific)", layout="wide", page_icon="🧭")

# Constanten
BENCHMARK_US = 'MDY'  # S&P 400 MidCap ETF
BENCHMARK_NAME = "S&P 400 (MidCap)"

SECTOR_MAP = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Comm Services': 'XLC', 'Staples': 'XLP'
}

# --- DATA FUNCTIES ---

@st.cache_data(ttl=24*3600)
def get_sp400_data():
    """Haalt S&P 400 tickers EN sectoren op van Wikipedia met user-agent spoofing."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.read_html(response.text)[0]
        
        sym_col = 'Symbol' if 'Symbol' in df.columns else 'Ticker Symbol'
        sec_col = 'GICS Sector' if 'GICS Sector' in df.columns else 'Sector'
        
        clean_df = df[[sym_col, sec_col]].copy()
        clean_df.columns = ['Ticker', 'Sector']
        clean_df['Ticker'] = clean_df['Ticker'].str.replace('.', '-', regex=False)
        return clean_df
    except Exception as e:
        return pd.DataFrame({'Ticker': ["JBL", "DECK", "rpm"], 'Sector': ["Tech", "Disc", "Mat"]})

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        # Haal iets meer data op voor de 200 SMA (1y = ~252 dagen)
        data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)['Close']
        return data
    except Exception:
        return pd.DataFrame()

def calculate_market_regime(ticker):
    """
    Mebane Faber Logic: Bepaal of we in een Bull of Bear markt zitten.
    Gebruikt de 10-maands SMA (benaderd als 200-dagen SMA).
    """
    try:
        data = yf.download(ticker, period="2y", progress=False, auto_adjust=True)['Close']
        if data.empty: return None, 0, 0
        
        current_price = data.iloc[-1]
        sma_200 = data.rolling(window=200).mean().iloc[-1]
        
        status = "BULL" if current_price > sma_200 else "BEAR"
        return status, current_price, sma_200
    except:
        return None, 0, 0

def calculate_rrg_metrics(df, benchmark):
    if benchmark not in df.columns: return pd.DataFrame()
    
    rrg_data = []
    
    for col in df.columns:
        if col == benchmark: continue
        
        # 1. RS Berekening
        rs = df[col] / df[benchmark]
        rs_ma = rs.rolling(window=100).mean()
        
        # 2. JdK Ratio & Momentum
        rs_ratio = 100 * (rs / rs_ma)
        rs_mom = 100 * (rs_ratio / rs_ratio.shift(10)) # Rate of Change van de ratio
        
        # Huidige waarden
        curr_r = rs_ratio.iloc[-1]
        curr_m = rs_mom.iloc[-1]
        
        # Vorige waarden (voor Heading berekening)
        prev_r = rs_ratio.iloc[-2]
        prev_m = rs_mom.iloc[-2]
        
        if pd.notna(curr_r) and pd.notna(curr_m):
            # Kwadrant
            if curr_r > 100 and curr_m > 100: status = "LEADING 🟢"
            elif curr_r < 100 and curr_m > 100: status = "IMPROVING 🔵"
            elif curr_r < 100 and curr_m < 100: status = "LAGGING 🔴"
            else: status = "WEAKENING 🟡"
            
            # --- WETENSCHAPPELIJKE TOEVOEGINGEN ---
            
            # 1. Distance (Alpha Potentieel): Afstand tot het centrum (100, 100)
            # Hoe verder weg, hoe sterker de trend (of de overdrijving).
            distance = np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            
            # 2. Heading (Richting): Hoek van de beweging in graden (0-360)
            # 0° = Oost, 90° = Noord, 180° = West, 270° = Zuid.
            # We willen 0-90° (NorthEast) zien.
            dx = curr_r - prev_r
            dy = curr_m - prev_m
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            if angle_deg < 0: angle_deg += 360
            
            sector_name = [k for k, v in SECTOR_MAP.items() if v == col]
            label = sector_name[0] if sector_name else col
            
            rrg_data.append({
                'Ticker': col,
                'Naam': label,
                'RS-Ratio': curr_r,
                'RS-Momentum': curr_m,
                'Kwadrant': status,
                'Distance': distance,
                'Heading': angle_deg
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
    
    # Gewogen score
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

def get_ai_advice(ticker, key, market_status):
    if not key: return "⚠️ Voer eerst een OpenAI API sleutel in."
    try:
        client = openai.OpenAI(api_key=key)
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        info = stock.info
        
        if hist.empty: return "Geen data."
        
        # Bereken heading voor AI context
        recent_closes = hist['Close'].tail(5).tolist()
        trend_context = "Stijgend" if recent_closes[-1] > recent_closes[0] else "Dalend"
        
        prompt = f"""
        Je bent een professionele kwantitatieve analist. 
        De gebruiker overweegt aandeel {ticker} ({info.get('longName')}).
        
        BELANGRIJKE CONTEXT:
        1. Markt Regime (Global Exit): De brede markt is momenteel in een {market_status} fase (t.o.v. 200-daags gemiddelde).
           Als dit BEAR is, wees dan zeer kritisch en adviseer voorzichtigheid/cash.
        2. Sector: {info.get('sector', 'Onbekend')}.
        3. Trend (kort): {trend_context}.

        Geef advies op basis van "Relative Strength" principes:
        - Fundamentele check (zeer kort).
        - Technische check (Is er relatieve sterkte?).
        - CONCLUSIE: KOPEN, HOUDEN of VERKOPEN (Houd rekening met het Markt Regime!).
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Fout: {str(e)}"

# --- UI OPBOUW ---

# 1. SIDEBAR: GLOBAL SETTINGS & MARKET REGIME
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

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Markt Regime (Global Filter)")

# Bepaal benchmark voor traffic light
check_ticker = BENCHMARK_US if market_code == "SP400" else ("^AEX" if market_code == "NL" else "^BFX")
regime, price, sma = calculate_market_regime(check_ticker)

if regime == "BULL":
    st.sidebar.success(f"✅ BULL MARKT\n\nKoers ({price:.0f}) > 200 SMA ({sma:.0f}).\nHet is veilig om long posities te openen.")
else:
    st.sidebar.error(f"⛔ BEAR MARKT\n\nKoers ({price:.0f}) < 200 SMA ({sma:.0f}).\nADVIES: Cash is king. Geen nieuwe aankopen.")

# --- MAIN CONTENT ---
st.title(f"🧭 Scientific Market Screener")

# DATA LADEN
tickers_to_scan = []
sector_df = pd.DataFrame()

if market_code == "SP400":
    sp400_df = get_sp400_data()
    if not sp400_df.empty:
        tickers_to_scan = sp400_df['Ticker'].tolist()
        sector_df = sp400_df
elif market_code == "NL":
    tickers_to_scan = ["ASML.AS", "UNA.AS", "HEIA.AS", "SHELL.AS", "AD.AS", "INGA.AS", "DSM.AS", "ABN.AS", "KPN.AS", "WKL.AS", "RAND.AS", "BESI.AS", "ADYEN.AS", "IMCD.AS"]
elif market_code == "BE":
    tickers_to_scan = ["KBC.BR", "UCB.BR", "SOLB.BR", "ACKB.BR", "ARGX.BR", "UMI.BR", "GBL.BR", "COFB.BR", "WDP.BR", "ELI.BR", "AED.BR"]

# TABS
tab0, tab1, tab2, tab3 = st.tabs(["ℹ️ Info & Strategie", "📊 Sector RRG", "🏆 Aandelen Ranking", "🤖 AI Analist"])

# TAB 0: INFO
with tab0:
    st.markdown("""
    ### Over deze Applicatie
    Deze tool is gebaseerd op wetenschappelijke literatuur over **Momentum** en **Relative Strength** (RS), met name het werk van *Mebane Faber* (Trend Following) en *Dorsey Wright* (Point & Figure RS).
    
    Het doel is om objectief te identificeren welke sectoren en aandelen beter presteren dan de markt.
    
    ### ⚠️ Belangrijke Risico's & Beperkingen
    1.  **Marktregime (Cruciaal):** Gebruik deze tool **alleen** als de 'Global Filter' in de sidebar op groen (BULL) staat. Zoals Faber stelt: *"You want to be in the market when it is above the 10-month SMA."*
    2.  **Transactiekosten & Turnover:** Rotatiestrategieën hebben een hoge omloopsnelheid. Dit kan winsten uithollen.
        * *Advies:* Gebruik een **Sell Filter**. Verkoop een aandeel niet direct als het van nummer 1 naar 2 zakt, maar pas als het uit de top 20% valt. Dit vermindert onnodige handel.
    3.  **Methodiek:** Deze app gebruikt "External RS" (Aandeel vs Benchmark). De zuiverdere "RS Matrix" (elk aandeel vs elk ander aandeel) is rekenkundig te zwaar voor deze web-omgeving, maar de huidige methode is een sterke benadering.
    
    ### Hoe gebruik ik dit?
    1.  **Check de Sidebar:** Is het een Bull Markt? Zo nee -> Niets doen.
    2.  **Sector RRG (Tab 2):** Zoek naar sectoren in het groene vlak (Leading) of blauwe vlak (Improving) die naar rechtsboven bewegen (NorthEast Heading).
    3.  **Ranking (Tab 3):** Filter op die sterke sectoren. Zoek aandelen met een hoge Score én voldoende 'Distance' (Alpha potentieel).
    4.  **AI Check (Tab 4):** Laat de AI een fundamentele sanity-check doen.
    """)

# TAB 1: RRG
with tab1:
    st.subheader(f"Sector Rotatie & Heading")
    st.caption("Focus op bollen met een 'staart' die naar rechtsboven wijst (45°-90°).")
    
    if market_code == "SP400":
        sector_tickers = list(SECTOR_MAP.values())
        with st.spinner('Sector data & headings berekenen...'):
            sector_data = get_price_data(sector_tickers + [BENCHMARK_US])
        
        if not sector_data.empty:
            rrg_df = calculate_rrg_metrics(sector_data, BENCHMARK_US)
            if not rrg_df.empty:
                
                fig = px.scatter(rrg_df, x="RS-Ratio", y="RS-Momentum", 
                                 color="Kwadrant", text="Naam", 
                                 size="Distance", # Grotere bol = meer alpha potentie
                                 hover_data=["Heading"],
                                 title=f"RRG Sectoren (Grootte = Alpha Potentieel)", height=650,
                                 color_discrete_map={
                                     "LEADING 🟢": "#00cc00", "WEAKENING 🟡": "#ffaa00",
                                     "LAGGING 🔴": "#ff0000", "IMPROVING 🔵": "#0000ff"
                                 })
                
                # Assen en lijnen
                fig.add_hline(y=100, line_dash="dash", line_color="gray")
                fig.add_vline(x=100, line_dash="dash", line_color="gray")
                fig.add_shape(type="rect", x0=100, y0=100, x1=115, y1=115, fillcolor="green", opacity=0.1, line_width=0)
                
                # Pijl voor ideale heading (45 graden)
                fig.add_annotation(x=110, y=110, text="Ideal Heading ↗️", showarrow=False, font=dict(color="green"))
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Onvoldoende data.")
    else:
        st.info("Sector analyse is geoptimaliseerd voor US data.")

# TAB 2: RANKING
with tab2:
    st.subheader(f"Momentum Ranking & Selectie")
    
    active_tickers = tickers_to_scan
    
    # Sector Filter
    if market_code == "SP400" and not sector_df.empty:
        sectors = sorted(sector_df['Sector'].unique().tolist())
        sel_sector = st.selectbox("Filter op Industrie:", ["Alle Sectoren"] + sectors)
        if sel_sector != "Alle Sectoren":
            active_tickers = sector_df[sector_df['Sector'] == sel_sector]['Ticker'].tolist()
            st.success(f"Selectie: {len(active_tickers)} aandelen.")

    if st.button("🚀 Start Analyse"):
        if regime == "BEAR":
            st.warning("⚠️ WAARSCHUWING: De markt is in een downtrend. Aankopen worden afgeraden.")
            
        with st.spinner(f"Analyseren van {len(active_tickers)} aandelen..."):
            stock_data = get_price_data(active_tickers)
            if not stock_data.empty:
                rank_df = calculate_ranking(stock_data)
                if not rank_df.empty:
                    st.session_state['top_stock'] = rank_df.iloc[0]['Ticker']
                    
                    # Style de tabel
                    st.dataframe(
                        rank_df.style.format({
                            'Prijs': '{:.2f}', '1M %': '{:+.1f}%', 
                            '3M %': '{:+.1f}%', '6M %': '{:+.1f}%', 'Score': '{:.1f}'
                        }).background_gradient(subset=['Score'], cmap='RdYlGn'),
                        use_container_width=True, height=600
                    )
                else:
                    st.warning("Geen data.")

# TAB 3: AI
with tab3:
    st.subheader("🤖 AI Risk Manager")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        def_ticker = st.session_state.get('top_stock', "")
        user_ticker = st.text_input("Ticker", value=def_ticker)
        
        # Geef visuele feedback over het marktregime aan de gebruiker
        if regime == "BULL":
            st.success("Marktlicht: GROEN (Kopen mag)")
        else:
            st.error("Marktlicht: ROOD (Pas op!)")
            
        analyze_btn = st.button("Vraag Advies")
    
    with col2:
        if analyze_btn:
            with st.spinner("AI analyseert marktomstandigheden en aandeel..."):
                advies = get_ai_advice(user_ticker, api_key, regime)
                st.markdown(advies)

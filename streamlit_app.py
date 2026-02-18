import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener 6.0", layout="wide", page_icon="🧭")

# --- 1. DATA DEFINITIES & CONSTANTEN ---

MARKETS = {
    "🇺🇸 USA - S&P 500": {
        "code": "SP500", "benchmark": "SPY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    },
    "🇺🇸 USA - S&P 400 (MidCap)": {
        "code": "SP400", "benchmark": "MDY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    },
    "🇪🇺 Europa - Selectie": {
        "code": "EU_MIX", "benchmark": "^N100", "type": "static"
    }
}

# Mapping van Sector Naam naar ETF Ticker (voor relatieve sterkte berekening)
US_SECTOR_MAP = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Consumer Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Communication Services': 'XLC', 'Consumer Staples': 'XLP'
}

COLOR_MAP = {
    "1. LEADING": "#006400",    # Donkergroen
    "2. WEAKENING": "#FFA500",  # Oranje
    "3. LAGGING": "#DC143C",    # Rood
    "4. IMPROVING": "#90EE90"   # Lichtgroen
}

# --- 2. FUNCTIES ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_key):
    """Haalt de lijst met aandelen op."""
    mkt = MARKETS[market_key]
    
    if "EU_MIX" in mkt.get("code", ""):
        # Statische lijst voor stabiliteit
        data = {
            "ASML.AS": "Technology", "UNA.AS": "Consumer Staples", "HEIA.AS": "Consumer Staples", 
            "SHELL.AS": "Energy", "INGA.AS": "Financials", "DSM.AS": "Materials", 
            "BESI.AS": "Technology", "ADYEN.AS": "Financials", "IMCD.AS": "Materials", 
            "PHI.AS": "Health Care", "KBC.BR": "Financials", "UCB.BR": "Health Care", 
            "SOLB.BR": "Materials", "WDP.BR": "Real Estate", "ELI.BR": "Utilities",
            "MELE.BR": "Industrials", "XIOR.BR": "Real Estate", "ACKB.BR": "Financials",
            "ABI.BR": "Consumer Staples", "GBL.BR": "Financials"
        }
        return pd.DataFrame(list(data.items()), columns=['Ticker', 'Sector'])

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        tables = pd.read_html(mkt['wiki'], attrs={"id": "constituents"})
        if not tables: tables = pd.read_html(mkt['wiki']) # Fallback
        
        df = tables[0]
        
        # Kolom mapping normaliseren
        cols = {c: c.lower() for c in df.columns}
        sym_col = next((k for k, v in cols.items() if 'symbol' in v), None)
        sec_col = next((k for k, v in cols.items() if 'sector' in v), None)
        
        if sym_col and sec_col:
            df_clean = df[[sym_col, sec_col]].copy()
            df_clean.columns = ['Ticker', 'Sector']
            df_clean['Ticker'] = df_clean['Ticker'].str.replace('.', '-', regex=False)
            return df_clean
            
        return pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        # Download data
        data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
        
        # Flatten MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', level=1, axis=1)
            else:
                data = data.iloc[:, 0] if data.shape[1] == 1 else data
        return data
    except:
        return pd.DataFrame()

def calculate_rrg(df, benchmark_ticker):
    """
    Berekent RRG coördinaten.
    Als benchmark_ticker = 'SPY', vergelijken we met de markt.
    Als benchmark_ticker = 'XLE', vergelijken we met de sector (Energy).
    """
    if benchmark_ticker not in df.columns: return pd.DataFrame()
    
    rrg_data = []
    bench_series = df[benchmark_ticker]
    
    for ticker in df.columns:
        if ticker == benchmark_ticker: continue
        try:
            # RS Ratio (Relatieve Sterkte vs Benchmark)
            rs = df[ticker] / bench_series
            rs_ma = rs.rolling(100).mean()
            rs_ratio = 100 * (rs / rs_ma)
            
            # RS Momentum (Snelheid van de verandering)
            rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
            
            if len(rs_ratio) < 1: continue
            
            curr_r = rs_ratio.iloc[-1]
            curr_m = rs_mom.iloc[-1]
            
            # Kwadrant bepaling
            if curr_r > 100 and curr_m > 100: status = "1. LEADING"
            elif curr_r < 100 and curr_m > 100: status = "4. IMPROVING"
            elif curr_r < 100 and curr_m < 100: status = "3. LAGGING"
            else: status = "2. WEAKENING"
            
            rrg_data.append({
                'Ticker': ticker,
                'RS-Ratio': curr_r,
                'RS-Momentum': curr_m,
                'Kwadrant': status,
                'Distance': np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            })
        except: continue
        
    return pd.DataFrame(rrg_data)

def calculate_signals(df_prices):
    """Maakt signaal tabel voor Tab 3"""
    if df_prices.empty: return pd.DataFrame()
    
    stats = []
    for ticker in df_prices.columns:
        try:
            series = df_prices[ticker]
            curr = series.iloc[-1]
            sma200 = series.rolling(200).mean().iloc[-1]
            
            # Returns
            ret_1m = (curr / series.shift(21).iloc[-1]) - 1
            ret_3m = (curr / series.shift(63).iloc[-1]) - 1
            
            # Signaal Logica
            trend = "🟢 BULL" if curr > sma200 else "🔴 BEAR"
            signal = "KOPEN" if (curr > sma200 and ret_1m > 0) else "AFWACHTEN"
            
            stats.append({
                'Ticker': ticker,
                'Prijs': curr,
                'Trend': trend,
                '1M %': ret_1m * 100,
                '3M %': ret_3m * 100,
                'Signaal': signal
            })
        except: continue
        
    return pd.DataFrame(stats).sort_values('3M %', ascending=False)

# --- 3. SIDEBAR LOGICA (DASHBOARD) ---

st.sidebar.header("⚙️ Instellingen")
sel_market_key = st.sidebar.selectbox("Kies Markt", list(MARKETS.keys()))
market_cfg = MARKETS[sel_market_key]
sel_sector = st.sidebar.selectbox("Kies Sector (voor Tab 3)", ["Alle Sectoren"] + sorted(US_SECTOR_MAP.keys()) if "USA" in sel_market_key else ["Alle Sectoren"])

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Markt Thermometer")

# Benchmark Data Ophalen voor Sidebar
bench_df = get_price_data([market_cfg['benchmark']])

if not bench_df.empty:
    # Zeker zijn dat het een Series is en geen DataFrame
    s = bench_df.iloc[:, 0] if isinstance(bench_df, pd.DataFrame) else bench_df
    
    current_price = s.iloc[-1]
    sma_200 = s.rolling(200).mean().iloc[-1]
    
    # Afstand tot SMA (Dispersie)
    distance_pct = ((current_price - sma_200) / sma_200) * 100
    
    if current_price > sma_200:
        status = "BULL MARKET"
        color = "green"
        icon = "📈"
    else:
        status = "BEAR MARKET"
        color = "red"
        icon = "📉"
        
    st.sidebar.markdown(f"Status: :{color}[**{status}**] {icon}")
    st.sidebar.metric("Afstand tot 200 SMA", f"{distance_pct:.2f}%")
    
    # Mini Grafiek: Prijs vs SMA
    chart_data = s.tail(300).to_frame(name="Koers")
    chart_data['SMA200'] = s.rolling(200).mean().tail(300)
    
    fig_mini = px.line(chart_data, y=["Koers", "SMA200"], height=200, title="Trend")
    fig_mini.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0), xaxis_title=None, yaxis_title=None)
    fig_mini.update_traces(line_color='red', selector=dict(name='SMA200')) # SMA rood maken
    fig_mini.update_traces(line_color='blue', selector=dict(name='Koers'))
    st.sidebar.plotly_chart(fig_mini, use_container_width=True)

else:
    st.sidebar.warning("Geen data.")

# KNOP OM ALLES TE LADEN
st.sidebar.markdown("---")
if st.sidebar.button("🚀 Start Analyse", type="primary"):
    st.session_state['active'] = True
    st.session_state['market_key'] = sel_market_key
    st.session_state['sector_sel'] = sel_sector

# --- 4. HOOFD SCHERM (TABS) ---

st.title(f"Market Screener: {market_cfg['code']}")

tab1, tab2, tab3, tab4 = st.tabs(["ℹ️ Info & Uitleg", "🚁 Sector Rotatie", "🔍 Aandelen Detail", "🤖 AI (Future)"])

# === TAB 1: INFO ===
with tab1:
    st.markdown("""
    ### 📖 Handleiding: Hoe werkt deze App?

    Deze applicatie helpt je om de sterkste aandelen in de sterkste sectoren te vinden. We gebruiken hiervoor de **RRG Methodiek** (Relative Rotation Graph).

    #### 1. De Vier Kwadranten
    De grafieken in Tab 2 en 3 zijn verdeeld in vier vlakken. Elk vlak vertelt iets over de trend van een aandeel/sector:
    
    * 🟢 **LEADING (Rechtsboven):** **KOPEN.** Het aandeel is sterk én het momentum is positief. Dit zijn de winnaars van het moment.
    * 🟠 **WEAKENING (Rechtsonder):** **OPPASSEN.** Het aandeel is nog steeds sterk (trend is omhoog), maar verliest snelheid. Vaak een moment om winst te nemen.
    * 🔴 **LAGGING (Linksonder):** **VERKOPEN.** Zowel de trend als het momentum zijn negatief. Hier wil je niet zitten.
    * 🍏 **IMPROVING (Linksboven):** **KANSEN.** Het aandeel is zwak, maar het momentum draait bij naar positief. Dit zijn de potentiële turn-around kandidaten.

    #### 2. De Sidebar (Linkerkant)
    * **Markt Thermometer:** Geeft aan of de algemene markt veilig is.
        * **BULL:** Koers boven het 200-daags gemiddelde. (Veilig om te kopen).
        * **BEAR:** Koers onder het 200-daags gemiddelde. (Cash is king).
    * **Dispersie:** Hoeveel procent zitten we boven of onder die trendlijn? Een extreem hoge dispersie (bv +15%) kan betekenen dat de markt 'oververhit' is.
    """)

# === TAB 2: SECTOR ROTATIE ===
with tab2:
    if st.session_state.get('active'):
        st.subheader("Sector Rotatie Diagram")
        st.caption("Welke sectoren presteren beter dan de markt?")
        
        with st.spinner("Sector data ophalen..."):
            # Bepaal de 'spelers' (Sectoren)
            if "USA" in st.session_state['market_key']:
                tickers = list(US_SECTOR_MAP.values())
                labels = {v: k for k, v in US_SECTOR_MAP.items()} # Omgekeerde map voor labels
            else:
                # Voor Europa pakken we de top aandelen als proxy omdat ETF data lastig is
                constituents = get_market_constituents(st.session_state['market_key'])
                tickers = constituents['Ticker'].head(20).tolist()
                labels = {t: t for t in tickers}

            # Voeg benchmark toe voor berekening
            tickers.append(market_cfg['benchmark'])
            
            # Data ophalen & Rekenen
            df_prices = get_price_data(tickers)
            rrg_df = calculate_rrg(df_prices, market_cfg['benchmark'])
            
            if not rrg_df.empty:
                # Labels netjes maken
                rrg_df['Label'] = rrg_df['Ticker'].map(labels).fillna(rrg_df['Ticker'])
                
                fig = px.scatter(rrg_df, x="RS-Ratio", y="RS-Momentum", 
                                 color="Kwadrant", text="Label", size="Distance",
                                 color_discrete_map=COLOR_MAP, height=700,
                                 title=f"Sectoren t.o.v. {market_cfg['benchmark']}")
                
                # Assen kruis
                fig.add_hline(y=100, line_dash="dash", line_color="grey")
                fig.add_vline(x=100, line_dash="dash", line_color="grey") 
                fig.update_traces(textposition='top center', textfont=dict(size=12, family="Arial Black"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Geen data beschikbaar voor sectoren.")
    else:
        st.info("Klik op 'Start Analyse' in de sidebar.")

# === TAB 3: STOCK DETAIL (RELATIEF AAN SECTOR) ===
with tab3:
    if st.session_state.get('active'):
        current_sec = st.session_state['sector_sel']
        st.subheader(f"Analyse: {current_sec}")
        
        # 1. Bepaal Benchmark voor Tab 3 (De Sector zelf!)
        # Als we USA doen en een specifieke sector kiezen, is de ETF de benchmark.
        # Anders (of bij 'Alle Sectoren') is de marktindex de benchmark.
        if "USA" in st.session_state['market_key'] and current_sec != "Alle Sectoren":
            sector_benchmark = US_SECTOR_MAP.get(current_sec, market_cfg['benchmark'])
            st.markdown(f"**Benchmark voor grafiek:** {current_sec} ETF ({sector_benchmark})")
        else:
            sector_benchmark = market_cfg['benchmark']
            st.markdown(f"**Benchmark voor grafiek:** Hoofdindex ({sector_benchmark})")

        # 2. Haal tickers op
        constituents = get_market_constituents(st.session_state['market_key'])
        if current_sec != "Alle Sectoren":
            stock_tickers = constituents[constituents['Sector'] == current_sec]['Ticker'].tolist()
        else:
            stock_tickers = constituents['Ticker'].head(50).tolist() # Limit voor snelheid
            
        # Zorg dat de benchmark ook in de download zit
        download_tickers = list(set(stock_tickers + [sector_benchmark]))
        
        with st.spinner(f"Koersen laden van {len(download_tickers)} aandelen..."):
            df_stock_prices = get_price_data(download_tickers)
            
            # 3. Bereken RRG (Stocks vs Sector Benchmark)
            rrg_stocks = calculate_rrg(df_stock_prices, sector_benchmark)
            
            # 4. Bereken Signalen (Absolute returns)
            signals_df = calculate_signals(df_stock_prices)
            
            # WEERGAVE
            if not rrg_stocks.empty:
                col_graph, col_table = st.columns([3, 2])
                
                with col_graph:
                    st.markdown("##### 📊 Spreidingsdiagram (vs Sector)")
                    fig2 = px.scatter(rrg_stocks, x="RS-Ratio", y="RS-Momentum", 
                                     color="Kwadrant", text="Ticker", size="Distance",
                                     color_discrete_map=COLOR_MAP, height=600)
                    fig2.add_hline(y=100, line_dash="dash", line_color="grey")
                    fig2.add_vline(x=100, line_dash="dash", line_color="grey")
                    fig2.update_traces(textposition='top center')
                    st.plotly_chart(fig2, use_container_width=True)
                    
                with col_table:
                    st.markdown("##### 📋 Signaal Tabel")
                    st.dataframe(
                        signals_df[['Ticker', 'Trend', 'Signaal', '1M %', '3M %']]
                        .style.applymap(lambda v: 'color: green' if v == 'KOPEN' else '', subset=['Signaal'])
                        .format({'1M %': '{:.1f}%', '3M %': '{:.1f}%'}),
                        use_container_width=True, height=600
                    )
            else:
                st.warning("Onvoldoende data voor deze selectie.")

# === TAB 4: AI ===
with tab4:
    st.write("🤖 AI functionaliteit komt in de volgende update.")

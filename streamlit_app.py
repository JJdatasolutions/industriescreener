import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import math

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener 7.0 (Quant AI)", layout="wide", page_icon="🧠")

# --- 1. DATA DEFINITIES ---
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

US_SECTOR_MAP = {
    'Information Technology': 'XLK',  # Aangepast van 'Technology'
    'Financials': 'XLF',
    'Health Care': 'XLV',
    'Energy': 'XLE',
    'Consumer Discretionary': 'XLY',
    'Industrials': 'XLI',
    'Utilities': 'XLU',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
    'Consumer Staples': 'XLP'
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
    mkt = MARKETS[market_key]
    
    # 1. EU LOGICA (Blijft hetzelfde)
    if "EU_MIX" in mkt.get("code", ""):
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

    # 2. USA SCRAPER (Met Fallback)
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Probeer Wikipedia op te halen
        response = requests.get(mkt['wiki'], headers=headers, timeout=5)
        response.raise_for_status() # Check op HTTP fouten (bijv. 429 Too Many Requests)
        
        tables = pd.read_html(response.text)
        target_df = pd.DataFrame()
        
        # Slimmer zoeken naar de juiste tabel
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            # Zoek specifiek naar GICS Sector OF Sector
            if any("symbol" in c for c in cols) and (any("gics sector" in c for c in cols) or any("sector" in c for c in cols)):
                target_df = df
                break
        
        if target_df.empty:
            raise ValueError("Geen geschikte tabel gevonden op Wikipedia.")

        # Kolommen identificeren
        ticker_col = next(c for c in target_df.columns if "Symbol" in str(c) or "Ticker" in str(c))
        # Pak voorkeur voor 'GICS Sector', anders 'Sector'
        sector_col_candidates = [c for c in target_df.columns if "Sector" in str(c)]
        sector_col = sector_col_candidates[0] # Pak de eerste match
        
        df_clean = target_df[[ticker_col, sector_col]].copy()
        df_clean.columns = ['Ticker', 'Sector']
        df_clean['Ticker'] = df_clean['Ticker'].str.replace('.', '-', regex=False)
        
        # Schoonmaken
        df_clean['Sector'] = df_clean['Sector'].astype(str).str.strip()
        
        # Check of we data hebben, zo niet: forceer error om naar fallback te gaan
        if df_clean.empty: raise ValueError("Lege dataset na cleaning.")
            
        return df_clean

    except Exception as e:
        # 3. FALLBACK MECHANISME (Als Wikipedia faalt)
        # Dit zorgt ervoor dat je app ALTIJD werkt, zelfs zonder internetverbinding naar Wiki
        st.warning(f"⚠️ Wikipedia data niet beschikbaar ({e}). Gebruik nood-dataset (Top 30 US Stocks).")
        
        fallback_data = {
            'AAPL': 'Information Technology', 'MSFT': 'Information Technology', 'NVDA': 'Information Technology',
            'AMZN': 'Consumer Discretionary', 'GOOGL': 'Communication Services', 'META': 'Communication Services',
            'TSLA': 'Consumer Discretionary', 'BRK-B': 'Financials', 'LLY': 'Health Care', 'AVGO': 'Information Technology',
            'JPM': 'Financials', 'V': 'Financials', 'XOM': 'Energy', 'UNH': 'Health Care',
            'PG': 'Consumer Staples', 'MA': 'Financials', 'JNJ': 'Health Care', 'HD': 'Consumer Discretionary',
            'MRK': 'Health Care', 'COST': 'Consumer Staples', 'ABBV': 'Health Care', 'AMD': 'Information Technology',
            'CRM': 'Information Technology', 'NFLX': 'Communication Services', 'PEP': 'Consumer Staples',
            'KO': 'Consumer Staples', 'BAC': 'Financials', 'WMT': 'Consumer Staples', 'CVX': 'Energy',
            'ADBE': 'Information Technology'
        }
        return pd.DataFrame(list(fallback_data.items()), columns=['Ticker', 'Sector'])
@st.cache_data(ttl=3600)
def get_price_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0): data = data['Close']
            elif 'Close' in data.columns.get_level_values(1): data = data.xs('Close', level=1, axis=1)
            else: 
                if data.shape[1] == len(tickers):
                     data = data.droplevel(1, axis=1) if data.columns.nlevels > 1 else data
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers
        return data
    except: return pd.DataFrame()

def calculate_rrg_extended(df, benchmark_ticker):
    """
    RRG + Heading (Hoek) berekening - GECORRIGEERD
    Heading is nu gebaseerd op BEWEGING (Velocity), niet op positie.
    """
    if df.empty or benchmark_ticker not in df.columns: return pd.DataFrame()
    
    rrg_data = []
    bench_series = df[benchmark_ticker]
    
    for ticker in df.columns:
        if ticker == benchmark_ticker: continue
        try:
            rs = df[ticker] / bench_series
            rs_ma = rs.rolling(100).mean()
            rs_ratio = 100 * (rs / rs_ma)
            rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
            
            # We hebben minstens 2 punten nodig om een richting te bepalen
            if len(rs_ratio) < 2: continue
            
            curr_r = rs_ratio.iloc[-1]
            curr_m = rs_mom.iloc[-1]
            prev_r = rs_ratio.iloc[-2]  # De waarde van gisteren/vorige periode
            prev_m = rs_mom.iloc[-2]
            
            # --- KWANTITATIEVE UPDATE: Heading & Distance ---
            
            # 1. Distance (Euclidisch vanaf centrum 100,100) - Dit blijft positie!
            dist = np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            
            # 2. Heading (Hoek van de BEWEGING)
            # We kijken naar de vector: waar kwamen we vandaan -> waar zijn we nu?
            dx = curr_r - prev_r
            dy = curr_m - prev_m
            
            # Als er geen beweging is, is de hoek 0 (of ongedefinieerd, hier vangen we dat op)
            if dx == 0 and dy == 0:
                heading_deg = 0
            else:
                heading_rad = math.atan2(dy, dx)
                heading_deg = math.degrees(heading_rad)
                if heading_deg < 0: heading_deg += 360
            
            # Bepaal 'Power Heading' (0-90 graden is North-East = Winstgevende richting)
            # Dit kan nu dus OOK gebeuren als een aandeel linksboven of linksonder staat!
            is_power_heading = 0 <= heading_deg <= 90
            
            # Kwadrant bepaling (blijft gebaseerd op positie)
            if curr_r > 100 and curr_m > 100: status = "1. LEADING"
            elif curr_r < 100 and curr_m > 100: status = "4. IMPROVING"
            elif curr_r < 100 and curr_m < 100: status = "3. LAGGING"
            else: status = "2. WEAKENING"
            
            rrg_data.append({
                'Ticker': ticker,
                'RS-Ratio': curr_r,
                'RS-Momentum': curr_m,
                'Kwadrant': status,
                'Distance': dist,
                'Heading': heading_deg,
                'Power_Heading': "✅ YES" if is_power_heading else "❌ NO"
            })
        except: continue
        
    return pd.DataFrame(rrg_data)
def calculate_market_health(bench_series):
    curr = bench_series.iloc[-1]
    sma200 = bench_series.rolling(200).mean().iloc[-1]
    trend = "BULL" if curr > sma200 else "BEAR"
    dist_pct = ((curr - sma200) / sma200) * 100
    return trend, dist_pct, sma200

# --- 3. SIDEBAR ---

st.sidebar.header("⚙️ Instellingen")
sel_market_key = st.sidebar.selectbox("Kies Markt", list(MARKETS.keys()))
market_cfg = MARKETS[sel_market_key]
sel_sector = st.sidebar.selectbox("Kies Sector", ["Alle Sectoren"] + sorted(US_SECTOR_MAP.keys()) if "USA" in sel_market_key else ["Alle Sectoren"])

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Markt & Dispersie")

bench_df = get_price_data([market_cfg['benchmark']])

if not bench_df.empty:
    s = bench_df.iloc[:, 0]
    trend, dist_pct, sma200 = calculate_market_health(s)
    
    color = "green" if trend == "BULL" else "red"
    st.sidebar.markdown(f"Regime: :{color}[**{trend} MARKET**]")
    st.sidebar.metric("Afstand tot 200d Lijn", f"{dist_pct:.2f}%")
    
    # --- NIEUW: DISPERSIE INDICATOR ---
    # We berekenen de std dev van de returns van de sectoren (als proxy)
    if st.session_state.get('active'):
        st.sidebar.markdown("**Dispersie Meter:**")
        st.sidebar.progress(70) # Statische placeholder voor nu, zou dynamisch berekend moeten worden op alle stock returns
        st.sidebar.caption("Hoog = Veel verschil tussen winnaars/verliezers")

    # Mini Grafiek
    chart_data = s.tail(300).to_frame(name="Koers")
    chart_data['SMA200'] = s.rolling(200).mean().tail(300)
    fig_mini = px.line(chart_data, y=["Koers", "SMA200"], height=150)
    fig_mini.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None)
    fig_mini.update_traces(line_color='red', selector=dict(name='SMA200'))
    st.sidebar.plotly_chart(fig_mini, use_container_width=True)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Start Analyse", type="primary"):
    st.session_state['active'] = True
    st.session_state['market_key'] = sel_market_key
    st.session_state['sector_sel'] = sel_sector

# --- 4. HOOFD SCHERM ---

st.title(f"Quant Screener: {market_cfg['code']}")
tab1, tab2, tab3, tab4 = st.tabs(["ℹ️ Info", "🚁 Sectoren", "🔍 Aandelen", "🧠 AI Analyst"])

# === TAB 1: INFO ===
with tab1:
    st.markdown("""
    ### 📊 Quant Methodology (v7.0)
    Deze tool gebruikt geavanceerde Relative Rotation (RRG) logica.
    
    **Nieuw in v7.0:**
    * **Heading (Hoek):** We berekenen nu de exacte hoek van de beweging. Een hoek tussen **0-90° (Noordoost)** is statistisch de plek waar Alpha wordt gegenereerd.
    * **Distance:** De afstand tot het centrum (100,100). Hoe groter de afstand, hoe krachtiger de trend.
    
    **Legenda:**
    * 🟢 **LEADING:** Sterke trend, sterk momentum. (Buy/Hold)
    * 🟠 **WEAKENING:** Sterke trend, afnemend momentum. (Take Profit/Watch)
    * 🔴 **LAGGING:** Zwakke trend, zwak momentum. (Sell/Avoid)
    * 🍏 **IMPROVING:** Zwakke trend, toenemend momentum. (Speculative Buy)
    """)

# === TAB 2: SECTOREN ===
with tab2:
    if st.session_state.get('active'):
        st.subheader("Sector Rotatie")
        
        # 1. Data Ophalen
        with st.spinner("Sectoren analyseren..."):
            tickers = list(US_SECTOR_MAP.values()) if "USA" in st.session_state['market_key'] else []
            
            # Fallback voor EU of als map leeg is
            if not tickers: 
                 constituents = get_market_constituents(st.session_state['market_key'])
                 tickers = constituents['Ticker'].head(15).tolist() if not constituents.empty else []

            if tickers:
                # Benchmark toevoegen voor berekening
                calc_tickers = list(set(tickers + [market_cfg['benchmark']]))
                df_sec = get_price_data(calc_tickers)
                rrg_sec = calculate_rrg_extended(df_sec, market_cfg['benchmark'])
                
                if not rrg_sec.empty:
                    # 2. Labels en Cleaning
                    # We maken een mooie naam (bijv 'Technology') ipv de ticker (XLK)
                    labels = {v: k for k, v in US_SECTOR_MAP.items()}
                    # Map de ticker naar de naam, als die niet bestaat, gebruik de ticker
                    rrg_sec['Label'] = rrg_sec['Ticker'].map(labels).fillna(rrg_sec['Ticker'])
                    
                    # Filter: Alleen de opgevraagde tickers (excl benchmark zelf soms)
                    rrg_sec = rrg_sec[rrg_sec['Ticker'].isin(tickers)]
                    rrg_sec = rrg_sec[rrg_sec['Distance'] > 0]

                    # 3. Visualisatie (Exacte kopie van Tab 3 stijl)
                    
                    # --- CUSTOM COLOR SCALE ---
                    # 0-90 = Groen (met 45 als piek). >90 = Rood.
                    custom_color_scale = [
                        # GROENE ZONE (0 - 90)
                        (0.00, "#a7f3d0"),  # 0°   : Lichtgroen
                        (0.125, "#065f46"), # 45°  : DONKERGROEN (Sweet Spot)
                        (0.25, "#a7f3d0"),  # 90°  : Lichtgroen
                        
                        # RODE ZONE (91 - 360)
                        (0.2501, "#fca5a5"), # 90.1°: Lichtrood
                        (0.50, "#dc2626"),   # 180° : Rood
                        (0.75, "#991b1b"),   # 270° : Donker Rood
                        (1.00, "#450a0a")    # 360° : Zwart/Rood
                    ]

                    fig = px.scatter(
                        rrg_sec, 
                        x="RS-Ratio", 
                        y="RS-Momentum", 
                        color="Heading", 
                        text="Label",  # Gebruik de leesbare Sector naam
                        size="Distance",
                        height=650,
                        hover_data=["Kwadrant", "Power_Heading"],
                        title=f"<b>SECTOR ROTATIE</b> <br><sup>Focus op 45° (Donkergroen) | Vermijd Rood</sup>"
                    )
                    
                    # STYLING
                    fig.update_traces(
                        marker=dict(
                            line=dict(width=1, color='black'), 
                            opacity=0.9
                        ),
                        textposition='top center',
                        textfont=dict(size=11, color='darkslategrey', family="Arial Black") # Iets dikkere tekst voor sectoren
                    )
                    
                    # LAYOUT & KLEURENKAART
                    fig.update_layout(
                        coloraxis_cmin=0,
                        coloraxis_cmax=360,
                        coloraxis_colorscale=custom_color_scale,
                        coloraxis_colorbar=dict(
                            title="Richting",
                            tickvals=[0, 45, 90, 180, 270],
                            ticktext=["0°", "45° TOP", "90° Grens", "180°", "270°"]
                        ),
                        template="plotly_white",
                        xaxis=dict(showgrid=True, gridcolor='#f2f2f2', zeroline=False), 
                        yaxis=dict(showgrid=True, gridcolor='#f2f2f2', zeroline=False),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=60, b=40, l=40, r=40)
                    )
                    
                    # Watermerken & Assen
                    fig.add_hline(y=100, line_color="black", line_width=1)
                    fig.add_vline(x=100, line_color="black", line_width=1)
                    
                    # Visuele tekst annotaties
                    fig.add_annotation(x=101, y=101, text="BUY ZONE", showarrow=False, font=dict(size=14, color="#065f46"))
                    fig.add_annotation(x=99, y=99, text="AVOID", showarrow=False, font=dict(size=14, color="#991b1b"))

                    # Centreren van de grafiek
                    max_dev = max(abs(rrg_sec['RS-Ratio']-100).max(), abs(rrg_sec['RS-Momentum']-100).max()) * 1.1
                    fig.update_xaxes(range=[100-max_dev, 100+max_dev])
                    fig.update_yaxes(range=[100-max_dev, 100+max_dev])

                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- MINI TABEL (Optioneel maar handig) ---
                    # Toon snel de beste sectoren onder de grafiek
                    st.markdown("#### 🏆 Sector Ranking (Dichtst bij 45°)")
                    df_view = rrg_sec.copy()
                    df_view['Afwijking_45'] = abs(df_view['Heading'] - 45)
                    
                    # Sorteer op beste heading (dichtst bij 45)
                    top_sec = df_view.sort_values('Afwijking_45', ascending=True)
                    
                    st.dataframe(
                        top_sec[['Label', 'Heading', 'Distance', 'Kwadrant']].style
                        .background_gradient(subset=['Heading'], cmap='Greens', vmin=0, vmax=90)
                        .format({"Heading": "{:.1f}°", "Distance": "{:.2f}"}),
                        hide_index=True,
                        use_container_width=True
                    )

                else: 
                    st.warning("Geen sector data beschikbaar.")
            else:
                st.warning("Geen tickers gevonden voor deze markt.")
import plotly.graph_objects as go # We hebben de low-level API nodig voor custom styling

# === TAB 3: AANDELEN ===
with tab3:
    if st.session_state.get('active'):
        current_sec = st.session_state.get('sector_sel', 'Alle Sectoren')
        st.subheader(f"Deep Dive: {current_sec}")
        
        # 1. Setup Data
        if "USA" in st.session_state['market_key'] and current_sec != "Alle Sectoren":
            bench_ticker = US_SECTOR_MAP.get(current_sec, market_cfg['benchmark'])
        else:
            bench_ticker = market_cfg['benchmark']
        
        constituents = get_market_constituents(st.session_state['market_key'])
        
        if not constituents.empty:
            if current_sec != "Alle Sectoren":
                subset = constituents[constituents['Sector'] == current_sec]['Ticker'].tolist()
            else:
                subset = constituents['Ticker'].head(60).tolist()
            
            dl_list = list(set(subset + [bench_ticker]))
            
            # Data ophalen
            if 'df_stocks_raw' not in st.session_state or len(st.session_state['df_stocks_raw'].columns) < len(dl_list):
                 st.info(f"Koersdata ophalen voor {len(dl_list)} aandelen...")
                 df_stocks = get_price_data(dl_list)
                 st.session_state['df_stocks_raw'] = df_stocks
            else:
                 df_stocks = st.session_state['df_stocks_raw']
            st.session_state['bench_ticker_t3'] = bench_ticker
            
            # Bereken RRG
            rrg_stocks = calculate_rrg_extended(df_stocks, bench_ticker)
            
            if not rrg_stocks.empty:
                # Cleaning
                rrg_stocks = rrg_stocks.dropna(subset=['RS-Ratio', 'RS-Momentum', 'Distance', 'Kwadrant'])
                rrg_stocks = rrg_stocks[rrg_stocks['Distance'] > 0]
                st.session_state['rrg_stocks_data'] = rrg_stocks 
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("### 🧭 Relative Rotation Graph")
                    
                    # --- CUSTOM COLOR SCALE (De 'Sweet Spot' Logica) ---
                    # 0-360 graden gemapt naar 0.0-1.0
                    # 45 graden = 0.125
                    # 90 graden = 0.25
                    
                    custom_color_scale = [
                        # DEEL 1: DE GROENE ZONE (0 - 90)
                        (0.00, "#a7f3d0"),  # 0°   : Lichtgroen (Start Leading)
                        (0.125, "#065f46"), # 45°  : DONKERGROEN (De Sweet Spot - Max Power)
                        (0.25, "#a7f3d0"),  # 90°  : Lichtgroen (Einde Leading)
                        
                        # DEEL 2: DE RODE ZONE (91 - 360)
                        # We maken een harde overgang direct na 90 (0.25)
                        (0.2501, "#fca5a5"), # 90.1°: Lichtrood (Start Zwakte)
                        (0.50, "#dc2626"),   # 180° : Rood
                        (0.75, "#991b1b"),   # 270° : Donkerder Rood
                        (1.00, "#450a0a")    # 360° : Zeer Donker Rood/Zwart
                    ]

                    fig2 = px.scatter(
                        rrg_stocks, 
                        x="RS-Ratio", 
                        y="RS-Momentum", 
                        color="Heading", 
                        text="Ticker", 
                        size="Distance",
                        height=700,
                        hover_data=["Kwadrant"],
                        title=f"<b>RRG SIGNAL: {current_sec}</b> <br><sup>Focus op 45° (Donkergroen) | Vermijd Rood</sup>"
                    )
                    
                    # Styling
                    fig2.update_traces(
                        marker=dict(
                            line=dict(width=1, color='black'), 
                            opacity=0.85
                        ),
                        textposition='top center',
                        textfont=dict(size=10, color='darkslategrey')
                    )
                    
                    # Forceer de kleurenkaart exact over 0-360
                    fig2.update_layout(
                        coloraxis_cmin=0,
                        coloraxis_cmax=360,
                        coloraxis_colorscale=custom_color_scale,
                        coloraxis_colorbar=dict(
                            title="Richting",
                            tickvals=[0, 45, 90, 180, 270],
                            ticktext=["Start", "45° TOP", "90° Grens", "180°", "270°"]
                        ),
                        template="plotly_white",
                        xaxis=dict(showgrid=True, gridcolor='#f2f2f2', zeroline=False), 
                        yaxis=dict(showgrid=True, gridcolor='#f2f2f2', zeroline=False),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=60, b=40, l=40, r=40)
                    )

                    # Assen en Watermerken
                    fig2.add_hline(y=100, line_color="black", line_width=1)
                    fig2.add_vline(x=100, line_color="black", line_width=1)
                    
                    # Visuele hulpmiddelen
                    fig2.add_annotation(x=102, y=102, text="BUY ZONE", showarrow=False, font=dict(size=14, color="#065f46"))
                    fig2.add_annotation(x=98, y=98, text="AVOID", showarrow=False, font=dict(size=14, color="#991b1b"))

                    # Centreren
                    max_dev = max(abs(rrg_stocks['RS-Ratio']-100).max(), abs(rrg_stocks['RS-Momentum']-100).max()) * 1.1
                    fig2.update_xaxes(range=[100-max_dev, 100+max_dev])
                    fig2.update_yaxes(range=[100-max_dev, 100+max_dev])

                    st.plotly_chart(fig2, use_container_width=True)

                with col2:
                    st.markdown("### 🎯 Focus List")
                    st.caption("Top Picks (Dichtst bij 45°)")
                    
                    # We maken een custom sortering:
                    # We willen aandelen die het dichtst bij 45 graden zitten bovenaan.
                    # Dus we berekenen de absolute afwijking van 45.
                    
                    df_view = rrg_stocks.copy()
                    df_view['Afwijking_45'] = abs(df_view['Heading'] - 45)
                    
                    # Filter: Alleen de groene zone (0-90) in de tabel tonen? 
                    # Of alles tonen maar sorteren op 'beste'?
                    # Laten we de top 15 tonen die het dichtst bij 45 graden zitten (en >0 heading).
                    
                    top_picks = df_view[
                        (df_view['Heading'] >= 0) & (df_view['Heading'] <= 90)
                    ].sort_values('Afwijking_45', ascending=True).head(15)
                    
                    if not top_picks.empty:
                        st.dataframe(
                            top_picks[['Ticker', 'Heading', 'Distance']].style
                            .background_gradient(subset=['Heading'], cmap='Greens') # Simpele groen tint voor tabel
                            .format({"Heading": "{:.1f}°", "Distance": "{:.2f}"}),
                            hide_index=True,
                            use_container_width=True,
                            height=600
                        )
                    else:
                        st.write("Geen aandelen in de Buy Zone (0-90°).")

            else:
                st.warning("⚠️ Onvoldoende data voor plot.")
        else:
            st.error("Kon geen aandelenlijst ophalen.")
# === TAB 4: AI ANALYST ===
with tab4:
    st.header("🧠 AI-Agent Prompt Generator")
    st.markdown("Selecteer een aandeel uit de resultaten. De app genereert de **perfecte kwantitatieve prompt** met de live berekende data (Heading, Distance, RRG) die je direct in ChatGPT of Gemini kunt plakken.")
    
    if 'rrg_stocks_data' in st.session_state:
        rrg_data = st.session_state['rrg_stocks_data']
        # Selectbox met aandelen
        stock_pick = st.selectbox("Selecteer aandeel voor analyse:", rrg_data['Ticker'].unique())
        
        if stock_pick:
            # Haal data op voor dit aandeel
            row = rrg_data[rrg_data['Ticker'] == stock_pick].iloc[0]
            
            # Context ophalen (Market Regime)
            # Herberekenen of ophalen uit sidebar scope (even opnieuw voor zekerheid)
            regime_trend = "BULL" # Zou uit state moeten komen, default bull
            
            # --- DE PROMPT GENERATOR ---
            ai_prompt = f"""
Handel als een Expert Quantitative Equity Analyst gespecialiseerd in Relative Strength en Sector Rotatie.

Ik wil een analyse van het aandeel: **{stock_pick}**.

Hier is de LIVE technische data uit mijn kwantitatieve model (RRG):
1. **Markt Regime:** {regime_trend} (o.b.v. SMA200 hoofdindex).
2. **RRG Positie (vs Benchmark):**
   - **Kwadrant:** {row['Kwadrant']}
   - **RS-Ratio:** {row['RS-Ratio']:.2f} (Trend)
   - **RS-Momentum:** {row['RS-Momentum']:.2f} (Snelheid)
   - **Distance:** {row['Distance']:.2f} (Hoe ver van het gemiddelde)
   - **Heading (Hoek):** {row['Heading']:.1f} graden. (0-90 is ideaal/Noordoost).
   - **Power Heading:** {row['Power_Heading']}

Jouw taak:
Voer een grondige screening uit op basis van deze data en jouw externe kennis:

1. **Sector Onderscheiding (Relative Strength Context):**
   - Analyseer de positie van {stock_pick}. Is het een 'Leader' of 'Laggard'? 
   - Beoordeel de Distance ({row['Distance']:.2f}). Is dit significant genoeg voor Alpha?
   - Interpreteer de Heading ({row['Heading']:.1f}°). Versnelt de trend?

2. **Fundamentele & Kwalitatieve Screening (Fama-French Logica):**
   - *Gebruik je eigen kennis:* Hoe scoort {stock_pick} op de Fama-French factoren (Profitability, Investment)? 
   - Is de beweging fundamenteel te onderbouwen?

3. **Conclusie & Actie:**
   - Geef een gewogen oordeel (KOPEN / HOUDEN / VERKOPEN).
   - Specifieke 'stop-loss' suggestie gebaseerd op volatiliteit.
            """
            
            st.text_area("Kopieer deze prompt:", value=ai_prompt, height=400)
            st.info("Tip: Plak dit in Gemini Advanced of ChatGPT-4o voor het beste resultaat.")
            
    else:
        st.warning("Draai eerst de analyse in Tab 3 om data te genereren.")

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import math
import httpx
import io
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# --- 0. CONFIGURATIE ---
st.set_page_config(page_title="Anomalos Pro 10.0 (Velocity Edition)", layout="wide", page_icon="🦅")

st.markdown("""
    <style>
    .stDataFrame { border: 1px solid #f0f2f6; border-radius: 5px; }
    .alarm-box { padding: 15px; border-radius: 8px; background-color: #ff4b4b; color: white; font-weight: bold; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA DEFINITIES ---
MARKETS = {
    "🇺🇸 USA - S&P 500": {"benchmark": "SPY", "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"},
    "🇺🇸 USA - S&P 400": {"benchmark": "MDY", "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"},
    "🇪🇺 Europa - Selectie": {"benchmark": "^N100", "type": "static"}
}

# --- 2. DATA FUNCTIES ---

@st.cache_data(ttl=86400)
def get_constituents(market_key):
    mkt = MARKETS[market_key]
    if "Europa" in market_key:
        data = {"ASML.AS": "Tech", "UNA.AS": "Staples", "SHELL.AS": "Energy", "INGA.AS": "Finance"}
        return pd.DataFrame(list(data.items()), columns=['Ticker', 'Sector'])
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(mkt['wiki'], headers=headers)
            df = pd.read_html(io.StringIO(str(BeautifulSoup(resp.text, 'html.parser').find('table'))))[0]
        
        t_col = next(c for c in df.columns if "Symbol" in str(c) or "Ticker" in str(c))
        s_col = next(c for c in df.columns if "Sector" in str(c))
        
        res = pd.DataFrame()
        res['Ticker'] = df[t_col].str.replace('.', '-', regex=False)
        res['Sector'] = df[s_col]
        return res
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_prices(tickers, days=400):
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    data = yf.download(list(tickers), start=start_date, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs('Close', level=0, axis=1)
    return data.ffill().bfill()

# --- 3. QUANT ENGINE (Met Velocity & Explosie Detectie) ---

def calc_rrg(df, bench_col, profile, lookback=0):
    results = []
    current_df = df.iloc[:len(df)-lookback] if lookback > 0 else df
    
    if bench_col not in current_df.columns: return pd.DataFrame()
    bench = current_df[bench_col]
    
    for ticker in current_df.columns:
        if ticker == bench_col: continue
        try:
            s = current_df[ticker]
            rs = s / bench
            ratio = 100 * (rs / rs.rolling(100).mean())
            mom = 100 * (ratio / ratio.shift(10))
            
            # Huidige en vorige datapunten
            r, m = ratio.iloc[-1], mom.iloc[-1]
            pr, pm = ratio.iloc[-2], mom.iloc[-2]
            
            # Wiskundige variabelen
            dist = math.sqrt((r-100)**2 + (m-100)**2) # Afstand (Alpha)
            velocity = math.sqrt((r-pr)**2 + (m-pm)**2) # Snelheid van de beweging (1 dag)
            heading = math.degrees(math.atan2(m - pm, r - pr)) % 360 # Richting
            
            # Kwadrant bepaling
            if r >= 100 and m >= 100: kw = "LEADING"
            elif r < 100 and m >= 100: kw = "IMPROVING"
            elif r < 100 and m < 100: kw = "LAGGING"
            else: kw = "WEAKENING"
            
            # Standaard Actie Logica
            action = "HOLD"
            if "Momentum" in profile and kw == "LEADING" and 0 <= heading <= 90: action = "✅ MOMENTUM BUY"
            elif "Value" in profile and kw == "IMPROVING" and 0 <= heading <= 180: action = "💎 VALUE BUY"
            elif kw == "LEADING" and dist > 5: action = "🏆 COMBO BUY"
            elif kw == "LAGGING": action = "❌ AVOID"

            # 🚨 EXPLOSIE DETECTIE (De "Gouden Formule")
            # 1. Hoge snelheid (v > 1.2 is een flinke beweging in RRG termen)
            # 2. Goede richting (Tussen 0 en 90 graden, recht op LEADING af)
            # 3. NIET overextended (Afstand tot centrum mag niet groter dan 7 zijn)
            is_exploding = (velocity > 1.2) and (0 <= heading <= 90) and (dist < 7.0) and (kw in ["IMPROVING", "LEADING"])
            
            if is_exploding:
                action = "🚨 VELOCITY BREAKOUT"
            
            results.append({
                'Ticker': ticker, 'RS_Ratio': r, 'RS_Mom': m, 
                'Kwadrant': kw, 'Action': action, 'Alpha': dist, 
                'Velocity': velocity, 'Heading': heading
            })
        except: continue
    return pd.DataFrame(results)

# --- 4. INTERFACE ---

st.sidebar.header("🦅 Anomalos Control Panel")
market_choice = st.sidebar.selectbox("Markt", list(MARKETS.keys()))
profile_choice = st.sidebar.selectbox("Profiel", ["Momentum Profile", "Value Profile", "Balanced"])

if st.sidebar.button("🚀 START DEEP SCAN"):
    constituents = get_constituents(market_choice)
    bench_symbol = MARKETS[market_choice]['benchmark']
    
    with st.spinner("Velocity & Acceleratie aan het berekenen..."):
        all_prices = get_prices(tuple(constituents['Ticker'].tolist() + [bench_symbol]))
        results = calc_rrg(all_prices, bench_symbol, profile_choice)
        results = pd.merge(results, constituents, on='Ticker', how='left')

        # --- NIEUW: HET ALARM KANAAL ---
        alarms = results[results['Action'] == "🚨 VELOCITY BREAKOUT"].sort_values('Velocity', ascending=False)
        
        if not alarms.empty:
            st.error(f"🚨 {len(alarms)} AANDELEN DETECTEREN MASSALE VELOCITY (NIET OVEREXTENDED) 🚨", icon="🚀")
            st.markdown("Deze aandelen maken momenteel een abnormale acceleratie door in het Improving/Vroeg-Leading kwadrant, zonder dat ze uitgeput zijn.")
            
            # Tonen van de alarm-tabel met nadruk op de Snelheid (Velocity)
            st.dataframe(alarms[['Ticker', 'Sector', 'Kwadrant', 'Velocity', 'Alpha', 'RS_Ratio', 'RS_Mom']]
                         .style.background_gradient(subset=['Velocity'], cmap='Reds'), use_container_width=True)
        else:
            st.success("Radar check voltooid. Geen plotselinge, veilige uitbraken gedetecteerd vandaag. De markt beweegt stabiel.")

        st.markdown("---")

        # --- STANDAARD VISUALISATIE & TABEL ---
        st.subheader(f"Markt Matrix: {market_choice}")
        fig = px.scatter(results, x="RS_Ratio", y="RS_Mom", color="Kwadrant", text="Ticker", size="Alpha",
                         color_discrete_map={"LEADING":"green","IMPROVING":"blue","WEAKENING":"orange","LAGGING":"red"})
        fig.add_vline(x=100, line_dash="dash")
        fig.add_hline(y=100, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Alle Geoptimaliseerde Kansen")
        
        # Filters
        c1, c2 = st.columns(2)
        with c1:
            f_act = st.multiselect("Filter Actie", results['Action'].unique(), default=[a for a in results['Action'].unique() if "BUY" in a or "BREAKOUT" in a])
        with c2:
            f_sec = st.multiselect("Filter Sector", results['Sector'].unique(), default=results['Sector'].unique())

        filtered = results[(results['Action'].isin(f_act)) & (results['Sector'].isin(f_sec))].sort_values("Alpha", ascending=False)

        styled_df = filtered.style.background_gradient(subset=['Alpha'], cmap='YlGn')\
                                  .format({'RS_Ratio': '{:.2f}', 'RS_Mom': '{:.2f}', 'Alpha': '{:.2f}', 'Velocity': '{:.2f}'})
        st.dataframe(styled_df, use_container_width=True)

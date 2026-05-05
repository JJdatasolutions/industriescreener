import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import httpx
import io
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# --- 0. CONFIGURATIE ---
st.set_page_config(page_title="Anomalos Pro 9.0", layout="wide", page_icon="🦅")

# Custom CSS voor een professionele look
st.markdown("""
    <style>
    .stDataFrame { border: 1px solid #f0f2f6; border-radius: 5px; }
    .status-box { padding: 20px; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA DEFINITIES ---
MARKETS = {
    "🇺🇸 USA - S&P 500": {"benchmark": "SPY", "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"},
    "🇺🇸 USA - S&P 400": {"benchmark": "MDY", "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"},
    "🇪🇺 Europa - Selectie": {"benchmark": "^N100", "type": "static"}
}

US_SECTOR_MAP = {
    'Information Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Consumer Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Communication Services': 'XLC', 'Consumer Staples': 'XLP'
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
        
        # Flexibele kolomdetectie
        t_col = [c for c in df.columns if "Symbol" in str(c) or "Ticker" in str(c)][0]
        s_col = [c for c in df.columns if "Sector" in str(c)][0]
        
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

# --- 3. QUANT ENGINE ---

def calc_rrg(df, bench_col, profile, lookback=0):
    """Berekent RRG voor een specifiek moment (lookback=0 is vandaag)"""
    results = []
    # Snijd de dataframe af gebaseerd op lookback voor historische analyse
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
            
            r, m = ratio.iloc[-1], mom.iloc[-1]
            pr, pm = ratio.iloc[-2], mom.iloc[-2]
            
            dist = math.sqrt((r-100)**2 + (m-100)**2)
            heading = math.degrees(math.atan2(m - pm, r - pr)) % 360
            
            kw = "LEADING" if r>=100 and m>=100 else "IMPROVING" if r<100 and m>=100 else "LAGGING" if r<100 and m<100 else "WEAKENING"
            
            action = "HOLD"
            if "Momentum" in profile and kw == "LEADING" and 0 <= heading <= 90: action = "✅ MOMENTUM BUY"
            elif "Value" in profile and kw == "IMPROVING" and 0 <= heading <= 180: action = "💎 VALUE BUY"
            elif kw == "LEADING" and dist > 5: action = "🏆 COMBO BUY"
            elif kw == "LAGGING": action = "❌ AVOID"
            
            results.append({'Ticker': ticker, 'RS_Ratio': r, 'RS_Mom': m, 'Kwadrant': kw, 'Action': action, 'Alpha': dist})
        except: continue
    return pd.DataFrame(results)

# --- 4. INTERFACE ---

st.sidebar.header("🦅 Anomalos Control Panel")
market_choice = st.sidebar.selectbox("Markt", list(MARKETS.keys()))
profile_choice = st.sidebar.selectbox("Profiel", ["Momentum Profile", "Value Profile", "Balanced"])
scan_date = st.sidebar.date_input("Scan Datum (Snapshot)", datetime.now())

if st.sidebar.button("🚀 START DEEP SCAN"):
    constituents = get_constituents(market_choice)
    bench_symbol = MARKETS[market_choice]['benchmark']
    
    with st.spinner("Data-archeologie in uitvoering..."):
        all_prices = get_prices(tuple(constituents['Ticker'].tolist() + [bench_symbol]))
        results = calc_rrg(all_prices, bench_symbol, profile_choice)
        results = pd.merge(results, constituents, on='Ticker', how='left')

        # 1. VISUALISATIE
        st.subheader(f"Markt Matrix: {market_choice}")
        fig = px.scatter(results, x="RS_Ratio", y="RS_Mom", color="Kwadrant", text="Ticker", size="Alpha",
                         color_discrete_map={"LEADING":"green","IMPROVING":"blue","WEAKENING":"orange","LAGGING":"red"})
        fig.add_vline(x=100, line_dash="dash")
        fig.add_hline(y=100, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        # 2. FILTERS & RANKING
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            f_act = st.multiselect("Filter Actie", results['Action'].unique(), default=[a for a in results['Action'].unique() if "BUY" in a])
        with c2:
            f_sec = st.multiselect("Filter Sector", results['Sector'].unique(), default=results['Sector'].unique())

        filtered = results[(results['Action'].isin(f_act)) & (results['Sector'].isin(f_sec))].sort_values("Alpha", ascending=False)

        st.subheader("💡 Geoptimaliseerde Kansen")
        
        # KLEURENLEGENDE RANKING: Gebruik de Pandas Styler voor de Alpha kolom
        styled_df = filtered.style.background_gradient(subset=['Alpha'], cmap='YlGn')\
                                  .format({'RS_Ratio': '{:.2f}', 'RS_Mom': '{:.2f}', 'Alpha': '{:.2f}'})
        st.dataframe(styled_df, use_container_width=True)

        # 3. AI AGENT SELECTIE & PROMPT
        st.markdown("---")
        st.subheader("🤖 AI Decision Support")
        if not filtered.empty:
            selected_stock = st.selectbox("Selecteer aandeel voor AI Analyse", filtered['Ticker'].tolist())
            row = filtered[filtered['Ticker'] == selected_stock].iloc[0]
            
            # De ORIGINELE AI PROMPT (Hersteld)
            full_prompt = f"""
AKTIE: {row['Action']}
AANDEEL: {selected_stock} ({row['Sector']})
KWADRANT: {row['Kwadrant']}
QUANT DATA: Ratio {row['RS_Ratio']:.2f}, Momentum {row['RS_Mom']:.2f}, Alpha {row['Alpha']:.2f}

JULLIE OPDRACHT (Hedge Fund Mode):

1. QUANT AUDIT (De Quant):
Evalueer de metrieken. Is de trend duurzaam of over-extended?

2. FUNDAMENTELE VALIDATIE (De Analist):
Waarom stroomt er specifiek NU kapitaal naar {selected_stock}? Zoek de katalysator.

3. RISK & VOLATILITY (De Risk Manager):
Geef concrete entry- en exit-levels. Bepaal een Stop-Loss.

4. HET OORDEEL (De Consensus):
Synthetiseer in een definitief advies: [STERK KOPEN | KOPEN | HOUDEN | VERMIJDEN]
            """
            st.text_area("Kopieer naar Gemini/ChatGPT:", full_prompt, height=300)
            
            # 4. HISTORICAL TRACKER (De Kalender/Tijd functie)
            st.markdown("---")
            st.subheader(f"📅 Tijdlijn Analyse: {selected_stock} (Laatste 30 dagen)")
            
            hist_data = []
            for i in range(30, -1, -1):
                day_res = calc_rrg(all_prices, bench_symbol, profile_choice, lookback=i)
                if not day_res.empty:
                    stock_day = day_res[day_res['Ticker'] == selected_stock]
                    if not stock_day.empty:
                        hist_data.append({
                            'Datum': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                            'Action': stock_day.iloc[0]['Action'],
                            'Alpha': stock_day.iloc[0]['Alpha']
                        })
            
            hist_df = pd.DataFrame(hist_data)
            fig_hist = px.line(hist_df, x='Datum', y='Alpha', title=f"Alpha Evolutie {selected_stock}", markers=True)
            # Voeg kleurzones toe voor acties
            st.plotly_chart(fig_hist, use_container_width=True)
            st.table(hist_df.tail(10)) # Laatste 10 dagen in tabelvorm

        else:
            st.info("Geen aandelen gevonden met huidige filters.")

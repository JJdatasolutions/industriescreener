import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
import httpx
import io
import math
import scipy.linalg as la
from bs4 import BeautifulSoup
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# --- 0. CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener 8.0 (Institutional)", layout="wide", page_icon="🦅")

# CSS voor styling van de kwadranten in de tabel
st.markdown("""
    <style>
    .reportview-container .main .block-container{ padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA DEFINITIES ---
MARKETS = {
    "🇺🇸 USA - S&P 500": {"code": "SP500", "benchmark": "SPY", "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"},
    "🇺🇸 USA - S&P 400 (MidCap)": {"code": "SP400", "benchmark": "MDY", "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"},
    "🇪🇺 Europa - Selectie": {"code": "EU_MIX", "benchmark": "^N100", "type": "static"}
}

US_SECTOR_MAP = {
    'Information Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Consumer Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Communication Services': 'XLC', 'Consumer Staples': 'XLP'
}

# --- 2. DATA ACQUISITIE (ROBUUST) ---

@st.cache_data(ttl=86400)
def get_market_constituents(market_key):
    mkt = MARKETS.get(market_key)
    if "EU_MIX" in mkt.get("code", ""):
        data = {
            "ASML.AS": "Technology", "UNA.AS": "Consumer Staples", "SHELL.AS": "Energy", 
            "INGA.AS": "Financials", "ADYEN.AS": "Financials", "PHI.AS": "Health Care", 
            "KBC.BR": "Financials", "ABI.BR": "Consumer Staples", "SAP.DE": "Technology"
        }
        return pd.DataFrame(list(data.items()), columns=['Ticker', 'Sector'])

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        with httpx.Client(timeout=15.0) as client:
            response = client.get(mkt['wiki'], headers=headers)
            response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        df = pd.read_html(io.StringIO(str(table)))[0]
        cols = [str(c).lower() for c in df.columns]
        t_idx = next(i for i, c in enumerate(cols) if "symbol" in c or "ticker" in c)
        s_idx = next(i for i, c in enumerate(cols) if "sector" in c)
        
        res = pd.DataFrame()
        res['Ticker'] = df.iloc[:, t_idx].astype(str).str.replace('.', '-', regex=False)
        res['Sector'] = df.iloc[:, s_idx].astype(str)
        return res
    except:
        return pd.DataFrame(columns=['Ticker', 'Sector'])

@st.cache_data(ttl=3600)
def get_price_data(tickers: tuple):
    data = yf.download(list(tickers), period="2y", progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs('Close', level=0, axis=1) if 'Close' in data.columns.get_level_values(0) else data.xs('Close', level=1, axis=1)
    return data.ffill().bfill()

# --- 3. QUANT LOGICA ---

def calculate_rrg_metrics(df, benchmark_ticker, profile):
    if df.empty or benchmark_ticker not in df.columns: return pd.DataFrame()
    
    rrg_results = []
    bench = df[benchmark_ticker]
    
    for ticker in df.columns:
        if ticker == benchmark_ticker: continue
        try:
            s = df[ticker]
            rs = (s / bench)
            rs_ratio = 100 * (rs / rs.rolling(100).mean())
            rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
            
            curr_r, curr_m = rs_ratio.iloc[-1], rs_mom.iloc[-1]
            prev_r, prev_m = rs_ratio.iloc[-2], rs_mom.iloc[-2]
            
            # Heading & Distance
            dist = math.sqrt((curr_r-100)**2 + (curr_m-100)**2)
            heading = math.degrees(math.atan2(curr_m - prev_m, curr_r - prev_r)) % 360
            
            # Kwadrant
            if curr_r >= 100 and curr_m >= 100: kw = "LEADING"
            elif curr_r < 100 and curr_m >= 100: kw = "IMPROVING"
            elif curr_r < 100 and curr_m < 100: kw = "LAGGING"
            else: kw = "WEAKENING"
            
            # Action Logic
            action = "HOLD"
            score = dist * (1.2 if 0 <= heading <= 90 else 0.8)
            
            if "Momentum" in profile and kw == "LEADING" and 0 <= heading <= 90: action = "✅ MOMENTUM BUY"
            elif "Value" in profile and kw == "IMPROVING" and 0 <= heading <= 180: action = "💎 VALUE BUY"
            elif kw == "LEADING" and score > 5: action = "🏆 COMBO BUY"
            elif kw == "LAGGING": action = "❌ AVOID"

            rrg_results.append({
                'Ticker': ticker, 'RS_Ratio': curr_r, 'RS_Mom': curr_m,
                'Kwadrant': kw, 'Heading': heading, 'Distance': dist,
                'Action': action, 'Alpha': score
            })
        except: continue
    return pd.DataFrame(rrg_results)

# --- 4. UI EN VISUALISATIE ---

# SIDEBAR ANALYTICS
st.sidebar.header("📊 Market Intelligence")
market_choice = st.sidebar.selectbox("Selecteer Markt", list(MARKETS.keys()))
profile_choice = st.sidebar.selectbox("Strategisch Profiel", ["Momentum Profile", "Value Profile", "Balanced"])

# Sector Performance Chart in Sidebar
if "USA" in market_choice:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sector Momentum")
    sector_tickers = list(US_SECTOR_MAP.values())
    s_data = get_price_data(tuple(sector_tickers + [MARKETS[market_choice]['benchmark']]))
    if not s_data.empty:
        perf = (s_data.iloc[-1] / s_data.iloc[-20] - 1) * 100
        fig_sidebar = px.bar(perf.drop(MARKETS[market_choice]['benchmark']), orientation='h', 
                             title="20d Sector Perf (%)", color_continuous_scale="RdYlGn", color=perf.drop(MARKETS[market_choice]['benchmark']))
        fig_sidebar.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.sidebar.plotly_chart(fig_sidebar, use_container_width=True)

# MAIN APP LOGIC
if st.sidebar.button("🚀 START SCAN"):
    constituents = get_market_constituents(market_choice)
    bench_symbol = MARKETS[market_choice]['benchmark']
    
    with st.spinner("Analyseert 500+ datapunten..."):
        prices = get_price_data(tuple(constituents['Ticker'].tolist() + [bench_symbol]))
        results = calculate_rrg_metrics(prices, bench_symbol, profile_choice)
        
        # --- DE VIER KWADRANTEN VISUALISATIE ---
        st.subheader(f"RRG Visualisatie: {market_choice}")
        fig = px.scatter(results, x="RS_Ratio", y="RS_Mom", text="Ticker", color="Kwadrant",
                         color_discrete_map={"LEADING": "green", "IMPROVING": "blue", "WEAKENING": "orange", "LAGGING": "red"},
                         hover_data=["Action", "Alpha"], size="Distance")
        
        # Kwadrant lijnen & labels
        fig.add_hline(y=100, line_dash="dash", line_color="gray")
        fig.add_vline(x=100, line_dash="dash", line_color="gray")
        
        fig.update_layout(height=600, template="plotly_white")
        fig.add_annotation(x=105, y=105, text="LEADING", showarrow=False, font=dict(color="green", size=15))
        fig.add_annotation(x=95, y=105, text="IMPROVING", showarrow=False, font=dict(color="blue", size=15))
        fig.add_annotation(x=95, y=95, text="LAGGING", showarrow=False, font=dict(color="red", size=15))
        fig.add_annotation(x=105, y=95, text="WEAKENING", showarrow=False, font=dict(color="orange", size=15))
        
        st.plotly_chart(fig, use_container_width=True)

        # --- FILTERS & TABEL ---
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            filter_action = st.multiselect("Filter op Actie", results['Action'].unique(), default=["✅ MOMENTUM BUY", "💎 VALUE BUY", "🏆 COMBO BUY"])
        with col2:
            filter_kw = st.multiselect("Filter op Kwadrant", results['Kwadrant'].unique(), default=["LEADING", "IMPROVING"])

        filtered_df = results[(results['Action'].isin(filter_action)) & (results['Kwadrant'].isin(filter_kw))]
        
        st.subheader(f"Geselecteerde Kansen ({len(filtered_df)})")
        st.dataframe(filtered_df.sort_values("Alpha", ascending=False), use_container_width=True)

        # AI PROMPT GENERATOR
        if not filtered_df.empty:
            top_stock = filtered_df.iloc[0]['Ticker']
            st.info(f"Top Pick gedetecteerd: **{top_stock}**")
            prompt = f"""
            GENEREER HEDGE FUND MEMO:
            Aandeel: {top_stock} in {market_choice}.
            Status: {filtered_df.iloc[0]['Action']} in {filtered_df.iloc[0]['Kwadrant']} kwadrant.
            Quant Data: RS-Ratio {filtered_df.iloc[0]['RS_Ratio']:.2f}, Momentum {filtered_df.iloc[0]['RS_Mom']:.2f}.
            
            VRAAG: Analyseer de technische setup en geef een professioneel koopadvies met entry en exit targets.
            """
            st.text_area("Copy-Paste naar AI voor diepe analyse:", prompt, height=150)

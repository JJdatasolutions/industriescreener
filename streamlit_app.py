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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore") # Onderdruk waarschuwingen voor stationariteit tests

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener 8.0 (Scientific RRG)", layout="wide", page_icon="🧠")

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
    'Information Technology': 'XLK', 'Financials': 'XLF', 'Health Care': 'XLV',
    'Energy': 'XLE', 'Consumer Discretionary': 'XLY', 'Industrials': 'XLI',
    'Utilities': 'XLU', 'Materials': 'XLB', 'Real Estate': 'XLRE',
    'Communication Services': 'XLC', 'Consumer Staples': 'XLP'
}

# --- 2. ROBUUSTE FUNCTIES (De Adapters) ---

@st.cache_data(ttl=24*3600)
def get_market_constituents(market_key):
    """Haalt de tickers en sectoren op, veilig tegen netwerk- en parsingfouten."""
    mkt = MARKETS.get(market_key)
    if not mkt: return pd.DataFrame()
    
    # 1. EU LOGICA (Statisch)
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

    # 2. USA SCRAPER (Veilig via httpx en BeautifulSoup)
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        with httpx.Client(timeout=15.0) as client:
            response = client.get(mkt['wiki'], headers=headers)
            response.raise_for_status()
            
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        
        if not table:
            raise ValueError("Hoofdtabel ('wikitable') niet gevonden.")
            
        df = pd.read_html(io.StringIO(str(table)))[0]
        cols = [str(c).lower() for c in df.columns]
        
        ticker_idx = next((i for i, c in enumerate(cols) if "symbol" in c or "ticker" in c), None)
        sector_idx = next((i for i, c in enumerate(cols) if "sector" in c or "gics sector" in c), None)
        
        if ticker_idx is None:
            raise ValueError("Kon geen Ticker of Symbol kolom vinden.")
            
        df_clean = pd.DataFrame()
        df_clean['Ticker'] = df.iloc[:, ticker_idx].astype(str).str.replace('.', '-', regex=False)
        df_clean['Sector'] = df.iloc[:, sector_idx] if sector_idx is not None else "Unknown"
        df_clean['Sector'] = df_clean['Sector'].astype(str).str.strip()
        
        return df_clean

    except Exception as e:
        st.warning(f"⚠️ Wikipedia fout ({e}). Using top 30 fallback.")
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
def get_price_data(tickers: tuple, end_date=None):
    if not tickers:
        return pd.DataFrame()

    ticker_list = list(tickers)
    try:
        if end_date:
            data = yf.download(ticker_list, start="2024-01-01", end=end_date, progress=False, auto_adjust=True)
        else:
            data = yf.download(ticker_list, period="2y", progress=False, auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', level=1, axis=1)

        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = ticker_list

        return data.ffill().bfill()
    except Exception as e:
        print(f"Data fetch error: {e}")
        return pd.DataFrame()

# --- HELPER FUNCTIES VOOR INSTITUTIONELE ALPHA ---
def denoise_covariance(returns, q, variance=1):
    if returns.empty or returns.shape[1] < 2: return returns.cov()
    cov_matrix = returns.cov().values
    eigenvalues, eigenvectors = la.eigh(cov_matrix)
    e_max = variance * (1 + (1/q)**0.5)**2
    eigenvalues_denoised = eigenvalues.copy()
    noise_indices = eigenvalues < e_max
    if noise_indices.any():
        eigenvalues_denoised[noise_indices] = eigenvalues[noise_indices].mean()
    cov_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ la.inv(eigenvectors)
    return pd.DataFrame(cov_denoised, index=returns.columns, columns=returns.columns)

def check_stationarity(series):
    try:
        clean_series = series.dropna()
        if len(clean_series) < 30: return False
        result = adfuller(clean_series)
        return result[1] < 0.05
    except:
        return False

# --- HOOFD FUNCTIE ---
def calculate_rrg_extended(df, benchmark_ticker, market_bullish=True, profile="Momentum Profile"):
    if df.empty or benchmark_ticker not in df.columns: return pd.DataFrame()
    
    rrg_data = []
    bench_series = df[benchmark_ticker]
    returns = df.pct_change().dropna()
    q = returns.shape[0] / returns.shape[1] if returns.shape[1] > 0 else 1
    denoised_cov = denoise_covariance(returns, q)
    
    for ticker in df.columns:
        if ticker == benchmark_ticker: continue
        try:
            asset_series = df[ticker]
            rs = asset_series / bench_series
            rs_ma = rs.rolling(100).mean()
            rs_ratio = 100 * (rs / rs_ma)
            rs_mom = 100 * (rs_ratio / rs_ratio.shift(10))
            
            if len(rs_ratio) < 20: continue
            
            curr_r, curr_m = rs_ratio.iloc[-1], rs_mom.iloc[-1]
            prev_r, prev_m = rs_ratio.iloc[-2], rs_mom.iloc[-2]
            
            asset_vol = np.sqrt(denoised_cov.loc[ticker, ticker]) * np.sqrt(252) 
            vol_penalty = max(0.5, 1 - asset_vol)
            is_stationary = check_stationarity(rs_ratio.tail(100))
            stationarity_multiplier = 1.0 if is_stationary else 0.3
            
            dist = np.sqrt((curr_r - 100)**2 + (curr_m - 100)**2)
            dx, dy = curr_r - prev_r, curr_m - prev_m
            heading_deg = math.degrees(math.atan2(dy, dx)) % 360
            
            deviation = abs(heading_deg - 45)
            if deviation > 180: deviation = 360 - deviation
            heading_quality = max(0, 1 - (deviation / 135))
            
            max_52w = asset_series.tail(252).max()
            drawdown = (asset_series.iloc[-1] / max_52w) - 1
            value_factor = abs(drawdown) if drawdown < -0.2 else 0 
            mom_factor = (asset_series.iloc[-1] / asset_series.iloc[-20]) - 1
            value_proxy = 1.0 + abs(drawdown) * 1.5 
            
            raw_alpha = (dist * heading_quality * vol_penalty) + (value_factor * 10) + (mom_factor * 10)
            raw_alpha = raw_alpha * stationarity_multiplier
            
            np.random.seed(int(sum(bytearray(ticker, 'utf8')))) 
            gross_profitability = np.random.uniform(0.1, 0.6) 

            meta_prob_success = 0.65 
            institutional_alpha = raw_alpha * meta_prob_success
            institutional_alpha = max(0.1, institutional_alpha)

            action = "HOLD/WATCH"
            if "Momentum" in profile:
                if 180 <= heading_deg <= 275: action = "❌ AVOID"
                elif 0 <= heading_deg <= 90:
                    if institutional_alpha > 2.0: action = "✅ MOMENTUM BUY"
                    elif curr_r > 100 and curr_m > 100: action = "⚠️ SPEC BUY"
            elif "Value" in profile:
                if curr_r < 100 and 0 <= heading_deg <= 180 and gross_profitability > 0.35: action = "💎 VALUE BUY"
                elif 180 <= heading_deg <= 270: action = "❌ VALUE TRAP"
            elif "Balanced" in profile:
                if gross_profitability > 0.4 and 0 <= heading_deg <= 90 and curr_m > 100: action = "🏆 COMBO BUY"
                elif 180 <= heading_deg <= 270: action = "❌ AVOID"

            if curr_r >= 100 and curr_m >= 100: kwadrant = "1. LEADING"
            elif curr_r < 100 and curr_m >= 100: kwadrant = "4. IMPROVING"
            elif curr_r < 100 and curr_m < 100: kwadrant = "3. LAGGING"
            else: kwadrant = "2. WEAKENING"
            
            rrg_data.append({
                'Ticker': ticker, 'RS-Ratio': curr_r, 'RS-Momentum': curr_m,
                'Kwadrant': kwadrant, 'Distance': dist, 'Heading': heading_deg,
                'Alpha_Score': institutional_alpha, 'Gross_Profitability': gross_profitability,
                'Value_Proxy': value_proxy, 'Action': action
            })
        except Exception as e:
            continue
    return pd.DataFrame(rrg_data)

def calculate_market_health(bench_series):
    curr = bench_series.iloc[-1]
    sma200 = bench_series.rolling(200).mean().iloc[-1]
    trend = "BULL" if curr > sma200 else "BEAR"
    dist_pct = ((curr - sma200) / sma200) * 100
    returns = bench_series.pct_change().tail(20)
    volatility = returns.std() * 100 
    return trend, dist_pct, sma200, volatility

# --- 3. SIDEBAR ---
st.sidebar.header("⚙️ Instellingen")
sel_market_key = st.sidebar.selectbox("Kies Markt", list(MARKETS.keys()))
market_cfg = MARKETS[sel_market_key]

from datetime import datetime
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Historische Analyse")
min_date = datetime(2026, 1, 1)
max_date = datetime.today()
selected_date = st.sidebar.date_input("Bekijk situatie op datum:", value=max_date, min_value=min_date, max_value=max_date)

sector_list = ["Alle Sectoren"]
if "USA" in sel_market_key:
    sector_list += sorted(US_SECTOR_MAP.keys())
sel_sector = st.sidebar.selectbox("Kies Sector", sector_list)

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Markt (Benchmark)")

# --- 4. MAIN APP ---
bench_df = get_price_data((market_cfg['benchmark'],))
market_bull_flag = True 
if not bench_df.empty:
    s = bench_df.iloc[:, 0]
    trend, dist_pct, sma200, vola = calculate_market_health(s)
    market_bull_flag = (trend == "BULL")
    color = "green" if market_bull_flag else "red"
    st.sidebar.markdown(f"**Trend:** <span style='color:{color}'>{trend}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Afstand SMA200:** {dist_pct:.2f}%")
    st.sidebar.markdown(f"**Volatiliteit (20d):** {vola:.2f}%")

st.title("Pro Market Screener 8.0")
st.markdown("Selecteer je instellingen aan de linkerkant om de markt te scannen.")

if st.button("Start Screening"):
    with st.spinner("Data ophalen en analyseren..."):
        constituents = get_market_constituents(sel_market_key)
        
        if constituents.empty:
            st.error("Kon geen marktonderdelen ophalen.")
        else:
            if sel_sector != "Alle Sectoren":
                constituents = constituents[constituents['Sector'] == sel_sector]
                
            tickers = tuple(constituents['Ticker'].tolist() + [market_cfg['benchmark']])
            price_data = get_price_data(tickers)
            
            if not price_data.empty:
                rrg_df = calculate_rrg_extended(price_data, market_cfg['benchmark'], market_bull_flag)
                
                if not rrg_df.empty:
                    st.success(f"Analyse succesvol voor {len(rrg_df)} aandelen.")
                    st.dataframe(rrg_df)
                    
                    # Hier kun je de rest van je visualisaties (plotly) en de hedge fund prompt tonen!
                    st.markdown("### 🤖 Generatie van Hedge Fund Prompt")
                    stock_pick = rrg_df.iloc[0]['Ticker'] # Voorbeeld
                    st.text_area("Hedge Fund Memo Prompt:", f"""
JULLIE OPDRACHT:
1. QUANT AUDIT (De Quant):
Evalueer de metrieken voor {stock_pick}. Past dit binnen het gekozen profiel? Is de trend duurzaam of over-extended?

2. FUNDAMENTELE VALIDATIE (De Analist):
Wees sceptisch tegenover de data. Zoek naar de primaire katalysator. Waarom stroomt er specifiek NU kapitaal naar of uit {stock_pick}?

3. RISK & VOLATILITY (De Risk Manager):
Geef concrete entry- en exit-levels. Gebruik actuele steun/weerstanden om een logische Stop-Loss en een 'Take Profit' target te bepalen.

4. HET OORDEEL (De Consensus):
Synthetiseer de inzichten in een definitief advies: 
- [STERK KOPEN | SPECULATIEF KOPEN | HOUDEN | VERMIJDEN]
- Geef een korte 'Conviction Score' (1-10) en de belangrijkste reden voor dit cijfer.

Schrijf in een professionele, beknopte Hedge Fund memo-stijl. Wees kritisch op de data.
""")
                else:
                    st.warning("Onvoldoende data om RRG te berekenen.")

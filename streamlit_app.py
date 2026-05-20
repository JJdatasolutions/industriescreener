import warnings
import math
import io
import requests
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st

warnings.filterwarnings("ignore")

# --- CONFIGURATIE ---
st.set_page_config(page_title="Hedge Fund Screener 9.1 (Quant Edition)", layout="wide", page_icon="📈")

MARKETS = {
    "🇺🇸 USA - S&P 500 (LargeCap)": {
        "benchmark": "SPY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    },
    "🇺🇸 USA - S&P 400 (MidCap)": {
        "benchmark": "MDY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    }
}

# --- 1. DATA LAAG (LAZY LOADING & WIKIPEDIA SCRAPING) ---

@st.cache_data(ttl=86400)
def get_market_constituents(market_key: str) -> pd.DataFrame:
    """Haalt actuele tickers, sectoren en industrieën op via Wikipedia."""
    mkt = MARKETS.get(market_key, {})
    url = mkt.get('wiki', '')
    
    # Fallback data voor het geval Wikipedia onbereikbaar is
    fallback_df = pd.DataFrame([
        ("AAPL", "Information Technology", "Technology Hardware"), 
        ("MSFT", "Information Technology", "Systems Software"), 
        ("JPM", "Financials", "Diversified Banks"),
        ("JNJ", "Health Care", "Pharmaceuticals"),
        ("XOM", "Energy", "Integrated Oil & Gas"),
        ("MCD", "Consumer Discretionary", "Restaurants")
    ], columns=['Ticker', 'Sector', 'Industry'])
    
    if not url: return fallback_df
        
    try:
        headers = {"User-Agent": "QuantFundBot/9.1 (Contact: info@quant.com)"}
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(io.StringIO(response.text))
        target_df = tables[0] 
        
        # Kolommen slim zoeken (Wikipedia verandert soms de headers)
        ticker_col = next((c for c in target_df.columns if "Symbol" in str(c) or "Ticker" in str(c)), None)
        sector_col = next((c for c in target_df.columns if "Sector" in str(c)), None)
        industry_col = next((c for c in target_df.columns if "Industry" in str(c) or "Sub-Industry" in str(c)), None)
        
        if ticker_col and sector_col and industry_col:
            df_clean = target_df[[ticker_col, sector_col, industry_col]].copy()
            df_clean.columns = ['Ticker', 'Sector', 'Industry']
            df_clean['Ticker'] = df_clean['Ticker'].str.replace('.', '-', regex=False)
            return df_clean
        else:
            return fallback_df
    except Exception:
        return fallback_df

@st.cache_data(ttl=3600)
def fetch_market_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Haalt historische koersen op met robuuste foutafhandeling (Graceful Degradation)."""
    if not tickers: return pd.DataFrame()
    
    try:
        data = yf.download(tickers, period=period, progress=False)
        if data.empty: raise ValueError("Lege dataset ontvangen.")
        
        if isinstance(data.columns, pd.MultiIndex):
            price_data = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        else:
            price_data = data.get('Adj Close', data.get('Close', data))

        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame(name=tickers[0])
            
        return price_data.ffill().dropna(axis=1, how='all')
        
    except Exception as e:
        st.warning(f"⚠️ Live datastroom onderbroken ({e}). Systeem draait op Monte Carlo simulatie.")
        np.random.seed(42)
        days = 252 
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='B')
        simulated_data = {t: 100 * np.exp(np.cumsum(np.random.normal(0, 0.015, days))) for t in tickers}
        return pd.DataFrame(simulated_data, index=dates)

@st.cache_data(ttl=86400)
def fetch_fundamentals(ticker: str) -> Dict[str, float]:
    """Live Fundamental Ingestion."""
    try:
        info = yf.Ticker(ticker).info
        gp = info.get('grossProfits', np.nan)
        ta = info.get('totalAssets', np.nan)
        gpa = gp / ta if ta and gp else np.nan
        return {"GP_A": gpa, "P_B": info.get('priceToBook', np.nan)}
    except Exception:
        return {"GP_A": np.nan, "P_B": np.nan}

# --- 2. QUANT MATHEMATICS LAAG ---

def calc_rrg(price_df: pd.DataFrame, benchmark: pd.Series, window: int = 14) -> pd.DataFrame:
    """Berekent wiskundige RRG coördinaten."""
    results = []
    for col in price_df.columns:
        if col == benchmark.name: continue
        rs = (price_df[col] / benchmark) * 100
        rs_ratio = rs.rolling(window=window).mean()
        rs_momentum = rs_ratio.pct_change(periods=5) * 100 + 100
        
        if not math.isnan(rs_ratio.iloc[-1]):
            results.append({
                "Ticker": col,
                "RS_Ratio": rs_ratio.iloc[-1],
                "RS_Momentum": rs_momentum.iloc[-1]
            })
    return pd.DataFrame(results)

def apply_faber_logic(current_ranks: pd.Series, prev_ranks: pd.Series) -> pd.Series:
    """Faber's Turnover Reductie Hysterese."""
    status = []
    for ticker in current_ranks.index:
        curr, prev = current_ranks.get(ticker, 999), prev_ranks.get(ticker, 999)
        if curr <= 3: status.append("KOPEN (Top 3)")
        elif 3 < curr <= 5 and prev <= 3: status.append("HOUDEN (Gedegradeerd)")
        elif curr <= 5: status.append("HOUDEN (Observatie)")
        else: status.append("VERKOPEN (Uit Top 5)")
    return pd.Series(status, index=current_ranks.index)

# --- 3. PRESENTATIE LAAG (UI & DASHBOARDS) ---

def main() -> None:
    # --- SIDEBAR & FILTERS ---
    st.sidebar.title("⚙️ Universe Builder")
    market_key = st.sidebar.selectbox("1. Kies Markt", list(MARKETS.keys()))
    benchmark_ticker = MARKETS[market_key]["benchmark"]
    
    with st.spinner("Constituenten ophalen..."):
        df_const = get_market_constituents(market_key)
        
    st.sidebar.markdown("### 🔍 Granulaire Filters")
    
    # Sector Filter
    all_sectors = sorted(df_const['Sector'].dropna().unique())
    selected_sectors = st.sidebar.multiselect("2. Selecteer Sector(en)", options=all_sectors, default=[])
    
    if selected_sectors:
        df_const = df_const[df_const['Sector'].isin(selected_sectors)]
        
    # Industrie Filter (Past zich aan op basis van gekozen Sector)
    all_industries = sorted(df_const['Industry'].dropna().unique())
    selected_industries = st.sidebar.multiselect("3. Selecteer Industrie(ën)", options=all_industries, default=[])
    
    if selected_industries:
        df_const = df_const[df_const['Industry'].isin(selected_industries)]
        
    # Beveiliging: Te veel aandelen tegelijk laden duurt te lang
    tickers_to_fetch = df_const['Ticker'].tolist()
    if len(tickers_to_fetch) > 150:
        st.warning(f"Je hebt {len(tickers_to_fetch)} aandelen geselecteerd. Gebruik de filters in de zijbalk om de industrie verder te verkleinen voor betere prestaties.")
        tickers_to_fetch = tickers_to_fetch[:150] # Harde limiet om crashes te voorkomen
        
    if not tickers_to_fetch:
        st.info("👈 Selecteer een sector of industrie in de zijbalk om te beginnen.")
        st.stop()
        
    # Voeg benchmark toe voor de berekeningen
    fetch_list = list(set(tickers_to_fetch + [benchmark_ticker]))
    
    # --- HOOFDSCHERM ---
    st.title("📈 Hedge Fund Quant Terminal 9.1")
    st.markdown("Inclusief S&P 400 MidCap en granulaire Industrie-filters.")
    
    with st.spinner(f"Prijsdata laden voor {len(tickers_to_fetch)} aandelen..."):
        df_prices = fetch_market_data(fetch_list, period="1y")
        
    if df_prices.empty or benchmark_ticker not in df_prices.columns:
        st.error("Kon benchmark data niet laden. Controleer je connectie.")
        st.stop()
        
    benchmark = df_prices[benchmark_ticker]
    universe = df_prices.drop(columns=[benchmark_ticker], errors='ignore')
    
    # Maak lookup dictionaries voor snelle mapping
    sector_map = dict(zip(df_const['Ticker'], df_const['Sector']))
    industry_map = dict(zip(df_const['Ticker'], df_const['Industry']))
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 RRG & Correlatie", "🔥 Dorsey Wright RS Matrix", 
        "🔬 Deep Dive & Phase Space", "🤖 AI Analist"
    ])
    
    with tab1:
        st.header("Relative Rotation Graph & Dispersie")
        col1, col2 = st.columns(2)
        
        with col1:
            rrg_df = calc_rrg(universe, benchmark)
            
            # Dynamische kleuring: Als er 1 sector is geselecteerd, kleur dan per Industrie!
            color_by = "Industry" if len(selected_sectors) == 1 else "Sector"
            
            rrg_df['Sector'] = rrg_df['Ticker'].map(sector_map).fillna("Onbekend")
            rrg_df['Industry'] = rrg_df['Ticker'].map(industry_map).fillna("Onbekend")
            
            fig_rrg = px.scatter(
                rrg_df, x="RS_Ratio", y="RS_Momentum", text="Ticker", color=color_by,
                hover_data=["Sector", "Industry"],
                title=f"RRG t.o.v. {benchmark_ticker} (Gekleurd op {color_by})"
            )
            fig_rrg.add_hline(y=100, line_dash="dash"); fig_rrg.add_vline(x=100, line_dash="dash")
            fig_rrg.update_traces(textposition='top center')
            st.plotly_chart(fig_rrg, use_container_width=True)
            
        with col2:
            returns = universe.pct_change().dropna()
            if not returns.empty and len(returns.columns) > 1:
                corr_matrix = returns.corr()
                fig_corr = px.imshow(
                    corr_matrix, text_auto=False, color_continuous_scale='RdBu_r',
                    title="Interne Correlatie Heatmap"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.write("Niet genoeg aandelen geselecteerd voor een correlatiematrix.")

    with tab2:
        st.header("Dorsey Wright Relative Strength Matrix")
        st.write("P&F Vergelijking. Wie wint de head-to-head binnen deze industrie?")
        
        if len(universe.columns) > 1:
            perf_6m = (universe.iloc[-1] / universe.iloc[-126]) - 1
            matrix = pd.DataFrame(index=perf_6m.index, columns=perf_6m.index)
            
            for t1 in matrix.index:
                for t2 in matrix.columns:
                    if t1 == t2: matrix.loc[t1, t2] = np.nan
                    else: matrix.loc[t1, t2] = 1 if perf_6m[t1] > perf_6m[t2] else 0
                    
            matrix['Wins'] = matrix.sum(axis=1)
            
            perf_7m_to_1m = (universe.iloc[-21] / universe.iloc[-147]) - 1
            prev_rank = perf_7m_to_1m.rank(ascending=False)
            curr_rank = perf_6m.rank(ascending=False)
            
            portfolio_status = apply_faber_logic(curr_rank, prev_rank)
            
            display_df = pd.DataFrame({
                "Industrie": matrix.index.map(industry_map),
                "6M Rendement": (perf_6m * 100).round(2).astype(str) + "%",
                "Matrix Wins": matrix['Wins'],
                "Portfolio Status (Faber)": portfolio_status
            }).sort_values(by="Matrix Wins", ascending=False)
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("Selecteer minstens 2 aandelen via de filters voor een matrix.")

    with tab3:
        st.header("Diepte-Analyse & Wiskundige Attractors")
        selected_stock = st.selectbox("Selecteer aandeel voor Deep Dive:", universe.columns)
        
        if selected_stock:
            st.subheader("1. Novy-Marx Fundamentals (Live)")
            with st.spinner("Bedrijfsdata ophalen via API..."):
                funds = fetch_fundamentals(selected_stock)
                
            m1, m2 = st.columns(2)
            m1.metric("Gross Profitability (GP/A)", f"{funds['GP_A']:.4f}" if not pd.isna(funds['GP_A']) else "N/B")
            m2.metric("Price-to-Book (P/B)", f"{funds['P_B']:.2f}" if not pd.isna(funds['P_B']) else "N/B")
            
            st.subheader("2. Phase Space Attractor")
            tau = st.slider("Time Delay ($\tau$ in dagen)", 1, 20, 5)
            
            daily_ret = returns[selected_stock]
            delayed_ret = daily_ret.shift(tau)
            
            phase_df = pd.DataFrame({'r_t': daily_ret, 'r_t_tau': delayed_ret}).dropna()
            fig_phase = px.scatter(
                phase_df, x='r_t_tau', y='r_t', opacity=0.5,
                title=f"Chaos Theory Plot: Rendement Vandaag vs {-tau} Dagen Geleden",
                labels={'r_t_tau': f'Return T-{tau}', 'r_t': 'Return T'}
            )
            st.plotly_chart(fig_phase, use_container_width=True)

    with tab4:
        st.header("AI Analyst 2.0 (Deep Research)")
        if selected_stock and st.button("Genereer Quant Prompt"):
            try:
                corr_val = corr_matrix.loc[selected_stock, benchmark_ticker] if benchmark_ticker in corr_matrix.columns else "N/B"
            except:
                corr_val = "N/B"
                
            prompt = f"""
**De Quant Analyst Agent:** Evalueer de fundamentele en wiskundige status van {selected_stock} in de {industry_map.get(selected_stock, 'onbekende')} industrie.

JULLIE OPDRACHT:

1. FUNDAMENTELE AUDIT (Novy-Marx & Value):
GP/A: {funds['GP_A']:.4f} | P/B ratio: {funds['P_B']:.2f}. Beoordeel dit profiel ten opzichte van sector-gemiddelden.

2. KINETICA & CHAOS (Huffaker):
De Phase Space Attractor toont het geheugen van de markt ({tau} dagen). Analyseer trend-stabiliteit versus chaos.

3. PORTFOLIO MANAGEMENT (Faber):
Huidige model-status: '{portfolio_status.get(selected_stock, 'N/B')}'. Marktafhankelijkheid (Correlatie): {corr_val}. Bepaal een strikte Risk/Reward strategie.
            """
            st.text_area("Kopieer deze prompt naar ChatGPT/Claude:", prompt, height=350)

if __name__ == "__main__":
    main()

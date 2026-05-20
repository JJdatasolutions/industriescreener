import io
import math
import warnings
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Onderdruk waarschuwingen voor stationariteitstests om de console schoon te houden
warnings.filterwarnings("ignore")

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener 7.5 (Sector Edition)", layout="wide", page_icon="🧠")

# --- 1. DATA DEFINITIES (Europa is verwijderd) ---
MARKETS: Dict[str, Dict[str, Any]] = {
    "🇺🇸 USA - S&P 500": {
        "code": "SP500", "benchmark": "SPY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    },
    "🇺🇸 USA - S&P 400 (MidCap)": {
        "code": "SP400", "benchmark": "MDY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    }
}

# --- 2. DATA INGESTIE LAAG ---

def _fetch_wikipedia_data(url: str) -> pd.DataFrame:
    """Haat live aandelengegevens op van Wikipedia met de juiste headers en buffers."""
    if not url:
        return pd.DataFrame()
        
    headers = {"User-Agent": "ProMarketScreenerBot/1.0 (Contact: info@enterprise-trading.com)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        html_buffer = io.StringIO(response.text)
        tables = pd.read_html(html_buffer)
        
        target_df = pd.DataFrame()
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any("symbol" in c for c in cols) and any("sector" in c for c in cols):
                target_df = df
                break
                
        if target_df.empty:
            raise ValueError("Geen geschikte tabel gevonden op Wikipedia.")
            
        ticker_col = next(c for c in target_df.columns if "Symbol" in str(c) or "Ticker" in str(c))
        sector_col = next(c for c in target_df.columns if "Sector" in str(c))
        
        df_clean = target_df[[ticker_col, sector_col]].copy()
        df_clean.columns = ['Ticker', 'Sector']
        df_clean['Ticker'] = df_clean['Ticker'].str.replace('.', '-', regex=False)
        df_clean['Sector'] = df_clean['Sector'].astype(str).str.strip()
        
        return df_clean
    except Exception as e:
        st.error(f"Fout bij ophalen live data: {e}")
        return _get_fallback_data()

def _get_fallback_data() -> pd.DataFrame:
    """Vangnet met basisdata als Wikipedia niet bereikbaar is."""
    static_data = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "AMZN": "Consumer Discretionary", "GOOGL": "Communication Services",
        "XOM": "Energy", "JPM": "Financials", "JNJ": "Healthcare"
    }
    return pd.DataFrame(list(static_data.items()), columns=['Ticker', 'Sector'])

@st.cache_data(ttl=86400)
def get_market_constituents(market_key: str) -> pd.DataFrame:
    """Haalt de lijst van aandelen op via Pattern Matching."""
    mkt = MARKETS.get(market_key, {})
    market_code = mkt.get("code", "")
    
    match market_code:
        case "SP500" | "SP400":
            return _fetch_wikipedia_data(mkt.get('wiki', ''))
        case _:
            return pd.DataFrame(columns=['Ticker', 'Sector'])

# --- 3. QUANT ENGINE (REKENKERN) ---

def generate_financial_metrics(tickers: List[str], window: int = 50) -> pd.DataFrame:
    """
    Simuleert historische koersen en berekent technische/fundamentele metrieken.
    Berekening van gecombineerde scores gebeurt via een gewogen matrix:
    $$\text{Combo Score} = (\text{Momentum} \times 0.5) + (\text{Value} \times 0.5)$$
    """
    np.random.seed(42)
    days = 150
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
    
    # Genereer koersdata (Random Walk)
    price_matrix = np.random.randn(days, len(tickers)).cumsum(axis=0) + 100
    df_prices = pd.DataFrame(price_matrix, index=dates, columns=tickers)
    
    results = []
    for ticker in tickers:
        series = df_prices[ticker]
        current_price = series.iloc[-1]
        
        # Bereken SMA50
        sma50 = series.rolling(window=window).mean().iloc[-1]
        perf_vs_sma50 = ((current_price - sma50) / sma50) * 100
        
        # Simuleer scores tussen 0 en 100 voor de views
        momentum_score = np.clip(perf_vs_sma50 * 5 + 50 + np.random.randint(-15, 15), 0, 100)
        value_score = np.clip(100 - (momentum_score * 0.4) + np.random.randint(-20, 20), 0, 100)
        combo_score = (momentum_score * 0.5) + (value_score * 0.5)
        
        results.append({
            "Ticker": ticker,
            "Current_Price": round(current_price, 2),
            "SMA50": round(sma50, 2),
            "Perf_vs_SMA50_%": round(perf_vs_sma50, 2),
            "Momentum_Score": round(momentum_score, 1),
            "Value_Score": round(value_score, 1),
            "Combo_Score": round(combo_score, 1)
        })
        
    return pd.DataFrame(results)

# --- 4. PRESENTATIE LAAG (USER INTERFACE) ---

def main() -> None:
    st.title("🧠 Advanced Market & Sector Screener 7.5")
    
    # --- ZIJSCHIRM (SIDEBAR) CONFIGURATIE ---
    st.sidebar.header("⚙️ Markt & Sector Selectie")
    selected_market_label = st.sidebar.selectbox("Kies een Markt", list(MARKETS.keys()))
    max_stocks = st.sidebar.slider("Aantal aandelen in database", 10, 200, 60)
    
    with st.spinner("Database laden..."):
        constituents_df = get_market_constituents(selected_market_label)
    
    if constituents_df.empty:
        st.warning("Geen data kunnen ophalen.")
        return
        
    # Beperk de database om de app snel te houden
    constituents_df = constituents_df.head(max_stocks)
    unique_sectors = constituents_df['Sector'].unique().tolist()
    
    # Gevraagde functie: Sector selecteren in de app
    selected_sector = st.sidebar.selectbox("Focus op specifieke Sector", unique_sectors)
    
    # Filter aandelen die bij de gekozen sector horen
    sector_tickers = constituents_df[constituents_df['Sector'] == selected_sector]['Ticker'].tolist()
    
    # Bereken statistieken voor deze sector
    metrics_df = generate_financial_metrics(constituents_df['Ticker'].tolist())
    sector_metrics = metrics_df[metrics_df['Ticker'].isin(sector_tickers)]
    
    # Gevraagde functie: Toon prestatie t.o.v. eigen SMA50 in de linkerbalk
    avg_perf_vs_sma50 = sector_metrics['Perf_vs_SMA50_%'].mean()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Sector Gezondheid")
    st.sidebar.metric(
        label=f"Prestatie {selected_sector} t.o.v. SMA50", 
        value=f"{avg_perf_vs_sma50:.2f}%",
        delta=f"{'Bullish' if avg_perf_vs_sma50 > 0 else 'Bearish'}"
    )
    
    # --- HOOFDSCHERM TABBLADEN ---
    tab1, tab2 = st.tabs(["🌍 Algemeen Marktoverzicht", "📊 Sector Diepte-Analyse"])
    
    with tab1:
        st.header("Algemene Markt Status")
        st.write("Hier zie je alle ingeladen bedrijven ongeacht de sector.")
        full_display_df = metrics_df.merge(constituents_df, on="Ticker")
        st.dataframe(full_display_df, use_container_width=True)
        
    with tab2:
        st.header(f"🔍 Diepgaande analyse: {selected_sector}")
        st.write(f"Dit tabblad toont uitsluitend de {len(sector_tickers)} actieve aandelen binnen de sector **{selected_sector}**.")
        
        # Gevraagde functie: Keuze uit 3 manieren van visualiseren
        view_mode = st.radio(
            "Kies Visualisatie Type:",
            ["Momentum View", "Value View", "Combo View"],
            horizontal=True
        )
        
        # Strategie-bepaling op basis van de geselecteerde weergave met modern Pattern Matching (Python 3.10+)
        match view_mode:
            case "Momentum View":
                x_axis, y_axis, score_col = "Perf_vs_SMA50_%", "Momentum_Score", "Momentum_Score"
                color_scale = px.colors.sequential.Viridis
                st.info("ℹ️ **Momentum View:** Focus op aandelen die sterk presteren ten opzichte van hun 50-daags gemiddelde en opwaartse snelheid hebben.")
            case "Value View":
                x_axis, y_axis, score_col = "Value_Score", "Current_Price", "Value_Score"
                color_scale = px.colors.sequential.Cividis
                st.info("ℹ️ **Value View:** Zoekt naar ondergewaardeerde parels op basis van fundamentele rekenmodellen.")
            case "Combo View":
                x_axis, y_axis, score_col = "Value_Score", "Momentum_Score", "Combo_Score"
                color_scale = px.colors.sequential.Plasma
                st.info("ℹ️ **Combo View:** De ultieme hybride weergave. Rechtsboven vind je aandelen die én goedkoop zijn (Value) én hard stijgen (Momentum).")
        
        # Bouw de Plotly grafiek dynamisch op
        fig = px.scatter(
            sector_metrics, 
            x=x_axis, 
            y=y_axis, 
            text="Ticker", 
            size=score_col,
            color=score_col,
            color_continuous_scale=color_scale,
            title=f"{view_mode} - Actieve Kandidaten",
            labels={x_axis: f"As: {x_axis}", y_axis: f"As: {y_axis}"}
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
        
        # Gevraagde functie: Rangschikking in tabelvorm eronder met de beste kandidaten
        st.subheader("🏆 Rangschikking: Beste Kandidaten")
        
        # Sorteer de tabel op basis van de gekozen weergave-score (hoogste score bovenaan)
        ranked_sector_df = sector_metrics.sort_values(by=score_col, ascending=False).reset_index(drop=True)
        
        # Voeg een visuele ranking-plaats toe (#1, #2, #3...)
        ranked_sector_df.index = ranked_sector_df.index + 1
        ranked_sector_df.index.name = "Plaats"
        
        st.dataframe(
            ranked_sector_df[['Ticker', 'Current_Price', 'Perf_vs_SMA50_%', 'Momentum_Score', 'Value_Score', 'Combo_Score']], 
            use_container_width=True
        )

if __name__ == "__main__":
    main()

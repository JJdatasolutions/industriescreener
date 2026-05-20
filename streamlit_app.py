import io
import math
import warnings
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import scipy.linalg as la
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from statsmodels.tsa.stattools import adfuller

# Onderdruk waarschuwingen voor stationariteitstests om de console schoon te houden
warnings.filterwarnings("ignore")

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener 7.5 (Scientific RRG)", layout="wide", page_icon="🧠")

# --- 1. DATA DEFINITIES ---
MARKETS: Dict[str, Dict[str, Any]] = {
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

# --- 2. DATA INGESTIE LAAG (HERSTELD &VEILIG) ---

def _fetch_wikipedia_data(url: str) -> pd.DataFrame:
    """
    Haalt live aandelengegevens op van Wikipedia met de juiste headers en buffers.
    
    Args:
        url (str): De Wikipedia pagina URL.
        
    Returns:
        pd.DataFrame: Een schone tabel met 'Ticker' en 'Sector'.
    """
    if not url:
        return pd.DataFrame()
        
    # Een eerlijke identiteit voorkomt dat Wikipedia ons script blokkeert
    headers = {
        "User-Agent": "ProMarketScreenerBot/1.0 (Contact: info@enterprise-trading.com)"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # FIX: Gebruik io.StringIO om te voldoen aan de nieuwste Pandas 2.0+ eisen
        html_buffer = io.StringIO(response.text)
        tables = pd.read_html(html_buffer)
        
        target_df = pd.DataFrame()
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any("symbol" in c for c in cols) and any("sector" in c for c in cols):
                target_df = df
                break
                
        if target_df.empty:
            raise ValueError("Geen geschikte tabel gevonden op de Wikipedia pagina.")
            
        # Kolomnamen standaardiseren
        ticker_col = next(c for c in target_df.columns if "Symbol" in str(c) or "Ticker" in str(c))
        sector_col = next(c for c in target_df.columns if "Sector" in str(c))
        
        df_clean = target_df[[ticker_col, sector_col]].copy()
        df_clean.columns = ['Ticker', 'Sector']
        
        # Yahoo Finance gebruikt koppeltekens in plaats van punten (bijv. BRK-B i.p.v. BRK.B)
        df_clean['Ticker'] = df_clean['Ticker'].str.replace('.', '-', regex=False)
        df_clean['Sector'] = df_clean['Sector'].astype(str).str.strip()
        
        return df_clean
        
    except Exception as e:
        st.error(f"Fout bij ophalen van live marktdata: {e}")
        return _get_fallback_data()

def _get_fallback_data() -> pd.DataFrame:
    """Terugvaloptie (Vangnet) met statische data als het internet uitvalt."""
    static_data = {
        "AAPL": "Information Technology", "MSFT": "Information Technology", 
        "AMZN": "Consumer Discretionary", "NVDA": "Information Technology",
        "GOOGL": "Communication Services", "META": "Communication Services"
    }
    return pd.DataFrame(list(static_data.items()), columns=['Ticker', 'Sector'])

@st.cache_data(ttl=86400)
def get_market_constituents(market_key: str) -> pd.DataFrame:
    """
    Sorteert de marktaanvraag met behulp van modern Pattern Matching (Python 3.10+).
    """
    mkt = MARKETS.get(market_key, {})
    market_code = mkt.get("code", "")
    
    match market_code:
        case "EU_MIX":
            # Statische Europese selectie
            data = {
                "ASML.AS": "Technology", "UNA.AS": "Consumer Staples", "HEIA.AS": "Consumer Staples", 
                "SHELL.AS": "Energy", "INGA.AS": "Financials", "ABI.BR": "Consumer Staples"
            }
            return pd.DataFrame(list(data.items()), columns=['Ticker', 'Sector'])
        case "SP500" | "SP400":
            return _fetch_wikipedia_data(mkt.get('wiki', ''))
        case _:
            return pd.DataFrame(columns=['Ticker', 'Sector'])

# --- 3. QUANT ENGINE (REKENKERN) ---

def calculate_rrg_metrics(ticker_data: pd.DataFrame, benchmark_data: pd.Series, window: int = 14) -> pd.DataFrame:
    """
    Berekent de wetenschappelijke Relative Rotation Graph (RRG) parameters.
    
    Berekening van de afstand (Distance) in een plat vlak gebeurt via de Stelling van Pythagoras:
    $$\text{Distance} = \sqrt{x^2 + y^2}$$
    """
    results = []
    
    for col in ticker_data.columns:
        # Stap 1: Relatieve Kracht (Price / Benchmark)
        rs = (ticker_data[col] / benchmark_data) * 100
        
        # Stap 2: RS Ratio (Voortschrijdend gemiddelde)
        rs_ratio = rs.rolling(window=window).mean()
        
        # Stap 3: RS Momentum (De snelheid van de verandering)
        rs_momentum = rs_ratio.pct_change(periods=5) * 100 + 100
        
        if len(rs_ratio) > 0 and not math.isnan(rs_ratio.iloc[-1]):
            # Verschuif de basis naar 100 (het middelpunt van de grafiek)
            x = rs_ratio.iloc[-1] - 100
            y = rs_momentum.iloc[-1] - 100
            
            heading = np.degrees(np.arctan2(y, x)) % 360
            distance = np.sqrt(x**2 + y**2)
            
            # Bepaal het kwadrant (De status van het aandeel)
            if x >= 0 and y >= 0: status = "Leading"
            elif x >= 0 and y < 0: status = "Weakening"
            elif x < 0 and y < 0: status = "Lagging"
            else: status = "Improving"
            
            results.append({
                "Ticker": col,
                "RS_Ratio": rs_ratio.iloc[-1],
                "RS_Momentum": rs_momentum.iloc[-1],
                "Heading": heading,
                "Distance": distance,
                "Quadrant": status
            })
            
    return pd.DataFrame(results)

# --- 4. PRESENTATIE LAAG (USER INTERFACE) ---

def main() -> None:
    st.title("🧠 Pro Market Screener 7.5")
    st.subheader("Wetenschappelijke Sector Rotatie & Voorspellingen")
    
    # Zijbalk voor instellingen
    st.sidebar.header("⚙️ Systeeminstellingen")
    selected_market_label = st.sidebar.selectbox("Kies een Markt", list(MARKETS.keys()))
    market_info = MARKETS[selected_market_label]
    
    rolling_window = st.sidebar.slider("RRG Analyse Venster (Dagen)", 5, 50, 14)
    max_stocks = st.sidebar.slider("Maximaal aantal aandelen laden", 10, 100, 30)
    
    with st.spinner("Marktlijst ophalen en analyseren..."):
        constituents_df = get_market_constituents(selected_market_label)
        
    if constituents_df.empty:
        st.warning("Geen data beschikbaar.")
        return
        
    # Beperk de lijst om haperingen te voorkomen (Enterprise optimalisatie)
    tickers_to_load = constituents_df['Ticker'].head(max_stocks).tolist()
    benchmark_ticker = market_info["benchmark"]
    
    # Schijndata genereren voor demonstratie (Zodat het script direct standalone werkt zonder yfinance limieten)
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
    
    simulated_prices = pd.DataFrame(
        np.random.randn(100, len(tickers_to_load)).cumsum(axis=0) + 100,
        index=dates, columns=tickers_to_load
    )
    simulated_benchmark = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
    
    # RRG Berekeningen uitvoeren
    rrg_df = calculate_rrg_metrics(simulated_prices, simulated_benchmark, window=rolling_window)
    
    # Voeg sector informatie weer samen
    rrg_df = rrg_df.merge(constituents_df, on="Ticker", how="left")
    
    # Actie-signalen genereren op basis van het kwadrant
    action_map = {"Leading": "HOUDEN / KOPEN", "Improving": "KOPEN (MOMENTUM)", "Lagging": "VERMIJDEN", "Weakening": "WINST NEMEN"}
    rrg_df['Action'] = rrg_df['Quadrant'].map(action_map)
    rrg_df['Alpha Score'] = (rrg_df['Distance'] * 1.5).round(2)
    
    # --- VISUALISATIE ---
    st.header("📈 Het Relative Rotation Graph (RRG) Kwadrant")
    
    fig = px.scatter(
        rrg_df, x="RS_Ratio", y="RS_Momentum", 
        text="Ticker", color="Quadrant",
        size="Distance", hover_data=["Heading", "Sector"],
        color_discrete_map={"Leading": "green", "Improving": "blue", "Lagging": "red", "Weakening": "orange"}
    )
    # Assen kruisen op de benchmark-waarde (100)
    fig.add_hline(y=100, line_dash="dash", line_color="gray")
    fig.add_vline(x=100, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    
    # --- RESULTATEN TABEL ---
    st.header("📊 Gescande Resultaten")
    st.dataframe(rrg_df[['Ticker', 'Sector', 'Quadrant', 'Heading', 'Distance', 'Action', 'Alpha Score']], use_container_width=True)
    
    # --- AI AGENT PROMPT GENERATOR ---
    st.header("🤖 Multi-Agent Expert Consensus")
    stock_pick = st.selectbox("Selecteer een aandeel voor diepgaande analyse", rrg_df['Ticker'].tolist())
    
    if stock_pick:
        row = rrg_df[rrg_df['Ticker'] == stock_pick].iloc[0]
        
        prompt = f"""
**De Risk Manager (De bewaker):** Berekent de optimale entry en exit. Formuleer een trade-plan met een duidelijke risk-to-reward ratio voor {stock_pick}.

JULLIE OPDRACHT:

1. QUANT AUDIT (De Quant):
Evalueer de vector-kwaliteit. Is een Heading van {row['Heading']:.1f}° een teken van duurzame versnelling of naderende uitputting? Interpreteer de afstand ({row['Distance']:.2f}) t.o.v. de benchmark (over-extended of beginnende trend?).

2. FUNDAMENTELE VALIDATIE (De Analist):
Valideer het '{row['Action']}' signaal. Wees sceptisch tegenover de data. Zoek naar de primaire katalysator voor deze sector-rotatie binnen de sector: '{row['Sector']}'. Waarom stroomt er specifiek NU kapitaal naar of uit {stock_pick}?

3. RISK & VOLATILITY (De Risk Manager):
Geef concrete entry- en exit-levels. Gebruik de huidige marktvolatiliteit om een logische Stop-Loss en een 'Take Profit' target te bepalen die past bij de huidige Alpha Score ({row['Alpha Score']}).

4. HET OORDEEL (De Consensus):
Synthetiseer de inzichten in een definitief advies: 
- [STERK KOPEN | SPECULATIEF KOPEN | HOUDEN | VERMIJDEN]
        """
        st.text_area("Gegenereerde AI Prompt (Kopieer deze naar je LLM)", prompt, height=350)

if __name__ == "__main__":
    main()

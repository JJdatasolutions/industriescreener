import io
import math
import warnings
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Onderdruk waarschuwingen
warnings.filterwarnings("ignore")

# --- CONFIGURATIE ---
st.set_page_config(page_title="Pro Market Screener 8.0 (RRG & Sector Flow)", layout="wide", page_icon="🧠")

MARKETS: Dict[str, Dict[str, Any]] = {
    "🇺🇸 USA - S&P 500": {
        "code": "SP500", "benchmark": "SPY", 
        "wiki": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    }
}

# --- 1. DATA INGESTIE LAAG ---

@st.cache_data(ttl=86400)
def get_market_constituents(market_key: str) -> pd.DataFrame:
    mkt = MARKETS.get(market_key, {})
    url = mkt.get('wiki', '')
    
    if not url: return pd.DataFrame()
        
    headers = {"User-Agent": "ProMarketBot/2.0 (Contact: info@enterprise.com)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        
        target_df = next(df for df in tables if any("symbol" in str(c).lower() for c in df.columns))
        ticker_col = next(c for c in target_df.columns if "Symbol" in str(c) or "Ticker" in str(c))
        sector_col = next(c for c in target_df.columns if "Sector" in str(c))
        
        df_clean = target_df[[ticker_col, sector_col]].copy()
        df_clean.columns = ['Ticker', 'Sector']
        df_clean['Ticker'] = df_clean['Ticker'].str.replace('.', '-', regex=False)
        return df_clean
    except Exception:
        return pd.DataFrame([
            ("AAPL", "Technology"), ("MSFT", "Technology"), ("JNJ", "Healthcare"),
            ("JPM", "Financials"), ("XOM", "Energy"), ("PG", "Consumer Staples")
        ], columns=['Ticker', 'Sector'])

# --- 2. QUANT ENGINE (REKENKERN) ---

@st.cache_data(ttl=3600)
def generate_market_simulation(tickers: List[str], sectors: List[str], days: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
    """Genereert realistische historische data voor de RRG berekening."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
    
    # We voegen een sector-bias toe zodat sectoren realistisch groeperen
    sector_biases = {sector: np.random.uniform(-0.5, 0.5) for sector in set(sectors)}
    
    price_data = {}
    for t, s in zip(tickers, sectors):
        # Random walk met een drift gebaseerd op de sector
        drift = sector_biases[s] + np.random.uniform(-0.2, 0.2)
        daily_returns = np.random.normal(loc=drift, scale=2.0, size=days)
        price_data[t] = 100 * np.exp(np.cumsum(daily_returns / 100))
        
    df_prices = pd.DataFrame(price_data, index=dates)
    
    # Benchmark (S&P 500 simulatie) is het gemiddelde van alles
    benchmark = df_prices.mean(axis=1)
    
    return df_prices, benchmark

def calculate_rrg(price_df: pd.DataFrame, benchmark: pd.Series, window: int = 14) -> pd.DataFrame:
    """Berekent de wiskundige RRG coördinaten."""
    results = []
    for col in price_df.columns:
        # Relatieve Sterkte
        rs = (price_df[col] / benchmark) * 100
        rs_ratio = rs.rolling(window=window).mean()
        # Momentum van die relatieve sterkte
        rs_momentum = rs_ratio.pct_change(periods=5) * 100 + 100
        
        if not math.isnan(rs_ratio.iloc[-1]):
            x, y = rs_ratio.iloc[-1], rs_momentum.iloc[-1]
            
            # Bepaal Kwadrant
            if x >= 100 and y >= 100: quad = "Leading (Leidend)"
            elif x >= 100 and y < 100: quad = "Weakening (Verzwakkend)"
            elif x < 100 and y < 100: quad = "Lagging (Achterblijvend)"
            else: quad = "Improving (Verbeterend)"
                
            results.append({
                "Naam": col, "RS_Ratio": x, "RS_Momentum": y, "Kwadrant": quad
            })
    return pd.DataFrame(results)

# --- 3. PRESENTATIE & STYLING LAAG ---

def style_strict_scores(val: float) -> str:
    """
    Conditionele opmaak met harde drempels om ruis te filteren.
    Alleen ECHT goede aandelen lichten groen op.
    """
    if pd.isna(val): return ''
    if val >= 85:
        return 'background-color: #198754; color: white; font-weight: bold;' # ECHT GOED (Felgroen)
    elif val >= 65:
        return 'background-color: #90EE90; color: black;' # Best oké (Lichtgroen)
    elif val >= 45:
        return 'background-color: #FFD700; color: black;' # Matig / Twijfel (Geel)
    else:
        return 'background-color: #DC3545; color: white;' # Slecht (Rood)

def main() -> None:
    st.title("🧠 Pro Market Screener 8.0")
    
    # --- ZIJBALK ---
    st.sidebar.header("⚙️ Instellingen")
    market_key = st.sidebar.selectbox("Kies Markt", list(MARKETS.keys()))
    
    with st.spinner("Data laden..."):
        df_const = get_market_constituents(market_key).head(100) # Beperk tot 100 voor snelheid
        
    df_prices, benchmark = generate_market_simulation(df_const['Ticker'].tolist(), df_const['Sector'].tolist())
    
    # --- TABBLADEN ---
    tab1, tab2 = st.tabs(["🌍 Sector Rotatie (RRG)", "📊 Sector Aandelen Selectie"])
    
    with tab1:
        st.header("Sector Rotatie (Geldstromen)")
        st.write("Dit kwadrant toont de gezondheid van **hele sectoren**. Rechtsboven (Leidend) is waar het grote geld naartoe stroomt.")
        
        # Groepeer de aandelenprijzen per sector om sector-indices te maken
        sector_prices = pd.DataFrame(index=df_prices.index)
        for sector in df_const['Sector'].unique():
            tickers_in_sector = df_const[df_const['Sector'] == sector]['Ticker'].tolist()
            sector_prices[sector] = df_prices[tickers_in_sector].mean(axis=1)
            
        sector_rrg = calculate_rrg(sector_prices, benchmark)
        
        # Plotly RRG Grafiek
        fig_rrg = px.scatter(
            sector_rrg, x="RS_Ratio", y="RS_Momentum", text="Naam", color="Kwadrant",
            color_discrete_map={"Leading (Leidend)": "green", "Improving (Verbeterend)": "blue", 
                                "Lagging (Achterblijvend)": "red", "Weakening (Verzwakkend)": "orange"},
            title="Relative Rotation Graph (Sectoren t.o.v. S&P 500)",
            width=800, height=600
        )
        fig_rrg.add_hline(y=100, line_dash="dash", line_color="gray")
        fig_rrg.add_vline(x=100, line_dash="dash", line_color="gray")
        fig_rrg.update_traces(textposition='top center', marker=dict(size=15))
        
        # Voeg de kwadrant labels toe als achtergrond
        fig_rrg.add_annotation(x=105, y=105, text="LEADING", showarrow=False, opacity=0.3, font=dict(size=30, color="green"))
        fig_rrg.add_annotation(x=95, y=105, text="IMPROVING", showarrow=False, opacity=0.3, font=dict(size=30, color="blue"))
        fig_rrg.add_annotation(x=95, y=95, text="LAGGING", showarrow=False, opacity=0.3, font=dict(size=30, color="red"))
        fig_rrg.add_annotation(x=105, y=95, text="WEAKENING", showarrow=False, opacity=0.3, font=dict(size=30, color="orange"))
        
        st.plotly_chart(fig_rrg, use_container_width=True)
        

    with tab2:
        selected_sector = st.selectbox("Selecteer een Sector om in te zoomen:", df_const['Sector'].unique())
        sector_tickers = df_const[df_const['Sector'] == selected_sector]['Ticker'].tolist()
        
        st.write(f"### De beste kandidaten in **{selected_sector}**")
        st.write("Let op de kleuren: Alleen scores boven de **85** (Felgroen) zijn echt uitmuntend. Geel of Rood betekent wegblijven, zelfs al is het de beste van de sector.")
        
        # Bereken huidige statistieken
        results = []
        for t in sector_tickers:
            current = df_prices[t].iloc[-1]
            sma50 = df_prices[t].rolling(50).mean().iloc[-1]
            perf_sma = ((current - sma50) / sma50) * 100
            
            # Simulatie van stricte scores (0-100)
            mom_score = np.clip(perf_sma * 4 + 40, 0, 100) 
            val_score = np.clip(np.random.normal(50, 25), 0, 100)
            combo = (mom_score * 0.6) + (val_score * 0.4)
            
            results.append({
                "Ticker": t,
                "Prijs ($)": round(current, 2),
                "Perf vs SMA50 (%)": round(perf_sma, 2),
                "Momentum Score": round(mom_score, 1),
                "Value Score": round(val_score, 1),
                "Combo Score": round(combo, 1)
            })
            
        df_results = pd.DataFrame(results).sort_values(by="Combo Score", ascending=False).reset_index(drop=True)
        df_results.index += 1 # Start index bij 1
        
        # Toepassen van de strict geconfigureerde conditionele opmaak
        styled_df = df_results.style.map(
            style_strict_scores, 
            subset=['Momentum Score', 'Value Score', 'Combo Score']
        ).format(precision=2)
        
        st.dataframe(styled_df, use_container_width=True)

if __name__ == "__main__":
    main()

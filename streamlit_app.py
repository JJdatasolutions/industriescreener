import warnings
import math
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st

warnings.filterwarnings("ignore")

# --- CONFIGURATIE ---
st.set_page_config(page_title="Hedge Fund Screener 9.0 (Quant Edition)", layout="wide", page_icon="📈")

# --- 1. DATA LAAG (LAZY LOADING, CACHING & GRACEFUL DEGRADATION) ---

@st.cache_data(ttl=3600)
def fetch_market_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Haalt historische koersen op met robuuste foutafhandeling en simulatie-terugval."""
    try:
        # Download de data in bulk om de API niet te overbelasten
        data = yf.download(tickers, period=period, progress=False)
        
        # Controleer of de dataset leeg is (Voorkomt de beruchte 'Adj Close' fout)
        if data.empty:
            raise ValueError("Yahoo Finance weigerde de verbinding of gaf een lege dataset terug.")
        
        # Dynamisch de juiste prijskolom bepalen (ondersteunt zowel oude als nieuwe yfinance versies)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.levels[0]:
                price_data = data['Adj Close']
            elif 'Close' in data.columns.levels[0]:
                price_data = data['Close']
            else:
                price_data = data
        else:
            if 'Adj Close' in data:
                price_data = data['Adj Close']
            elif 'Close' in data:
                price_data = data['Close']
            else:
                price_data = data

        # Forceer naar een correct tabel-formaat als er maar 1 aandeel overblijft
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame(name=tickers[0])
            
        return price_data.ffill().dropna(axis=1, how='all')
        
    except Exception as e:
        # VANGNET: Als Yahoo faalt, valt het systeem wiskundig terug op Monte Carlo simulaties.
        st.warning(f"⚠️ Live datastroom tijdelijk onderbroken ({e}). Systeem draait op Monte Carlo simulatie.")
        
        np.random.seed(42)
        days = 252 # Aantal handelsdagen in 1 jaar
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='B')
        
        simulated_data = {}
        for t in tickers:
            # Wiskundige Random Walk simulatie
            simulated_data[t] = 100 * np.exp(np.cumsum(np.random.normal(0, 0.015, days)))
            
        return pd.DataFrame(simulated_data, index=dates)

@st.cache_data(ttl=86400)
def fetch_fundamentals(ticker: str) -> Dict[str, float]:
    """
    Live Fundamental Ingestion.
    Wordt 'lazy' (pas op het laatste moment) geladen om IP-blokkades te voorkomen.
    """
    try:
        info = yf.Ticker(ticker).info
        gp = info.get('grossProfits', np.nan)
        ta = info.get('totalAssets', np.nan)
        pb = info.get('priceToBook', np.nan)
        
        # Gross Profitability Premium (GP/A)
        gpa = gp / ta if ta and gp else np.nan
        
        return {"GP_A": gpa, "P_B": pb}
    except Exception:
        return {"GP_A": np.nan, "P_B": np.nan}

# --- 2. QUANT MATHEMATICS LAAG (REKENKERN) ---

def calc_rrg(price_df: pd.DataFrame, benchmark: pd.Series, window: int = 14) -> pd.DataFrame:
    """Berekent wiskundige Relative Rotation Graph coördinaten."""
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
    """
    Faber's Turnover Reductie Hysterese. 
    Helpt om transactiekosten (Technical Debt in trading) te minimaliseren.
    """
    status = []
    for ticker in current_ranks.index:
        curr = current_ranks.get(ticker, 999)
        prev = prev_ranks.get(ticker, 999)
        
        if curr <= 3:
            status.append("KOPEN (Top 3)")
        elif 3 < curr <= 5 and prev <= 3:
            status.append("HOUDEN (Gedegradeerd, maar in Top 5)")
        elif curr <= 5:
            status.append("HOUDEN (In observatie)")
        else:
            status.append("VERKOPEN (Uit Top 5)")
    return pd.Series(status, index=current_ranks.index)

# --- 3. PRESENTATIE LAAG (UI & DASHBOARDS) ---

def main() -> None:
    st.title("📈 Hedge Fund Quant Terminal 9.0")
    st.markdown("Integratie van Novy-Marx Profitability, Phase Space Attractors en Dorsey Wright.")
    
    # Een stabiel, vast universum voor de demonstratie
    sector_map = {
        "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", 
        "JPM": "Fin", "BAC": "Fin", "GS": "Fin",
        "JNJ": "Health", "PFE": "Health", "UNH": "Health",
        "XOM": "Energy", "CVX": "Energy", "SPY": "Benchmark"
    }
    tickers = list(sector_map.keys())
    
    with st.spinner("Kwantitatieve modellen initialiseren..."):
        df_prices = fetch_market_data(tickers, period="1y")
        
    if df_prices.empty:
        st.stop()
        
    benchmark = df_prices['SPY']
    universe = df_prices.drop(columns=['SPY'])
    
    # De 4 Professionele Tabbladen
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Macro & Correlatie", "🔥 Dorsey Wright RS Matrix", 
        "🔬 Deep Dive & Phase Space", "🤖 AI Analist"
    ])
    
    with tab1:
        st.header("Sector Rotatie & Correlatiedispersie")
        col1, col2 = st.columns(2)
        
        with col1:
            rrg_df = calc_rrg(universe, benchmark)
            rrg_df['Sector'] = rrg_df['Ticker'].map(sector_map)
            
            fig_rrg = px.scatter(
                rrg_df, x="RS_Ratio", y="RS_Momentum", text="Ticker", color="Sector",
                title="Relative Rotation Graph"
            )
            fig_rrg.add_hline(y=100, line_dash="dash"); fig_rrg.add_vline(x=100, line_dash="dash")
            st.plotly_chart(fig_rrg, use_container_width=True)
            
            
        with col2:
            returns = universe.pct_change().dropna()
            corr_matrix = returns.corr()
            
            fig_corr = px.imshow(
                corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                title="Correlatie Heatmap (Dispersie Check)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("In mensentaal: Lage correlatie (blauw/wit) betekent dat aandelen hun eigen weg gaan. Dit is noodzakelijk voor succesvolle sectorrotatie.")

    with tab2:
        st.header("Dorsey Wright Relative Strength Matrix")
        st.write("Head-to-head vergelijking (welk aandeel is sterker dan het andere?).")
        
        perf_6m = (universe.iloc[-1] / universe.iloc[-126]) - 1
        matrix = pd.DataFrame(index=perf_6m.index, columns=perf_6m.index)
        
        for t1 in matrix.index:
            for t2 in matrix.columns:
                if t1 == t2: matrix.loc[t1, t2] = np.nan
                else: matrix.loc[t1, t2] = 1 if perf_6m[t1] > perf_6m[t2] else 0
                
        matrix['Wins'] = matrix.sum(axis=1)
        matrix = matrix.sort_values(by='Wins', ascending=False)
        
        perf_7m_to_1m = (universe.iloc[-21] / universe.iloc[-147]) - 1
        prev_rank = perf_7m_to_1m.rank(ascending=False)
        curr_rank = perf_6m.rank(ascending=False)
        
        portfolio_status = apply_faber_logic(curr_rank, prev_rank)
        
        display_df = pd.DataFrame({
            "6 Maands Rendement": (perf_6m * 100).round(2).astype(str) + "%",
            "Matrix Overwinningen": matrix['Wins'],
            "Portfolio Status (Faber)": portfolio_status
        }).sort_values(by="Matrix Overwinningen", ascending=False)
        
        st.dataframe(display_df, use_container_width=True)

    with tab3:
        st.header("Diepte-Analyse & Wiskundige Attractors")
        selected_stock = st.selectbox("Selecteer aandeel voor Deep Dive:", universe.columns)
        
        st.subheader("1. Fama-French & Novy-Marx Fundamentals (Live)")
        with st.spinner("Live bedrijfsdata ophalen via API..."):
            funds = fetch_fundamentals(selected_stock)
            
        m1, m2 = st.columns(2)
        m1.metric("Gross Profitability (GP/A)", f"{funds['GP_A']:.4f}" if not pd.isna(funds['GP_A']) else "Niet Beschikbaar", 
                  help="Hoge GP/A wijst op kwaliteitsbedrijven (Novy-Marx).")
        m2.metric("Price-to-Book (P/B)", f"{funds['P_B']:.2f}" if not pd.isna(funds['P_B']) else "Niet Beschikbaar", 
                  help="Waardering ten opzichte van boekwaarde (Fama-French).")
        
        st.subheader("2. Phase Space Attractor (Chaostheorie)")
        tau = st.slider("Time Delay (Dagen in het verleden)", 1, 20, 5)
        
        daily_ret = returns[selected_stock]
        delayed_ret = daily_ret.shift(tau)
        
        phase_df = pd.DataFrame({'r_t': daily_ret, 'r_t_tau': delayed_ret}).dropna()
        
        fig_phase = px.scatter(
            phase_df, x='r_t_tau', y='r_t', opacity=0.5,
            title=f"Phase Space Plot: Rendement Vandaag vs Rendement {-tau} Dagen Geleden",
            labels={'r_t_tau': f'Rendement T-{tau}', 'r_t': 'Rendement Vandaag'}
        )
        st.plotly_chart(fig_phase, use_container_width=True)
        
        st.caption("In mensentaal: Concentreert de puntenwolk zich in een schuine sigaar-vorm? Dan zit er een voorspelbare trend in het aandeel. Is het een perfecte, willekeurige bolkraam? Dan heerst er pure chaos en kun je beter wegblijven.")

    with tab4:
        st.header("AI Analyst 2.0 (Deep Research)")
        if st.button("Genereer Quant Prompt"):
            prompt = f"""
**De Quant Analyst Agent:** Evalueer de fundamentele en wiskundige status van {selected_stock}.

JULLIE OPDRACHT:

1. FUNDAMENTELE AUDIT (Novy-Marx & Value):
De huidige Gross Profitability (GP/A) is {funds['GP_A']:.4f} en de P/B ratio is {funds['P_B']:.2f}. Beoordeel dit profiel ten opzichte van sector-gemiddelden. Is dit een 'value trap' of een high-quality aandeel?

2. KINETICA & CHAOS (Huffaker):
De Phase Space Attractor toont de correlatie tussen het rendement van vandaag en {tau} dagen geleden. Analyseer of de huidige marktfase van dit aandeel een stabiele trend vertoont of dat het chaotisch gedrag is.

3. PORTFOLIO MANAGEMENT (Faber & Dorsey Wright):
Dit aandeel heeft de status: '{portfolio_status[selected_stock]}'. De actuele correlatie met de brede markt (SPY) is {corr_matrix.loc[selected_stock, 'SPY']:.2f}. 
Bepaal de optimale positiegrootte. Als de correlatie extreem hoog is en de sector dispersie laag, adviseer dan om de positie te verkleinen wegens gebrek aan diversificatie.
            """
            st.text_area("Kopieer deze data-gedreven prompt naar je favoriete LLM:", prompt, height=350)

if __name__ == "__main__":
    main()

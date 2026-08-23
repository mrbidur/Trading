"""
QQQ/TQQQ Dynamic Asset-Switching Strategy — Streamlit Web App
==============================================================
Interactive backtesting UI with adjustable parameters, charts, and CSV export.

Run: streamlit run streamlit_switching_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime
from typing import List
import io

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="QQQ/TQQQ Switching Strategy Backtest",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #22c55e; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }
    .stApp { background-color: #0f172a; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA FETCHING (cached)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(base_ticker: str, lev_ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch and align daily close prices for both assets."""
    base = yf.Ticker(base_ticker)
    lev = yf.Ticker(lev_ticker)

    base_df = base.history(start=start, end=end, auto_adjust=True)
    lev_df = lev.history(start=start, end=end, auto_adjust=True)

    if base_df.empty or lev_df.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns
    if isinstance(base_df.columns, pd.MultiIndex):
        base_df.columns = base_df.columns.get_level_values(0)
    if isinstance(lev_df.columns, pd.MultiIndex):
        lev_df.columns = lev_df.columns.get_level_values(0)

    # Timezone-naive
    if base_df.index.tz is not None:
        base_df.index = base_df.index.tz_localize(None)
    if lev_df.index.tz is not None:
        lev_df.index = lev_df.index.tz_localize(None)

    combined = pd.DataFrame({
        "base_close": base_df["Close"],
        "lev_close": lev_df["Close"],
    }).dropna()

    return combined


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def run_backtest(
    data: pd.DataFrame,
    ema_period: int,
    initial_capital: float,
    drawdown_tiers: List[float],
    transfer_pct: float,
    reset_threshold: float,
    signal_asset: str,  # "base" or "leveraged"
) -> pd.DataFrame:
    """
    Run the multi-tier switching backtest.

    Signal: EMA distance calculated on the chosen signal asset (TQQQ or QQQ).
    Action: Transfer portions from QQQ into TQQQ when tiers are breached.
    Reset: When signal asset recovers above EMA.
    """
    # Compute EMA on the signal asset
    signal_col = "lev_close" if signal_asset == "leveraged" else "base_close"
    data = data.copy()
    data["ema"] = data[signal_col].ewm(span=ema_period, adjust=False).mean()
    data["ema_dist_pct"] = ((data[signal_col] - data["ema"]) / data["ema"]) * 100.0
    data = data.dropna()

    if data.empty:
        return pd.DataFrame()

    # Initial allocation: all in QQQ
    first_base_price = float(data["base_close"].iloc[0])
    qqq_shares = initial_capital / first_base_price
    tqqq_shares = 0.0

    n_tiers = len(drawdown_tiers)
    tiers_triggered = [False] * n_tiers

    rows = []
    for dt, row in data.iterrows():
        bp = float(row["base_close"])
        lp = float(row["lev_close"])
        ema_val = float(row["ema"])
        dist = float(row["ema_dist_pct"])

        action = "HOLD"
        transfer = 0.0

        # Reset when signal asset above EMA
        if dist > reset_threshold:
            if any(tiers_triggered):
                action = "RESET"
            tiers_triggered = [False] * n_tiers
        else:
            # Check tiers
            for t_idx, thresh in enumerate(drawdown_tiers):
                if dist <= thresh and not tiers_triggered[t_idx] and qqq_shares > 0:
                    tiers_triggered[t_idx] = True
                    sell_shares = qqq_shares * (transfer_pct / 100.0)
                    cash = sell_shares * bp
                    tqqq_shares += cash / lp
                    qqq_shares -= sell_shares
                    transfer = cash
                    action = f"TIER {t_idx + 1} ({thresh:.0f}%)"
                    break

        qqq_val = qqq_shares * bp
        tqqq_val = tqqq_shares * lp
        total = qqq_val + tqqq_val
        benchmark = (initial_capital / first_base_price) * bp

        rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "QQQ Price": round(bp, 4),
            "TQQQ Price": round(lp, 4),
            "EMA": round(ema_val, 4),
            "EMA Dist %": round(dist, 2),
            "QQQ Shares": round(qqq_shares, 4),
            "TQQQ Shares": round(tqqq_shares, 4),
            "QQQ Value": round(qqq_val, 2),
            "TQQQ Value": round(tqqq_val, 2),
            "Portfolio Value": round(total, 2),
            "Benchmark (QQQ)": round(benchmark, 2),
            "Tiers Active": sum(tiers_triggered),
            "Action": action,
            "Transfer ($)": round(transfer, 2),
        })

    return pd.DataFrame(rows)


# =============================================================================
# METRICS CALCULATOR
# =============================================================================
def compute_metrics(results: pd.DataFrame, initial_capital: float) -> dict:
    """Compute performance metrics."""
    vals = results["Portfolio Value"].values
    bench = results["Benchmark (QQQ)"].values

    years = max(
        (pd.to_datetime(results["Date"].iloc[-1]) - pd.to_datetime(results["Date"].iloc[0])).days / 365.25,
        0.01,
    )

    # Strategy metrics
    final_strat = vals[-1]
    cagr_strat = ((final_strat / initial_capital) ** (1 / years) - 1) * 100
    warmup = min(252, len(vals) // 4)
    mdd_vals = vals[warmup:]
    peak = np.maximum.accumulate(mdd_vals)
    mdd_strat = float(((mdd_vals - peak) / peak).min()) * 100 if len(mdd_vals) > 0 else 0

    # Benchmark metrics
    final_bench = bench[-1]
    cagr_bench = ((final_bench / initial_capital) ** (1 / years) - 1) * 100
    mdd_bench_vals = bench[warmup:]
    peak_b = np.maximum.accumulate(mdd_bench_vals)
    mdd_bench = float(((mdd_bench_vals - peak_b) / peak_b).min()) * 100 if len(mdd_bench_vals) > 0 else 0

    switches = results[results["Action"].str.contains("TIER")].shape[0]
    resets = results[results["Action"] == "RESET"].shape[0]

    return {
        "final_strategy": final_strat,
        "final_benchmark": final_bench,
        "return_strategy": (final_strat / initial_capital - 1) * 100,
        "return_benchmark": (final_bench / initial_capital - 1) * 100,
        "cagr_strategy": cagr_strat,
        "cagr_benchmark": cagr_bench,
        "mdd_strategy": mdd_strat,
        "mdd_benchmark": mdd_bench,
        "years": years,
        "switches": switches,
        "resets": resets,
    }


# =============================================================================
# SIDEBAR — INTERACTIVE CONTROLS
# =============================================================================
st.sidebar.title("⚙️ Strategy Parameters")

st.sidebar.markdown("### Assets")
base_asset = st.sidebar.text_input("Base Asset (hold)", value="QQQ")
lev_asset = st.sidebar.text_input("Leveraged Asset (switch into)", value="TQQQ")

st.sidebar.markdown("### Timeline")
start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 2, 11))
end_date = st.sidebar.date_input("End Date", value=date.today())

st.sidebar.markdown("### Capital")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=10000, min_value=1000)

st.sidebar.markdown("### EMA Settings")
ema_period = st.sidebar.slider("EMA Period (days)", min_value=10, max_value=500, value=200, step=10)
signal_asset = st.sidebar.radio("EMA Signal Calculated On", ["TQQQ (leveraged)", "QQQ (base)"], index=0)

st.sidebar.markdown("### Drawdown Tiers")
st.sidebar.caption("% below EMA that triggers each transfer")
tier1 = st.sidebar.slider("Tier 1 (%)", min_value=-80, max_value=0, value=-20, step=5)
tier2 = st.sidebar.slider("Tier 2 (%)", min_value=-80, max_value=0, value=-30, step=5)
tier3 = st.sidebar.slider("Tier 3 (%)", min_value=-80, max_value=0, value=-40, step=5)
tier4 = st.sidebar.slider("Tier 4 (%)", min_value=-80, max_value=0, value=-50, step=5)

st.sidebar.markdown("### Transfer Settings")
transfer_pct = st.sidebar.slider("Transfer % per Tier", min_value=5, max_value=100, value=25, step=5)
reset_threshold = st.sidebar.slider("Reset Threshold (% above EMA)", min_value=-10, max_value=20, value=0, step=1)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")


# =============================================================================
# MAIN AREA
# =============================================================================
st.title("📊 QQQ/TQQQ Dynamic Switching Strategy")
st.caption(
    f"Multi-tier drawdown switching from **{base_asset}** to **{lev_asset}** based on "
    f"{ema_period}-day EMA distance. Tiers: {tier1}%, {tier2}%, {tier3}%, {tier4}% | "
    f"Transfer: {transfer_pct}% per tier"
)

if run_button:
    drawdown_tiers = sorted([tier1, tier2, tier3, tier4])
    signal_choice = "leveraged" if "TQQQ" in signal_asset else "base"

    with st.spinner("Fetching market data..."):
        data = fetch_data(
            base_asset, lev_asset,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

    if data.empty:
        st.error("❌ No data found. Check your ticker symbols and date range.")
    else:
        with st.spinner("Running backtest simulation..."):
            results = run_backtest(
                data=data,
                ema_period=ema_period,
                initial_capital=initial_capital,
                drawdown_tiers=drawdown_tiers,
                transfer_pct=transfer_pct,
                reset_threshold=reset_threshold,
                signal_asset=signal_choice,
            )

        if results.empty:
            st.error("❌ Backtest produced no results. EMA period may be too long for the data range.")
        else:
            metrics = compute_metrics(results, initial_capital)

            # ----- METRICS PANEL -----
            st.markdown("---")
            st.subheader("Performance Summary")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Strategy Final", f"${metrics['final_strategy']:,.0f}", f"{metrics['return_strategy']:.1f}%")
            col2.metric("Benchmark Final", f"${metrics['final_benchmark']:,.0f}", f"{metrics['return_benchmark']:.1f}%")
            col3.metric("Strategy CAGR", f"{metrics['cagr_strategy']:.2f}%")
            col4.metric("Strategy MDD", f"{metrics['mdd_strategy']:.2f}%")
            col5.metric("Benchmark MDD", f"{metrics['mdd_benchmark']:.2f}%")

            col6, col7, col8 = st.columns(3)
            col6.metric("Duration", f"{metrics['years']:.1f} years")
            col7.metric("Tier Switches", f"{metrics['switches']}")
            col8.metric("Reset Events", f"{metrics['resets']}")

            # ----- PORTFOLIO GROWTH CHART -----
            st.markdown("---")
            st.subheader("Portfolio Growth: Strategy vs Benchmark")

            chart_df = results[["Date", "Portfolio Value", "Benchmark (QQQ)"]].copy()
            chart_df["Date"] = pd.to_datetime(chart_df["Date"])
            chart_df = chart_df.set_index("Date")
            st.line_chart(chart_df, use_container_width=True)

            # ----- EMA DISTANCE CHART -----
            st.subheader("EMA Distance % (Trigger Zones)")

            ema_chart = results[["Date", "EMA Dist %"]].copy()
            ema_chart["Date"] = pd.to_datetime(ema_chart["Date"])
            ema_chart = ema_chart.set_index("Date")
            st.area_chart(ema_chart, use_container_width=True)

            # ----- ALLOCATION CHART -----
            st.subheader("Asset Allocation Over Time")

            alloc_chart = results[["Date", "QQQ Value", "TQQQ Value"]].copy()
            alloc_chart["Date"] = pd.to_datetime(alloc_chart["Date"])
            alloc_chart = alloc_chart.set_index("Date")
            st.area_chart(alloc_chart, use_container_width=True)

            # ----- EVENT LOG -----
            st.subheader("Switching Events")
            events = results[results["Action"] != "HOLD"][["Date", "Action", "EMA Dist %", "Transfer ($)", "Portfolio Value", "Tiers Active"]]
            st.dataframe(events, use_container_width=True, height=300)

            # ----- CSV DOWNLOAD -----
            st.markdown("---")
            csv_buffer = io.StringIO()
            results.to_csv(csv_buffer, index=False)

            st.download_button(
                label="📥 Download Full Results CSV",
                data=csv_buffer.getvalue(),
                file_name="strategy_backtest_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # ----- RAW DATA (expandable) -----
            with st.expander("📋 View Raw Data (first 100 rows)"):
                st.dataframe(results.head(100), use_container_width=True)

else:
    st.info("👈 Configure your strategy parameters in the sidebar and click **Run Backtest** to see results.")
    st.markdown("""
    ### How This Strategy Works

    1. **Start**: $100K invested 100% in QQQ
    2. **Signal**: Calculate the % distance of TQQQ from its 200-day EMA
    3. **Tiers**: When the distance crosses below each threshold (e.g., -20%, -30%, -40%, -50%), 
       transfer a chunk (e.g., 25%) of your QQQ holdings into TQQQ
    4. **Reset**: When the signal asset recovers above the EMA, all tier flags reset for the next cycle
    5. **Result**: During drawdowns you accumulate discounted TQQQ; during recoveries you benefit from 3x leverage

    ### Quick Start
    - Use the **default settings** for the original strategy
    - Click **Run Backtest** to see 15+ years of results
    - Adjust the sliders to test different tier levels and transfer amounts
    """)

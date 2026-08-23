"""
QQQ/TQQQ Dynamic DCA + Switching Strategy — Streamlit Web App
===============================================================
Weekly DCA into QQQ with multi-tier switching into TQQQ based on
TQQQ's distance below its 200-day EMA.

NO lump-sum. Weekly $200 DCA accumulates QQQ continuously.
When TQQQ drops below EMA by configured tiers, transfer portions
of QQQ holdings into TQQQ. Reset when TQQQ recovers above EMA.

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
    page_title="QQQ/TQQQ DCA + Switching Backtest",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
        "qqq_close": base_df["Close"],
        "tqqq_close": lev_df["Close"],
    }).dropna()

    return combined


# =============================================================================
# BACKTEST ENGINE — Weekly DCA + Multi-Tier Switching
# =============================================================================
def run_backtest(
    data: pd.DataFrame,
    ema_period: int,
    weekly_dca: float,
    drawdown_tiers: List[float],
    transfer_pct: float,
    reset_threshold: float,
) -> pd.DataFrame:
    """
    Weekly DCA + Multi-Tier Switching Strategy.

    Rules:
    1. Every Monday (or first trading day of the week): inject $weekly_dca
       and buy QQQ shares at that day's close price.
    2. Calculate TQQQ's 200-day EMA and % distance from it daily.
    3. When TQQQ's distance crosses below each tier threshold, transfer
       transfer_pct% of current QQQ holdings into TQQQ.
    4. When TQQQ recovers above EMA (dist > reset_threshold), reset all
       tier flags for the next drawdown cycle.
    5. Track daily: shares held, values, portfolio total vs benchmark.
    """
    data = data.copy()

    # Compute EMA on TQQQ (the signal asset)
    data["tqqq_ema"] = data["tqqq_close"].ewm(span=ema_period, adjust=False).mean()
    data["ema_dist_pct"] = ((data["tqqq_close"] - data["tqqq_ema"]) / data["tqqq_ema"]) * 100.0
    data = data.dropna()

    if data.empty:
        return pd.DataFrame()

    # State
    qqq_shares = 0.0
    tqqq_shares = 0.0
    total_invested = 0.0
    n_tiers = len(drawdown_tiers)
    tiers_triggered = [False] * n_tiers
    last_dca_week = None

    # Benchmark: pure QQQ DCA (same weekly amount, no switching)
    bench_qqq_shares = 0.0
    bench_invested = 0.0

    rows = []

    for dt, row in data.iterrows():
        qqq_price = float(row["qqq_close"])
        tqqq_price = float(row["tqqq_close"])
        tqqq_ema = float(row["tqqq_ema"])
        dist_pct = float(row["ema_dist_pct"])

        current_dt = dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else dt
        current_week = (current_dt.isocalendar()[0], current_dt.isocalendar()[1])
        is_dca_day = (current_week != last_dca_week)

        action = "HOLD"
        deposit = 0.0
        transfer_amount = 0.0

        # ----------------------------------------------------------
        # WEEKLY DCA: Buy QQQ every new week
        # ----------------------------------------------------------
        if is_dca_day:
            last_dca_week = current_week
            deposit = weekly_dca
            shares_bought = deposit / qqq_price
            qqq_shares += shares_bought
            total_invested += deposit

            # Benchmark also buys QQQ
            bench_qqq_shares += deposit / qqq_price
            bench_invested += deposit

            if action == "HOLD":
                action = "DCA"

        # ----------------------------------------------------------
        # RESET: When TQQQ recovers above EMA
        # ----------------------------------------------------------
        if dist_pct > reset_threshold:
            if any(tiers_triggered):
                action = "RESET"
            tiers_triggered = [False] * n_tiers

        # ----------------------------------------------------------
        # TIER SWITCHING: Transfer QQQ → TQQQ at each tier
        # ----------------------------------------------------------
        else:
            for t_idx, thresh in enumerate(drawdown_tiers):
                if dist_pct <= thresh and not tiers_triggered[t_idx] and qqq_shares > 0:
                    tiers_triggered[t_idx] = True

                    # Transfer transfer_pct% of current QQQ holdings to TQQQ
                    shares_to_sell = qqq_shares * (transfer_pct / 100.0)
                    cash_from_sale = shares_to_sell * qqq_price
                    tqqq_bought = cash_from_sale / tqqq_price

                    qqq_shares -= shares_to_sell
                    tqqq_shares += tqqq_bought
                    transfer_amount = cash_from_sale

                    action = f"TIER {t_idx + 1} ({thresh:.0f}%)"
                    break  # Only one tier per day

        # ----------------------------------------------------------
        # PORTFOLIO VALUATION
        # ----------------------------------------------------------
        qqq_value = qqq_shares * qqq_price
        tqqq_value = tqqq_shares * tqqq_price
        total_portfolio = qqq_value + tqqq_value
        benchmark_value = bench_qqq_shares * qqq_price

        rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "QQQ Price": round(qqq_price, 4),
            "TQQQ Price": round(tqqq_price, 4),
            "TQQQ EMA": round(tqqq_ema, 4),
            "EMA Dist %": round(dist_pct, 2),
            "QQQ Shares": round(qqq_shares, 4),
            "TQQQ Shares": round(tqqq_shares, 4),
            "QQQ Value ($)": round(qqq_value, 2),
            "TQQQ Value ($)": round(tqqq_value, 2),
            "Portfolio Value ($)": round(total_portfolio, 2),
            "Benchmark (QQQ DCA)": round(benchmark_value, 2),
            "Total Invested ($)": round(total_invested, 2),
            "Tiers Active": sum(tiers_triggered),
            "Action": action,
            "Deposit ($)": round(deposit, 2),
            "Transfer ($)": round(transfer_amount, 2),
        })

    return pd.DataFrame(rows)


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(results: pd.DataFrame) -> dict:
    """Compute performance metrics."""
    if results.empty:
        return {}

    vals = results["Portfolio Value ($)"].values
    bench = results["Benchmark (QQQ DCA)"].values
    invested = results["Total Invested ($)"].values

    total_invested = invested[-1]
    final_strat = vals[-1]
    final_bench = bench[-1]

    years = max(
        (pd.to_datetime(results["Date"].iloc[-1]) - pd.to_datetime(results["Date"].iloc[0])).days / 365.25,
        0.01,
    )

    # Return on invested capital
    roi_strat = (final_strat / total_invested - 1) * 100
    roi_bench = (final_bench / total_invested - 1) * 100

    # MDD (skip first year warmup)
    warmup = min(252, len(vals) // 4)
    mdd_vals = vals[warmup:]
    if len(mdd_vals) > 0:
        peak = np.maximum.accumulate(mdd_vals)
        mdd_strat = float(((mdd_vals - peak) / peak).min()) * 100
    else:
        mdd_strat = 0.0

    mdd_bench_vals = bench[warmup:]
    if len(mdd_bench_vals) > 0:
        peak_b = np.maximum.accumulate(mdd_bench_vals)
        mdd_bench = float(((mdd_bench_vals - peak_b) / peak_b).min()) * 100
    else:
        mdd_bench = 0.0

    switches = results[results["Action"].str.contains("TIER")].shape[0]
    resets = results[results["Action"] == "RESET"].shape[0]
    dca_count = results[results["Action"] == "DCA"].shape[0]

    return {
        "total_invested": total_invested,
        "final_strategy": final_strat,
        "final_benchmark": final_bench,
        "roi_strategy": roi_strat,
        "roi_benchmark": roi_bench,
        "mdd_strategy": mdd_strat,
        "mdd_benchmark": mdd_bench,
        "years": years,
        "switches": switches,
        "resets": resets,
        "dca_weeks": dca_count,
        "alpha": final_strat - final_bench,
    }


# =============================================================================
# SIDEBAR — INTERACTIVE CONTROLS
# =============================================================================
st.sidebar.title("⚙️ Strategy Parameters")

st.sidebar.markdown("### 💰 Weekly DCA")
weekly_dca = st.sidebar.slider(
    "Weekly Deposit ($)", min_value=50, max_value=2000, value=200, step=50,
    help="Amount invested into QQQ every week"
)

st.sidebar.markdown("### 📅 Timeline")
start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 2, 11))
end_date = st.sidebar.date_input("End Date", value=date.today())

st.sidebar.markdown("### 📈 EMA Settings")
ema_period = st.sidebar.slider(
    "EMA Period (days)", min_value=10, max_value=500, value=200, step=10,
    help="200-day EMA calculated on TQQQ price"
)

st.sidebar.markdown("### 📉 Drawdown Tiers")
st.sidebar.caption("TQQQ % below its 200 EMA that triggers each switch")
tier1 = st.sidebar.slider("Tier 1 (%)", min_value=-80, max_value=0, value=-20, step=5)
tier2 = st.sidebar.slider("Tier 2 (%)", min_value=-80, max_value=0, value=-30, step=5)
tier3 = st.sidebar.slider("Tier 3 (%)", min_value=-80, max_value=0, value=-40, step=5)
tier4 = st.sidebar.slider("Tier 4 (%)", min_value=-80, max_value=0, value=-50, step=5)

st.sidebar.markdown("### 🔄 Transfer Settings")
transfer_pct = st.sidebar.slider(
    "Transfer % of QQQ per Tier", min_value=5, max_value=100, value=25, step=5,
    help="% of current QQQ holdings moved to TQQQ at each tier"
)
reset_threshold = st.sidebar.slider(
    "Reset Threshold (%)", min_value=-10, max_value=20, value=0, step=1,
    help="TQQQ EMA distance above which tier flags reset"
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")


# =============================================================================
# MAIN AREA
# =============================================================================
st.title("📊 QQQ/TQQQ Weekly DCA + Switching Strategy")
st.caption(
    f"Weekly ${weekly_dca} DCA into QQQ | "
    f"Switch to TQQQ when TQQQ drops {tier1}%/{tier2}%/{tier3}%/{tier4}% below {ema_period}-day EMA | "
    f"Transfer {transfer_pct}% of QQQ per tier"
)

if run_button:
    drawdown_tiers = sorted([tier1, tier2, tier3, tier4])

    with st.spinner("📡 Fetching market data from Yahoo Finance..."):
        data = fetch_data(
            "QQQ", "TQQQ",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

    if data.empty:
        st.error("❌ No data found. Check your date range (TQQQ starts Feb 2010).")
    else:
        with st.spinner("⚙️ Running backtest simulation..."):
            results = run_backtest(
                data=data,
                ema_period=ema_period,
                weekly_dca=weekly_dca,
                drawdown_tiers=drawdown_tiers,
                transfer_pct=transfer_pct,
                reset_threshold=reset_threshold,
            )

        if results.empty:
            st.error("❌ No results. EMA period may be too long for the date range.")
        else:
            metrics = compute_metrics(results)

            # ===================== METRICS PANEL =====================
            st.markdown("---")
            st.subheader("📊 Performance Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Invested", f"${metrics['total_invested']:,.0f}", f"{metrics['dca_weeks']} weeks")
            col2.metric("Strategy Final Value", f"${metrics['final_strategy']:,.0f}",
                        f"+{metrics['roi_strategy']:.1f}% ROI")
            col3.metric("Benchmark Final (QQQ DCA)", f"${metrics['final_benchmark']:,.0f}",
                        f"+{metrics['roi_benchmark']:.1f}% ROI")

            col4, col5, col6 = st.columns(3)
            col4.metric("Strategy vs Benchmark", f"+${metrics['alpha']:,.0f}",
                        "outperformance" if metrics['alpha'] > 0 else "underperformance")
            col5.metric("Strategy Max Drawdown", f"{metrics['mdd_strategy']:.2f}%")
            col6.metric("Benchmark Max Drawdown", f"{metrics['mdd_benchmark']:.2f}%")

            col7, col8, col9 = st.columns(3)
            col7.metric("Duration", f"{metrics['years']:.1f} years")
            col8.metric("Tier Switches", f"{metrics['switches']}")
            col9.metric("Reset Events", f"{metrics['resets']}")

            # ===================== PORTFOLIO CHART =====================
            st.markdown("---")
            st.subheader("📈 Portfolio Growth: Strategy vs Benchmark vs Invested")

            chart_df = results[["Date", "Portfolio Value ($)", "Benchmark (QQQ DCA)", "Total Invested ($)"]].copy()
            chart_df["Date"] = pd.to_datetime(chart_df["Date"])
            chart_df = chart_df.set_index("Date")
            st.line_chart(chart_df, use_container_width=True)

            # ===================== EMA DISTANCE CHART =====================
            st.subheader("📉 TQQQ EMA Distance % (Tier Trigger Zones)")

            ema_chart = results[["Date", "EMA Dist %"]].copy()
            ema_chart["Date"] = pd.to_datetime(ema_chart["Date"])
            ema_chart = ema_chart.set_index("Date")

            # Add tier lines info
            st.caption(f"Tier triggers at: {tier1}%, {tier2}%, {tier3}%, {tier4}% | Reset at: {reset_threshold}%")
            st.area_chart(ema_chart, use_container_width=True)

            # ===================== ALLOCATION CHART =====================
            st.subheader("🏦 Asset Allocation Over Time")

            alloc_chart = results[["Date", "QQQ Value ($)", "TQQQ Value ($)"]].copy()
            alloc_chart["Date"] = pd.to_datetime(alloc_chart["Date"])
            alloc_chart = alloc_chart.set_index("Date")
            st.area_chart(alloc_chart, use_container_width=True)

            # ===================== EVENT LOG =====================
            st.subheader("📋 Switching Events")
            events = results[results["Action"].isin(["RESET"]) | results["Action"].str.contains("TIER")]
            events_display = events[["Date", "Action", "EMA Dist %", "Transfer ($)",
                                     "Portfolio Value ($)", "QQQ Shares", "TQQQ Shares", "Tiers Active"]]
            st.dataframe(events_display, use_container_width=True, height=300)

            # ===================== CSV DOWNLOAD =====================
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

            # ===================== RAW DATA =====================
            with st.expander("📋 View Raw Data (first 100 rows)"):
                st.dataframe(results.head(100), use_container_width=True)

else:
    # Landing page when no backtest has been run
    st.info("👈 Configure parameters in the sidebar and click **Run Backtest**")

    st.markdown("""
    ### How This Strategy Works

    | Step | Rule |
    |------|------|
    | **1. Weekly DCA** | Every week, deposit a fixed amount (e.g., $200) and buy QQQ shares |
    | **2. Signal** | Calculate TQQQ's % distance below its 200-day EMA daily |
    | **3. Tier Switches** | When TQQQ drops -20%, -30%, -40%, -50% below EMA → transfer 25% of QQQ to TQQQ at each level |
    | **4. Reset** | When TQQQ recovers above EMA → reset all tier flags for next cycle |
    | **5. Accumulation** | During drawdowns you accumulate discounted TQQQ; during recoveries you benefit from 3x leverage |

    ### Key Differences from Lump-Sum
    - **No initial lump sum** — capital builds gradually through weekly DCA
    - **Signal on TQQQ** — EMA distance measured on the leveraged asset (more volatile = earlier triggers)
    - **Transfers from QQQ** — only move existing QQQ holdings into TQQQ (no new cash)

    ### Quick Start
    1. Leave defaults or adjust sliders
    2. Click **🚀 Run Backtest**
    3. Compare Strategy vs pure QQQ DCA benchmark
    4. Download CSV for detailed analysis
    """)

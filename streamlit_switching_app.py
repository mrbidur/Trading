"""
QQQ/TQQQ Two-Way Asset Rotation Strategy — Streamlit Web App
==============================================================
Weekly DCA into QQQ with:
  - DOWNWARD ENTRY: Multi-tier switching QQQ → TQQQ when TQQQ drops below EMA
  - UPWARD EXIT: Full rotation TQQQ → QQQ when TQQQ surges above EMA

All thresholds are adjustable via interactive sliders.

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
    page_title="QQQ/TQQQ Two-Way Rotation Backtest",
    page_icon="🔄",
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

    if isinstance(base_df.columns, pd.MultiIndex):
        base_df.columns = base_df.columns.get_level_values(0)
    if isinstance(lev_df.columns, pd.MultiIndex):
        lev_df.columns = lev_df.columns.get_level_values(0)

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
# BACKTEST ENGINE — Two-Way Rotation
# =============================================================================
def run_backtest(
    data: pd.DataFrame,
    ema_period: int,
    weekly_dca: float,
    entry_tiers: List[float],
    entry_transfer_pct: float,
    exit_threshold: float,
    reset_threshold: float,
) -> pd.DataFrame:
    """
    Two-Way Asset Rotation Strategy:

    WEEKLY DCA:
      Every new week → deposit $weekly_dca → buy QQQ

    DOWNWARD ENTRY (QQQ → TQQQ):
      When TQQQ's EMA distance crosses below each entry tier,
      transfer entry_transfer_pct% of QQQ holdings into TQQQ.
      Tiers fire sequentially, once per cycle.

    UPWARD EXIT / PROFIT-BOOKING (TQQQ → QQQ):
      When TQQQ's EMA distance surges above exit_threshold,
      sell ALL TQQQ shares and rotate proceeds back into QQQ.
      This locks in leveraged gains and resets to base exposure.

    RESET:
      Entry tier flags reset when TQQQ recovers above reset_threshold
      (allowing re-entry on the next drawdown cycle).
      Exit can fire independently whenever the threshold is crossed.
    """
    data = data.copy()

    # Compute EMA on TQQQ
    data["tqqq_ema"] = data["tqqq_close"].ewm(span=ema_period, adjust=False).mean()
    data["ema_dist_pct"] = ((data["tqqq_close"] - data["tqqq_ema"]) / data["tqqq_ema"]) * 100.0
    data = data.dropna()

    if data.empty:
        return pd.DataFrame()

    # State
    qqq_shares = 0.0
    tqqq_shares = 0.0
    total_invested = 0.0
    n_tiers = len(entry_tiers)
    tiers_triggered = [False] * n_tiers
    last_dca_week = None

    # Benchmark: pure QQQ DCA (same weekly amount, no rotation)
    bench_qqq_shares = 0.0
    bench_invested = 0.0

    # Track profit-booking events
    exit_armed = True  # Can fire exit anytime TQQQ is held

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

            action = "DCA"

        # ----------------------------------------------------------
        # UPWARD EXIT: TQQQ → QQQ (Profit Booking)
        # When TQQQ surges above the exit threshold, sell ALL TQQQ
        # and rotate proceeds back into QQQ to lock in profits.
        # ----------------------------------------------------------
        if dist_pct >= exit_threshold and tqqq_shares > 0:
            # Sell all TQQQ
            cash_from_tqqq = tqqq_shares * tqqq_price
            # Buy QQQ with proceeds
            qqq_bought = cash_from_tqqq / qqq_price
            qqq_shares += qqq_bought
            transfer_amount = cash_from_tqqq

            tqqq_shares = 0.0
            action = f"EXIT (+{exit_threshold:.0f}%)"

            # Reset entry tiers after profit-booking (ready for next cycle)
            tiers_triggered = [False] * n_tiers

        # ----------------------------------------------------------
        # ENTRY TIER RESET: When TQQQ recovers above reset threshold
        # ----------------------------------------------------------
        elif dist_pct > reset_threshold:
            if any(tiers_triggered):
                if action == "DCA":
                    action = "DCA + RESET"
                else:
                    action = "RESET"
            tiers_triggered = [False] * n_tiers

        # ----------------------------------------------------------
        # DOWNWARD ENTRY: QQQ → TQQQ (Tier Switching)
        # ----------------------------------------------------------
        else:
            for t_idx, thresh in enumerate(entry_tiers):
                if dist_pct <= thresh and not tiers_triggered[t_idx] and qqq_shares > 0:
                    tiers_triggered[t_idx] = True

                    # Transfer % of QQQ holdings into TQQQ
                    shares_to_sell = qqq_shares * (entry_transfer_pct / 100.0)
                    cash_from_sale = shares_to_sell * qqq_price
                    tqqq_bought = cash_from_sale / tqqq_price

                    qqq_shares -= shares_to_sell
                    tqqq_shares += tqqq_bought
                    transfer_amount = cash_from_sale

                    tier_action = f"ENTRY T{t_idx + 1} ({thresh:.0f}%)"
                    action = tier_action if action == "HOLD" else f"DCA + {tier_action}"
                    break  # Only one tier per day

        # ----------------------------------------------------------
        # PORTFOLIO VALUATION
        # ----------------------------------------------------------
        qqq_value = qqq_shares * qqq_price
        tqqq_value = tqqq_shares * tqqq_price
        total_portfolio = qqq_value + tqqq_value
        benchmark_value = bench_qqq_shares * qqq_price

        # Allocation percentages
        qqq_alloc = (qqq_value / total_portfolio * 100) if total_portfolio > 0 else 100
        tqqq_alloc = (tqqq_value / total_portfolio * 100) if total_portfolio > 0 else 0

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
            "QQQ Alloc %": round(qqq_alloc, 1),
            "TQQQ Alloc %": round(tqqq_alloc, 1),
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

    roi_strat = (final_strat / total_invested - 1) * 100
    roi_bench = (final_bench / total_invested - 1) * 100

    # MDD
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

    entries = results[results["Action"].str.contains("ENTRY")].shape[0]
    exits = results[results["Action"].str.contains("EXIT")].shape[0]
    resets = results[results["Action"].str.contains("RESET")].shape[0]

    return {
        "total_invested": total_invested,
        "final_strategy": final_strat,
        "final_benchmark": final_bench,
        "roi_strategy": roi_strat,
        "roi_benchmark": roi_bench,
        "mdd_strategy": mdd_strat,
        "mdd_benchmark": mdd_bench,
        "years": years,
        "entries": entries,
        "exits": exits,
        "resets": resets,
        "alpha": final_strat - final_bench,
        "alpha_pct": (final_strat / final_bench - 1) * 100 if final_bench > 0 else 0,
    }


# =============================================================================
# SIDEBAR — INTERACTIVE CONTROLS
# =============================================================================
st.sidebar.title("🔄 Two-Way Rotation")

st.sidebar.markdown("### 💰 Weekly DCA")
weekly_dca = st.sidebar.slider(
    "Weekly Deposit ($)", min_value=50, max_value=2000, value=200, step=50,
    help="Fixed amount invested into QQQ every week"
)

st.sidebar.markdown("### 📅 Timeline")
start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 2, 11))
end_date = st.sidebar.date_input("End Date", value=date.today())

st.sidebar.markdown("### 📈 EMA Settings")
ema_period = st.sidebar.slider(
    "EMA Period (days)", min_value=10, max_value=500, value=200, step=10,
    help="Calculated on TQQQ price"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📉 DOWNWARD ENTRY (QQQ → TQQQ)")
st.sidebar.caption("When TQQQ drops below EMA by these %s, move QQQ into TQQQ")
tier1 = st.sidebar.slider("Entry Tier 1 (%)", min_value=-80, max_value=0, value=-20, step=5)
tier2 = st.sidebar.slider("Entry Tier 2 (%)", min_value=-80, max_value=0, value=-30, step=5)
tier3 = st.sidebar.slider("Entry Tier 3 (%)", min_value=-80, max_value=0, value=-40, step=5)
tier4 = st.sidebar.slider("Entry Tier 4 (%)", min_value=-80, max_value=0, value=-50, step=5)
entry_transfer_pct = st.sidebar.slider(
    "Transfer % per Entry Tier", min_value=5, max_value=100, value=25, step=5,
    help="% of current QQQ holdings moved to TQQQ at each tier"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 UPWARD EXIT (TQQQ → QQQ)")
st.sidebar.caption("When TQQQ surges above EMA by this %, sell ALL TQQQ back to QQQ")
exit_threshold = st.sidebar.slider(
    "Profit-Booking Threshold (%)", min_value=10, max_value=150, value=40, step=5,
    help="TQQQ distance above EMA that triggers full rotation back to QQQ"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔁 Reset")
reset_threshold = st.sidebar.slider(
    "Entry Reset Threshold (%)", min_value=-10, max_value=20, value=0, step=1,
    help="EMA dist above which entry tier flags reset for next cycle"
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")


# =============================================================================
# MAIN AREA
# =============================================================================
st.title("🔄 QQQ/TQQQ Two-Way Asset Rotation")
st.caption(
    f"Weekly ${weekly_dca} DCA → QQQ | "
    f"Entry into TQQQ at {tier1}%/{tier2}%/{tier3}%/{tier4}% ({entry_transfer_pct}% per tier) | "
    f"Exit TQQQ → QQQ at +{exit_threshold}% above EMA"
)

if run_button:
    entry_tiers = sorted([tier1, tier2, tier3, tier4])

    with st.spinner("📡 Fetching market data..."):
        data = fetch_data(
            "QQQ", "TQQQ",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

    if data.empty:
        st.error("❌ No data found. TQQQ starts Feb 2010.")
    else:
        with st.spinner("⚙️ Running two-way rotation simulation..."):
            results = run_backtest(
                data=data,
                ema_period=ema_period,
                weekly_dca=weekly_dca,
                entry_tiers=entry_tiers,
                entry_transfer_pct=entry_transfer_pct,
                exit_threshold=exit_threshold,
                reset_threshold=reset_threshold,
            )

        if results.empty:
            st.error("❌ No results produced.")
        else:
            metrics = compute_metrics(results)

            # ===================== METRICS =====================
            st.markdown("---")
            st.subheader("📊 Performance Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Invested", f"${metrics['total_invested']:,.0f}")
            col2.metric("Strategy Final", f"${metrics['final_strategy']:,.0f}",
                        f"+{metrics['roi_strategy']:.1f}% ROI")
            col3.metric("Benchmark (QQQ DCA)", f"${metrics['final_benchmark']:,.0f}",
                        f"+{metrics['roi_benchmark']:.1f}% ROI")

            col4, col5, col6 = st.columns(3)
            alpha_color = "normal" if metrics['alpha'] >= 0 else "inverse"
            col4.metric("Alpha vs Benchmark",
                        f"${metrics['alpha']:+,.0f}",
                        f"{metrics['alpha_pct']:+.1f}%",
                        delta_color=alpha_color)
            col5.metric("Strategy MDD", f"{metrics['mdd_strategy']:.1f}%")
            col6.metric("Benchmark MDD", f"{metrics['mdd_benchmark']:.1f}%")

            col7, col8, col9, col10 = st.columns(4)
            col7.metric("Duration", f"{metrics['years']:.1f} yrs")
            col8.metric("Entry Switches", f"{metrics['entries']}")
            col9.metric("Profit Exits", f"{metrics['exits']}")
            col10.metric("Resets", f"{metrics['resets']}")

            # ===================== PORTFOLIO CHART =====================
            st.markdown("---")
            st.subheader("📈 Portfolio Growth")

            chart_df = results[["Date", "Portfolio Value ($)", "Benchmark (QQQ DCA)", "Total Invested ($)"]].copy()
            chart_df["Date"] = pd.to_datetime(chart_df["Date"])
            chart_df = chart_df.set_index("Date")
            st.line_chart(chart_df, use_container_width=True)

            # ===================== EMA DISTANCE =====================
            st.subheader("📉 TQQQ EMA Distance % — Entry & Exit Zones")
            st.caption(
                f"🟢 Entry tiers: {tier1}%, {tier2}%, {tier3}%, {tier4}% | "
                f"🔴 Exit threshold: +{exit_threshold}% | "
                f"⚪ Reset: {reset_threshold}%"
            )

            ema_chart = results[["Date", "EMA Dist %"]].copy()
            ema_chart["Date"] = pd.to_datetime(ema_chart["Date"])
            ema_chart = ema_chart.set_index("Date")
            st.area_chart(ema_chart, use_container_width=True)

            # ===================== ALLOCATION =====================
            st.subheader("🏦 Asset Allocation")

            alloc_chart = results[["Date", "QQQ Value ($)", "TQQQ Value ($)"]].copy()
            alloc_chart["Date"] = pd.to_datetime(alloc_chart["Date"])
            alloc_chart = alloc_chart.set_index("Date")
            st.area_chart(alloc_chart, use_container_width=True)

            # ===================== ALLOCATION % =====================
            st.subheader("📊 Allocation % Over Time")
            alloc_pct = results[["Date", "QQQ Alloc %", "TQQQ Alloc %"]].copy()
            alloc_pct["Date"] = pd.to_datetime(alloc_pct["Date"])
            alloc_pct = alloc_pct.set_index("Date")
            st.area_chart(alloc_pct, use_container_width=True)

            # ===================== EVENTS =====================
            st.subheader("📋 Rotation Events")
            events = results[
                results["Action"].str.contains("ENTRY") |
                results["Action"].str.contains("EXIT") |
                results["Action"].str.contains("RESET")
            ]
            events_display = events[["Date", "Action", "EMA Dist %", "Transfer ($)",
                                     "Portfolio Value ($)", "QQQ Alloc %", "TQQQ Alloc %", "Tiers Active"]]
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
    st.info("👈 Set your entry tiers, exit threshold, and DCA amount in the sidebar, then click **Run Backtest**")

    st.markdown("""
    ### 🔄 How Two-Way Rotation Works

    | Direction | Trigger | Action |
    |-----------|---------|--------|
    | **⬇️ ENTRY** | TQQQ drops -20%, -30%, -40%, -50% below EMA | Transfer 25% of QQQ → TQQQ at each tier |
    | **⬆️ EXIT** | TQQQ surges +40% above EMA | Sell ALL TQQQ → rotate back to QQQ |
    | **🔁 RESET** | TQQQ recovers above EMA (>0%) | Entry tier flags clear for next cycle |
    | **💰 DCA** | Every week | Deposit $200 → buy QQQ |

    ### Why Two-Way?
    - **Entry tiers** accumulate discounted TQQQ during crashes (3x leverage at low prices)
    - **Exit threshold** locks in profits when TQQQ becomes overextended (prevents round-trip losses)
    - **Result**: Capture leveraged upside during recoveries, book profits at peaks, repeat

    ### Quick Start
    1. Leave defaults or adjust sliders
    2. Click **🚀 Run Backtest**
    3. Compare Strategy vs pure QQQ DCA
    4. Experiment with different exit thresholds (+30% vs +50% vs +80%)
    """)

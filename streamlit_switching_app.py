"""
QQQ/TQQQ Phased Two-Way Asset Rotation — Streamlit Web App
=============================================================
Fully customizable multi-tier entry AND multi-tier exit with
independent weights and dynamic reset criteria per tier.

ENTRY: QQQ → TQQQ at configurable drawdown tiers (each with its own weight)
EXIT:  TQQQ → QQQ at configurable surge tiers (each with its own weight + reset level)

Run: streamlit run streamlit_switching_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime
from typing import List, Tuple
import io

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="QQQ/TQQQ Phased Two-Way Rotation",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# DATA FETCHING (cached)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(start: str, end: str) -> pd.DataFrame:
    """Fetch and align QQQ + TQQQ daily close prices."""
    qqq = yf.Ticker("QQQ")
    tqqq = yf.Ticker("TQQQ")

    qqq_df = qqq.history(start=start, end=end, auto_adjust=True)
    tqqq_df = tqqq.history(start=start, end=end, auto_adjust=True)

    if qqq_df.empty or tqqq_df.empty:
        return pd.DataFrame()

    for df in [qqq_df, tqqq_df]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    combined = pd.DataFrame({
        "qqq": qqq_df["Close"],
        "tqqq": tqqq_df["Close"],
    }).dropna()
    return combined


# =============================================================================
# BACKTEST ENGINE — Phased Two-Way Rotation
# =============================================================================
def run_backtest(
    data: pd.DataFrame,
    ema_period: int,
    weekly_dca: float,
    # Entry config: list of (threshold_pct, transfer_weight_pct)
    entry_tiers: List[Tuple[float, float]],
    entry_reset_threshold: float,
    # Exit config: list of (threshold_pct, sell_weight_pct, reset_drop_to_pct)
    exit_tiers: List[Tuple[float, float, float]],
) -> pd.DataFrame:
    """
    Phased Two-Way Rotation:

    ENTRY (QQQ → TQQQ):
      Each tier has its own threshold and weight.
      When TQQQ dist crosses below tier threshold, transfer that tier's
      weight% of QQQ into TQQQ. Fires once per cycle.
      Resets when dist > entry_reset_threshold.

    EXIT (TQQQ → QQQ):
      Each tier has its own threshold, weight, AND reset level.
      When TQQQ dist surges above tier threshold, sell that tier's
      weight% of TQQQ back into QQQ.
      That tier resets ONLY when dist drops back below its reset level.
      This prevents daily re-triggering.
    """
    data = data.copy()
    data["ema"] = data["tqqq"].ewm(span=ema_period, adjust=False).mean()
    data["dist"] = ((data["tqqq"] - data["ema"]) / data["ema"]) * 100.0
    data = data.dropna()

    if data.empty:
        return pd.DataFrame()

    # State
    qqq_shares = 0.0
    tqqq_shares = 0.0
    total_invested = 0.0
    last_dca_week = None

    # Entry tier tracking
    n_entry = len(entry_tiers)
    entry_triggered = [False] * n_entry

    # Exit tier tracking (each has independent trigger + reset)
    n_exit = len(exit_tiers)
    exit_triggered = [False] * n_exit

    # Benchmark
    bench_shares = 0.0

    rows = []

    for dt, row in data.iterrows():
        qp = float(row["qqq"])
        tp = float(row["tqqq"])
        ema_val = float(row["ema"])
        dist = float(row["dist"])

        current_dt = dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else dt
        current_week = (current_dt.isocalendar()[0], current_dt.isocalendar()[1])
        is_dca = (current_week != last_dca_week)

        action = "HOLD"
        deposit = 0.0
        transfer = 0.0

        # --- WEEKLY DCA ---
        if is_dca:
            last_dca_week = current_week
            deposit = weekly_dca
            qqq_shares += deposit / qp
            total_invested += deposit
            bench_shares += deposit / qp
            action = "DCA"

        # --- ENTRY RESET: when dist rises above entry_reset_threshold ---
        if dist > entry_reset_threshold:
            if any(entry_triggered):
                action = "DCA + ENTRY RESET" if action == "DCA" else "ENTRY RESET"
            entry_triggered = [False] * n_entry

        # --- EXIT TIER RESETS: each tier resets independently ---
        for e_idx in range(n_exit):
            if exit_triggered[e_idx]:
                _, _, reset_level = exit_tiers[e_idx]
                if dist < reset_level:
                    exit_triggered[e_idx] = False

        # --- PHASED EXIT: TQQQ → QQQ (profit booking at each tier) ---
        for e_idx, (threshold, weight, _) in enumerate(exit_tiers):
            if dist >= threshold and not exit_triggered[e_idx] and tqqq_shares > 0:
                exit_triggered[e_idx] = True
                shares_to_sell = tqqq_shares * (weight / 100.0)
                cash = shares_to_sell * tp
                qqq_shares += cash / qp
                tqqq_shares -= shares_to_sell
                transfer = cash
                tier_label = f"EXIT T{e_idx+1} (+{threshold:.0f}%, {weight:.0f}%)"
                action = tier_label if action == "HOLD" else f"{action} + {tier_label}"
                break  # One exit tier per day

        # --- PHASED ENTRY: QQQ → TQQQ (drawdown tiers) ---
        if dist <= entry_reset_threshold:  # Only check entries when below reset
            for t_idx, (threshold, weight) in enumerate(entry_tiers):
                if dist <= threshold and not entry_triggered[t_idx] and qqq_shares > 0:
                    entry_triggered[t_idx] = True
                    shares_to_sell = qqq_shares * (weight / 100.0)
                    cash = shares_to_sell * qp
                    tqqq_shares += cash / tp
                    qqq_shares -= shares_to_sell
                    transfer = cash
                    tier_label = f"ENTRY T{t_idx+1} ({threshold:.0f}%, {weight:.0f}%)"
                    action = tier_label if action in ["HOLD", "DCA"] else f"{action} + {tier_label}"
                    break  # One entry tier per day

        # --- VALUATION ---
        qqq_val = qqq_shares * qp
        tqqq_val = tqqq_shares * tp
        total = qqq_val + tqqq_val
        bench_val = bench_shares * qp
        qqq_pct = (qqq_val / total * 100) if total > 0 else 100
        tqqq_pct = (tqqq_val / total * 100) if total > 0 else 0

        rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "QQQ Price": round(qp, 4),
            "TQQQ Price": round(tp, 4),
            "TQQQ EMA": round(ema_val, 4),
            "EMA Dist %": round(dist, 2),
            "QQQ Shares": round(qqq_shares, 4),
            "TQQQ Shares": round(tqqq_shares, 4),
            "QQQ Value ($)": round(qqq_val, 2),
            "TQQQ Value ($)": round(tqqq_val, 2),
            "Portfolio ($)": round(total, 2),
            "Benchmark ($)": round(bench_val, 2),
            "Invested ($)": round(total_invested, 2),
            "QQQ %": round(qqq_pct, 1),
            "TQQQ %": round(tqqq_pct, 1),
            "Entry Tiers": sum(entry_triggered),
            "Exit Tiers": sum(exit_triggered),
            "Action": action,
            "Deposit ($)": round(deposit, 2),
            "Transfer ($)": round(transfer, 2),
        })

    return pd.DataFrame(rows)


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    v = df["Portfolio ($)"].values
    b = df["Benchmark ($)"].values
    inv = df["Invested ($)"].values[-1]
    yrs = max((pd.to_datetime(df["Date"].iloc[-1]) - pd.to_datetime(df["Date"].iloc[0])).days / 365.25, 0.01)

    final_s, final_b = v[-1], b[-1]
    roi_s = (final_s / inv - 1) * 100
    roi_b = (final_b / inv - 1) * 100
    cagr_s = ((final_s / inv) ** (1 / yrs) - 1) * 100
    cagr_b = ((final_b / inv) ** (1 / yrs) - 1) * 100

    w = min(252, len(v) // 4)
    def mdd(arr):
        a = arr[w:] if len(arr) > w else arr
        if len(a) == 0: return 0
        p = np.maximum.accumulate(a)
        return float(((a - p) / p).min()) * 100

    entries = df["Action"].str.contains("ENTRY T").sum()
    exits = df["Action"].str.contains("EXIT T").sum()
    resets = df["Action"].str.contains("RESET").sum()

    return dict(
        invested=inv, final_s=final_s, final_b=final_b,
        roi_s=roi_s, roi_b=roi_b, cagr_s=cagr_s, cagr_b=cagr_b,
        mdd_s=mdd(v), mdd_b=mdd(b), years=yrs,
        alpha=final_s - final_b, alpha_pct=(final_s/final_b-1)*100 if final_b>0 else 0,
        entries=entries, exits=exits, resets=resets,
    )


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("🔄 Phased Rotation Controls")

# --- DCA ---
st.sidebar.markdown("## 💰 Weekly DCA")
weekly_dca = st.sidebar.slider("Weekly Deposit ($)", 50, 2000, 200, 50)

# --- Timeline ---
st.sidebar.markdown("## 📅 Timeline")
col_s, col_e = st.sidebar.columns(2)
start_date = col_s.date_input("Start", value=datetime(2010, 2, 11))
end_date = col_e.date_input("End", value=date.today())

# --- EMA ---
st.sidebar.markdown("## 📈 EMA")
ema_period = st.sidebar.slider("EMA Period (days)", 10, 500, 200, 10)

# --- ENTRY TIERS ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 📉 ENTRY TIERS (QQQ → TQQQ)")
st.sidebar.caption("When TQQQ drops below EMA by X%, transfer Y% of QQQ → TQQQ")

n_entry_tiers = st.sidebar.number_input("Number of Entry Tiers", 1, 8, 4, 1)
entry_tiers = []
for i in range(int(n_entry_tiers)):
    c1, c2 = st.sidebar.columns(2)
    thresh = c1.slider(f"E{i+1} Trigger %", -80, 0, -20 - i*10, 5, key=f"et_{i}")
    weight = c2.slider(f"E{i+1} Weight %", 5, 100, 25, 5, key=f"ew_{i}")
    entry_tiers.append((thresh, weight))

entry_reset = st.sidebar.slider("Entry Reset Threshold (%)", -10, 20, 0, 1,
    help="Entry flags reset when TQQQ dist rises above this level")

# --- EXIT TIERS ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 📈 EXIT TIERS (TQQQ → QQQ)")
st.sidebar.caption("When TQQQ surges above EMA by X%, sell Y% of TQQQ → QQQ. Resets when dist drops below Z%.")

n_exit_tiers = st.sidebar.number_input("Number of Exit Tiers", 1, 8, 3, 1)
exit_tiers = []
for i in range(int(n_exit_tiers)):
    c1, c2, c3 = st.sidebar.columns(3)
    thresh = c1.slider(f"X{i+1} Trigger", 10, 150, 30 + i*15, 5, key=f"xt_{i}")
    weight = c2.slider(f"X{i+1} Sell %", 5, 100, 50 if i == 0 else 100, 5, key=f"xw_{i}")
    reset_at = c3.slider(f"X{i+1} Reset", -10, 100, max(0, 30 + i*15 - 10), 5, key=f"xr_{i}")
    exit_tiers.append((thresh, weight, reset_at))

# --- RUN ---
st.sidebar.markdown("---")
run = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")


# =============================================================================
# MAIN
# =============================================================================
st.title("🔄 Phased Two-Way QQQ/TQQQ Rotation")

# Summary of config
entry_str = " | ".join([f"{t:.0f}%→{w:.0f}%" for t, w in entry_tiers])
exit_str = " | ".join([f"+{t:.0f}%→sell {w:.0f}% (reset@{r:.0f}%)" for t, w, r in exit_tiers])
st.caption(f"DCA ${weekly_dca}/wk | Entry: [{entry_str}] | Exit: [{exit_str}]")

if run:
    with st.spinner("📡 Fetching QQQ & TQQQ data..."):
        data = fetch_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if data.empty:
        st.error("❌ No data. TQQQ starts Feb 2010.")
    else:
        with st.spinner("⚙️ Running phased rotation..."):
            results = run_backtest(
                data, ema_period, weekly_dca,
                entry_tiers, entry_reset, exit_tiers,
            )

        if results.empty:
            st.error("❌ No results.")
        else:
            m = compute_metrics(results)

            # === METRICS ===
            st.markdown("---")
            st.subheader("📊 Performance")

            r1 = st.columns(4)
            r1[0].metric("Invested", f"${m['invested']:,.0f}")
            r1[1].metric("Strategy", f"${m['final_s']:,.0f}", f"+{m['roi_s']:.1f}%")
            r1[2].metric("Benchmark (QQQ DCA)", f"${m['final_b']:,.0f}", f"+{m['roi_b']:.1f}%")
            r1[3].metric("Alpha", f"${m['alpha']:+,.0f}", f"{m['alpha_pct']:+.1f}%",
                         delta_color="normal" if m['alpha'] >= 0 else "inverse")

            r2 = st.columns(4)
            r2[0].metric("CAGR (Strategy)", f"{m['cagr_s']:.2f}%")
            r2[1].metric("CAGR (Benchmark)", f"{m['cagr_b']:.2f}%")
            r2[2].metric("MDD (Strategy)", f"{m['mdd_s']:.1f}%")
            r2[3].metric("MDD (Benchmark)", f"{m['mdd_b']:.1f}%")

            r3 = st.columns(4)
            r3[0].metric("Duration", f"{m['years']:.1f} yrs")
            r3[1].metric("Entry Switches", m['entries'])
            r3[2].metric("Exit Switches", m['exits'])
            r3[3].metric("Resets", m['resets'])

            # === CHARTS ===
            st.markdown("---")
            st.subheader("📈 Portfolio Growth")
            c = results[["Date","Portfolio ($)","Benchmark ($)","Invested ($)"]].copy()
            c["Date"] = pd.to_datetime(c["Date"])
            st.line_chart(c.set_index("Date"), use_container_width=True)

            st.subheader("📉 TQQQ EMA Distance %")
            st.caption(f"Entry zones: {[t for t,_ in entry_tiers]} | Exit zones: {[t for t,_,_ in exit_tiers]}")
            e = results[["Date","EMA Dist %"]].copy()
            e["Date"] = pd.to_datetime(e["Date"])
            st.area_chart(e.set_index("Date"), use_container_width=True)

            st.subheader("🏦 Allocation ($)")
            a = results[["Date","QQQ Value ($)","TQQQ Value ($)"]].copy()
            a["Date"] = pd.to_datetime(a["Date"])
            st.area_chart(a.set_index("Date"), use_container_width=True)

            st.subheader("📊 Allocation %")
            p = results[["Date","QQQ %","TQQQ %"]].copy()
            p["Date"] = pd.to_datetime(p["Date"])
            st.area_chart(p.set_index("Date"), use_container_width=True)

            # === EVENTS ===
            st.subheader("📋 Rotation Events")
            ev = results[results["Action"].str.contains("ENTRY|EXIT|RESET")]
            st.dataframe(
                ev[["Date","Action","EMA Dist %","Transfer ($)","Portfolio ($)","QQQ %","TQQQ %"]],
                use_container_width=True, height=300,
            )

            # === CSV ===
            st.markdown("---")
            buf = io.StringIO()
            results.to_csv(buf, index=False)
            st.download_button("📥 Download CSV", buf.getvalue(),
                               "strategy_backtest_results.csv", "text/csv",
                               use_container_width=True)

            with st.expander("📋 Raw Data (first 100 rows)"):
                st.dataframe(results.head(100), use_container_width=True)

else:
    st.info("👈 Configure all parameters in the sidebar and click **Run Backtest**")
    st.markdown("""
    ### 🔄 Phased Two-Way Rotation

    | | Direction | Trigger | Action | Reset |
    |-|-----------|---------|--------|-------|
    | ⬇️ | **ENTRY** | TQQQ drops -20% below EMA | Transfer 25% QQQ → TQQQ | When dist > 0% |
    | ⬇️ | **ENTRY** | TQQQ drops -30% below EMA | Transfer 25% QQQ → TQQQ | When dist > 0% |
    | ⬆️ | **EXIT** | TQQQ surges +30% above EMA | Sell 50% TQQQ → QQQ | When dist drops < +20% |
    | ⬆️ | **EXIT** | TQQQ surges +50% above EMA | Sell 100% TQQQ → QQQ | When dist drops < +40% |

    ### Key Features
    - **Independent weights per tier**: Entry Tier 1 can transfer 20%, Tier 2 can transfer 40%, etc.
    - **Independent exit resets**: Each exit tier has its own reset level to prevent daily re-triggering
    - **Adjustable number of tiers**: Add up to 8 entry tiers and 8 exit tiers
    - **All parameters via sliders**: Experiment in real-time
    """)

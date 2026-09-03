"""
QQQ/TQQQ Phased Two-Way Rotation with Time-Delay Exits — Streamlit App
========================================================================
Fully customizable multi-tier entry AND multi-tier exit with:
  - Per-tier weights (entry and exit)
  - Time-based holding period before exits are allowed
  - Dynamic per-tier reset criteria

ENTRY: QQQ → TQQQ at configurable drawdown tiers
EXIT:  TQQQ → QQQ at configurable surge tiers, gated by a minimum holding delay

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
    page_title="QQQ/TQQQ Rotation + Time-Delay Exits",
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
# BACKTEST ENGINE — Phased Two-Way Rotation with Time-Delay Exits
# =============================================================================
def run_backtest(
    data: pd.DataFrame,
    ema_period: int,
    weekly_dca: float,
    entry_tiers: List[Tuple[float, float]],       # (threshold_pct, weight_pct)
    entry_reset_threshold: float,
    exit_tiers: List[Tuple[float, float, float]], # (threshold_pct, sell_weight_pct, reset_drop_to_pct)
    hold_delay_days: int,                         # min trading days in TQQQ before exit allowed
) -> pd.DataFrame:
    """
    Phased Two-Way Rotation with a time-based holding delay on exits.

    TIME-DELAY RULE:
      - When the FIRST entry into TQQQ occurs, a countdown starts.
      - Exit tiers cannot fire until `hold_delay_days` trading days have
        elapsed since that entry. This prevents premature exits on brief spikes.
      - The delay clock resets each time TQQQ position is rebuilt from zero
        (i.e., a fresh entry after being fully rotated out).
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

    n_entry = len(entry_tiers)
    entry_triggered = [False] * n_entry

    n_exit = len(exit_tiers)
    exit_triggered = [False] * n_exit

    bench_shares = 0.0

    # Time-delay tracking
    days_since_entry = -1  # -1 = no active TQQQ position
    had_tqqq_prev = False

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

        # --- Advance time-delay counter ---
        if tqqq_shares > 0:
            if days_since_entry < 0:
                days_since_entry = 0  # Position just started being tracked
            else:
                days_since_entry += 1
        else:
            days_since_entry = -1  # No position

        # --- WEEKLY DCA ---
        if is_dca:
            last_dca_week = current_week
            deposit = weekly_dca
            qqq_shares += deposit / qp
            total_invested += deposit
            bench_shares += deposit / qp
            action = "DCA"

        # --- ENTRY RESET ---
        if dist > entry_reset_threshold:
            if any(entry_triggered):
                action = "DCA + ENTRY RESET" if action == "DCA" else "ENTRY RESET"
            entry_triggered = [False] * n_entry

        # --- EXIT TIER RESETS (each independent) ---
        for e_idx in range(n_exit):
            if exit_triggered[e_idx]:
                _, _, reset_level = exit_tiers[e_idx]
                if dist < reset_level:
                    exit_triggered[e_idx] = False

        # --- PHASED EXIT (gated by time-delay) ---
        exit_allowed = (days_since_entry >= hold_delay_days) and (tqqq_shares > 0)
        if exit_allowed:
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
                    break

        # --- PHASED ENTRY (QQQ → TQQQ) ---
        if dist <= entry_reset_threshold:
            for t_idx, (threshold, weight) in enumerate(entry_tiers):
                if dist <= threshold and not entry_triggered[t_idx] and qqq_shares > 0:
                    entry_triggered[t_idx] = True
                    was_flat = (tqqq_shares == 0)
                    shares_to_sell = qqq_shares * (weight / 100.0)
                    cash = shares_to_sell * qp
                    tqqq_shares += cash / tp
                    qqq_shares -= shares_to_sell
                    transfer = cash
                    # Start/reset the delay clock on a fresh entry
                    if was_flat:
                        days_since_entry = 0
                    tier_label = f"ENTRY T{t_idx+1} ({threshold:.0f}%, {weight:.0f}%)"
                    action = tier_label if action in ["HOLD", "DCA"] else f"{action} + {tier_label}"
                    break

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
            "Days Held": days_since_entry if days_since_entry >= 0 else 0,
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
st.sidebar.title("🔄 Rotation Controls")

st.sidebar.markdown("## 💰 Weekly DCA")
weekly_dca = st.sidebar.slider("Weekly Deposit ($)", 50, 2000, 200, 50)

st.sidebar.markdown("## 📅 Timeline")
col_s, col_e = st.sidebar.columns(2)
start_date = col_s.date_input("Start", value=datetime(2010, 2, 11))
end_date = col_e.date_input("End", value=date.today())

st.sidebar.markdown("## 📈 EMA")
ema_period = st.sidebar.slider("EMA Period (days)", 10, 500, 200, 10)

# --- ENTRY ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 📉 ENTRY TIERS (QQQ → TQQQ)")
st.sidebar.caption("TQQQ drops X% below EMA → transfer Y% of QQQ → TQQQ")
n_entry_tiers = st.sidebar.number_input("Number of Entry Tiers", 1, 8, 4, 1)
entry_tiers = []
for i in range(int(n_entry_tiers)):
    c1, c2 = st.sidebar.columns(2)
    thresh = c1.slider(f"E{i+1} Trigger %", -80, 0, -20 - i*10, 5, key=f"et_{i}")
    weight = c2.slider(f"E{i+1} Weight %", 5, 100, 25, 5, key=f"ew_{i}")
    entry_tiers.append((thresh, weight))
entry_reset = st.sidebar.slider("Entry Reset Threshold (%)", -10, 20, 0, 1,
    help="Entry flags reset when TQQQ dist rises above this")

# --- TIME DELAY ---
st.sidebar.markdown("---")
st.sidebar.markdown("## ⏳ EXIT TIME-DELAY")
st.sidebar.caption("Minimum holding time in TQQQ before ANY exit can fire")
delay_unit = st.sidebar.radio("Delay Unit", ["Trading Days", "Weeks"], index=0, horizontal=True)
delay_value = st.sidebar.slider(
    f"Hold Delay ({delay_unit})", 0, 260, 20, 1,
    help="Blocks premature profit-booking on brief spikes"
)
hold_delay_days = delay_value * 5 if delay_unit == "Weeks" else delay_value

# --- EXIT ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 📈 EXIT TIERS (TQQQ → QQQ)")
st.sidebar.caption("TQQQ surges X% above EMA → sell Y% of TQQQ. Resets when dist < Z%.")
n_exit_tiers = st.sidebar.number_input("Number of Exit Tiers", 1, 8, 3, 1)
exit_tiers = []
for i in range(int(n_exit_tiers)):
    c1, c2, c3 = st.sidebar.columns(3)
    thresh = c1.slider(f"X{i+1} Trigger", 10, 150, 30 + i*15, 5, key=f"xt_{i}")
    weight = c2.slider(f"X{i+1} Sell %", 5, 100, 50 if i == 0 else 100, 5, key=f"xw_{i}")
    reset_at = c3.slider(f"X{i+1} Reset", -10, 100, max(0, 30 + i*15 - 10), 5, key=f"xr_{i}")
    exit_tiers.append((thresh, weight, reset_at))

st.sidebar.markdown("---")
run = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")


# =============================================================================
# MAIN
# =============================================================================
st.title("🔄 Phased QQQ/TQQQ Rotation + Time-Delay Exits")

entry_str = " | ".join([f"{t:.0f}%→{w:.0f}%" for t, w in entry_tiers])
exit_str = " | ".join([f"+{t:.0f}%→{w:.0f}%(reset@{r:.0f}%)" for t, w, r in exit_tiers])
st.caption(
    f"DCA ${weekly_dca}/wk | Hold-delay: {delay_value} {delay_unit.lower()} ({hold_delay_days} days) | "
    f"Entry: [{entry_str}] | Exit: [{exit_str}]"
)

if run:
    with st.spinner("📡 Fetching QQQ & TQQQ data..."):
        data = fetch_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if data.empty:
        st.error("❌ No data. TQQQ starts Feb 2010.")
    else:
        with st.spinner("⚙️ Running simulation..."):
            results = run_backtest(
                data, ema_period, weekly_dca,
                entry_tiers, entry_reset, exit_tiers, hold_delay_days,
            )

        if results.empty:
            st.error("❌ No results.")
        else:
            m = compute_metrics(results)

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

            st.subheader("📋 Rotation Events")
            ev = results[results["Action"].str.contains("ENTRY|EXIT|RESET")]
            st.dataframe(
                ev[["Date","Action","EMA Dist %","Days Held","Transfer ($)","Portfolio ($)","QQQ %","TQQQ %"]],
                use_container_width=True, height=300,
            )

            st.markdown("---")
            buf = io.StringIO()
            results.to_csv(buf, index=False)
            st.download_button("📥 Download CSV", buf.getvalue(),
                               "strategy_backtest_results.csv", "text/csv",
                               use_container_width=True)

            with st.expander("📋 Raw Data (first 100 rows)"):
                st.dataframe(results.head(100), use_container_width=True)

else:
    st.info("👈 Configure parameters (including the exit time-delay) and click **Run Backtest**")
    st.markdown("""
    ### ⏳ New: Time-Based Holding Delay on Exits

    | Feature | Description |
    |---------|-------------|
    | **Hold Delay** | Set a minimum number of trading days (or weeks) that must pass after entering TQQQ before any profit-booking exit is allowed |
    | **Purpose** | Prevents premature exits on brief spikes — forces the strategy to hold through short-term noise |
    | **Clock Reset** | The delay clock restarts each time a fresh TQQQ position is built from zero |

    ### 🔄 Full Rule Set

    | Direction | Trigger | Action | Gate/Reset |
    |-----------|---------|--------|-----------|
    | ⬇️ ENTRY | TQQQ drops below EMA tiers | Transfer % of QQQ → TQQQ | Resets above EMA |
    | ⬆️ EXIT | TQQQ surges above EMA tiers | Sell % of TQQQ → QQQ | **Only after hold delay** + per-tier reset |
    | 💰 DCA | Weekly | Buy QQQ | — |

    ### Example
    - Enter TQQQ at -20%, -30% drawdowns
    - Set hold-delay = 20 trading days (~1 month)
    - Even if TQQQ spikes +40% after 5 days, exit is BLOCKED until day 20
    - After delay passes, exit tiers fire normally (+30% sell 50%, +50% sell 100%)
    """)

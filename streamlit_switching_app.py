"""
QQQ/TQQQ/CASH Phased Rotation with Profit-to-Cash Exits — Streamlit App
=========================================================================
Three-asset rotation: CASH (baseline) ↔ TQQQ (leveraged) with profit-booking to CASH.

FLOW:
  - Weekly DCA accumulates into CASH (baseline holding)
  - ENTRY:  When TQQQ drops below EMA tiers → deploy CASH into TQQQ
  - EXIT:   When TQQQ surges above EMA tiers → book profits back into CASH
            (gated by a minimum holding delay)
  - Cash can optionally earn a yield (APY) while sitting as dry powder

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
    page_title="QQQ/TQQQ/CASH Rotation Backtest",
    page_icon="💵",
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
# BACKTEST ENGINE — CASH ↔ TQQQ Rotation with Profit-to-Cash Exits
# =============================================================================
def run_backtest(
    data: pd.DataFrame,
    ema_period: int,
    weekly_dca: float,
    entry_tiers: List[Tuple[float, float]],       # (threshold_pct, weight_pct of CASH)
    entry_reset_threshold: float,
    exit_tiers: List[Tuple[float, float, float]], # (threshold_pct, sell_weight_pct, reset_drop_to_pct)
    hold_delay_days: int,
    cash_apy: float,
) -> pd.DataFrame:
    """
    Three-state rotation: CASH ↔ TQQQ (profit-booking goes to CASH).

    WEEKLY DCA:
      Every new week → add $weekly_dca to CASH balance.

    ENTRY (CASH → TQQQ):
      When TQQQ dist crosses below each entry tier, deploy that tier's
      weight% of CURRENT CASH into TQQQ. Fires once per cycle.
      Resets when dist > entry_reset_threshold.

    EXIT / PROFIT-BOOKING (TQQQ → CASH):
      When TQQQ dist surges above each exit tier (AND hold delay elapsed),
      sell that tier's weight% of TQQQ and move proceeds to CASH.
      Each exit tier resets independently when dist drops below its reset level.

    CASH YIELD:
      Idle cash compounds daily at cash_apy/252 (dry powder earns T-bill yield).

    TIME DELAY:
      Exits blocked until hold_delay_days trading days pass after a fresh
      TQQQ entry (from zero). Prevents premature exits on brief spikes.
    """
    data = data.copy()
    data["ema"] = data["tqqq"].ewm(span=ema_period, adjust=False).mean()
    data["dist"] = ((data["tqqq"] - data["ema"]) / data["ema"]) * 100.0
    data = data.dropna()

    if data.empty:
        return pd.DataFrame()

    # State — starts fully in CASH
    cash = 0.0
    tqqq_shares = 0.0
    total_invested = 0.0
    last_dca_week = None

    n_entry = len(entry_tiers)
    entry_triggered = [False] * n_entry
    n_exit = len(exit_tiers)
    exit_triggered = [False] * n_exit

    # Benchmark: pure QQQ DCA (buy QQQ every week, never sell)
    bench_shares = 0.0

    # Time-delay tracking
    days_since_entry = -1

    # Daily cash yield factor
    daily_yield = (1 + cash_apy / 100.0) ** (1.0 / 252.0)

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

        # --- Cash yield (compounds daily) ---
        if cash > 0 and cash_apy > 0:
            cash *= daily_yield

        # --- Advance time-delay counter ---
        if tqqq_shares > 0:
            days_since_entry = 0 if days_since_entry < 0 else days_since_entry + 1
        else:
            days_since_entry = -1

        # --- WEEKLY DCA into CASH ---
        if is_dca:
            last_dca_week = current_week
            deposit = weekly_dca
            cash += deposit
            total_invested += deposit
            bench_shares += deposit / qp  # benchmark buys QQQ
            action = "DCA"

        # --- ENTRY RESET ---
        if dist > entry_reset_threshold:
            if any(entry_triggered):
                action = "DCA + ENTRY RESET" if action == "DCA" else "ENTRY RESET"
            entry_triggered = [False] * n_entry

        # --- EXIT TIER RESETS (independent) ---
        for e_idx in range(n_exit):
            if exit_triggered[e_idx]:
                _, _, reset_level = exit_tiers[e_idx]
                if dist < reset_level:
                    exit_triggered[e_idx] = False

        # --- PHASED EXIT: TQQQ → CASH (gated by time-delay) ---
        exit_allowed = (days_since_entry >= hold_delay_days) and (tqqq_shares > 0)
        if exit_allowed:
            for e_idx, (threshold, weight, _) in enumerate(exit_tiers):
                if dist >= threshold and not exit_triggered[e_idx] and tqqq_shares > 0:
                    exit_triggered[e_idx] = True
                    shares_to_sell = tqqq_shares * (weight / 100.0)
                    proceeds = shares_to_sell * tp
                    cash += proceeds          # PROFIT BOOKED TO CASH
                    tqqq_shares -= shares_to_sell
                    transfer = proceeds
                    tier_label = f"EXIT→CASH T{e_idx+1} (+{threshold:.0f}%, {weight:.0f}%)"
                    action = tier_label if action == "HOLD" else f"{action} + {tier_label}"
                    break

        # --- PHASED ENTRY: CASH → TQQQ ---
        if dist <= entry_reset_threshold:
            for t_idx, (threshold, weight) in enumerate(entry_tiers):
                if dist <= threshold and not entry_triggered[t_idx] and cash > 0:
                    entry_triggered[t_idx] = True
                    was_flat = (tqqq_shares == 0)
                    deploy_cash = cash * (weight / 100.0)
                    tqqq_shares += deploy_cash / tp
                    cash -= deploy_cash        # CASH DEPLOYED INTO TQQQ
                    transfer = deploy_cash
                    if was_flat:
                        days_since_entry = 0
                    tier_label = f"ENTRY T{t_idx+1} ({threshold:.0f}%, {weight:.0f}%)"
                    action = tier_label if action in ["HOLD", "DCA"] else f"{action} + {tier_label}"
                    break

        # --- VALUATION ---
        tqqq_val = tqqq_shares * tp
        total = cash + tqqq_val
        bench_val = bench_shares * qp
        cash_pct = (cash / total * 100) if total > 0 else 100
        tqqq_pct = (tqqq_val / total * 100) if total > 0 else 0

        rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "QQQ Price": round(qp, 4),
            "TQQQ Price": round(tp, 4),
            "TQQQ EMA": round(ema_val, 4),
            "EMA Dist %": round(dist, 2),
            "Cash ($)": round(cash, 2),
            "TQQQ Shares": round(tqqq_shares, 4),
            "TQQQ Value ($)": round(tqqq_val, 2),
            "Portfolio ($)": round(total, 2),
            "Benchmark ($)": round(bench_val, 2),
            "Invested ($)": round(total_invested, 2),
            "Cash %": round(cash_pct, 1),
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
    exits = df["Action"].str.contains("EXIT").sum()
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
st.sidebar.title("💵 Cash-Rotation Controls")

st.sidebar.markdown("## 💰 Weekly DCA")
weekly_dca = st.sidebar.slider("Weekly Deposit ($)", 50, 2000, 200, 50,
    help="Added to CASH each week (baseline dry powder)")

st.sidebar.markdown("## 📅 Timeline")
col_s, col_e = st.sidebar.columns(2)
start_date = col_s.date_input("Start", value=datetime(2010, 2, 11))
end_date = col_e.date_input("End", value=date.today())

st.sidebar.markdown("## 📈 EMA")
ema_period = st.sidebar.slider("EMA Period (days)", 10, 500, 200, 10)

st.sidebar.markdown("## 🏦 Cash Yield")
cash_apy = st.sidebar.slider("Cash APY (%)", 0.0, 8.0, 4.5, 0.5,
    help="Idle cash earns this yield (T-bill dry powder)")

# --- ENTRY ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 📉 ENTRY TIERS (CASH → TQQQ)")
st.sidebar.caption("TQQQ drops X% below EMA → deploy Y% of CASH into TQQQ")
n_entry_tiers = st.sidebar.number_input("Number of Entry Tiers", 1, 8, 4, 1)
entry_tiers = []
for i in range(int(n_entry_tiers)):
    c1, c2 = st.sidebar.columns(2)
    thresh = c1.slider(f"E{i+1} Trigger %", -80, 0, -20 - i*10, 5, key=f"et_{i}")
    weight = c2.slider(f"E{i+1} Cash %", 5, 100, 25, 5, key=f"ew_{i}")
    entry_tiers.append((thresh, weight))
entry_reset = st.sidebar.slider("Entry Reset Threshold (%)", -10, 20, 0, 1)

# --- TIME DELAY ---
st.sidebar.markdown("---")
st.sidebar.markdown("## ⏳ EXIT TIME-DELAY")
st.sidebar.caption("Min holding time in TQQQ before profit-booking to cash")
delay_unit = st.sidebar.radio("Delay Unit", ["Trading Days", "Weeks"], index=0, horizontal=True)
delay_value = st.sidebar.slider(f"Hold Delay ({delay_unit})", 0, 260, 20, 1)
hold_delay_days = delay_value * 5 if delay_unit == "Weeks" else delay_value

# --- EXIT ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 📈 EXIT TIERS (TQQQ → CASH)")
st.sidebar.caption("TQQQ surges X% above EMA → book Y% of TQQQ to CASH. Resets when dist < Z%.")
n_exit_tiers = st.sidebar.number_input("Number of Exit Tiers", 1, 8, 3, 1)
exit_tiers = []
for i in range(int(n_exit_tiers)):
    c1, c2, c3 = st.sidebar.columns(3)
    thresh = c1.slider(f"X{i+1} Trigger", 10, 150, 30 + i*15, 5, key=f"xt_{i}")
    weight = c2.slider(f"X{i+1} to Cash %", 5, 100, 50 if i == 0 else 100, 5, key=f"xw_{i}")
    reset_at = c3.slider(f"X{i+1} Reset", -10, 100, max(0, 30 + i*15 - 10), 5, key=f"xr_{i}")
    exit_tiers.append((thresh, weight, reset_at))

st.sidebar.markdown("---")
run = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")


# =============================================================================
# MAIN
# =============================================================================
st.title("💵 QQQ/TQQQ/CASH Phased Rotation (Profit-to-Cash)")

entry_str = " | ".join([f"{t:.0f}%→{w:.0f}%" for t, w in entry_tiers])
exit_str = " | ".join([f"+{t:.0f}%→cash {w:.0f}%(reset@{r:.0f}%)" for t, w, r in exit_tiers])
st.caption(
    f"DCA ${weekly_dca}/wk → CASH ({cash_apy}% APY) | Hold-delay: {delay_value} {delay_unit.lower()} | "
    f"Entry: [{entry_str}] | Exit→Cash: [{exit_str}]"
)

if run:
    with st.spinner("📡 Fetching QQQ & TQQQ data..."):
        data = fetch_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if data.empty:
        st.error("❌ No data. TQQQ starts Feb 2010.")
    else:
        with st.spinner("⚙️ Running cash-rotation simulation..."):
            results = run_backtest(
                data, ema_period, weekly_dca,
                entry_tiers, entry_reset, exit_tiers, hold_delay_days, cash_apy,
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
            r3[2].metric("Exit→Cash", m['exits'])
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

            st.subheader("🏦 Allocation ($): Cash vs TQQQ")
            a = results[["Date","Cash ($)","TQQQ Value ($)"]].copy()
            a["Date"] = pd.to_datetime(a["Date"])
            st.area_chart(a.set_index("Date"), use_container_width=True)

            st.subheader("📊 Allocation %")
            p = results[["Date","Cash %","TQQQ %"]].copy()
            p["Date"] = pd.to_datetime(p["Date"])
            st.area_chart(p.set_index("Date"), use_container_width=True)

            st.subheader("📋 Rotation Events")
            ev = results[results["Action"].str.contains("ENTRY|EXIT|RESET")]
            st.dataframe(
                ev[["Date","Action","EMA Dist %","Days Held","Transfer ($)","Portfolio ($)","Cash %","TQQQ %"]],
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
    st.info("👈 Configure parameters and click **Run Backtest**")
    st.markdown("""
    ### 💵 Profit-to-Cash Rotation Strategy

    | State | Description |
    |-------|-------------|
    | **CASH** | Baseline dry powder — earns configurable APY (T-bill yield) |
    | **TQQQ** | Leveraged position built by deploying cash during drawdowns |

    | Direction | Trigger | Action | Gate/Reset |
    |-----------|---------|--------|-----------|
    | ⬇️ ENTRY | TQQQ drops below EMA tiers | Deploy % of **CASH** → TQQQ | Resets above EMA |
    | ⬆️ EXIT | TQQQ surges above EMA tiers | Book % of TQQQ → **CASH** | Only after hold delay + per-tier reset |
    | 💰 DCA | Weekly | Add cash to dry powder | — |

    ### Key Difference from Previous Version
    - **Exits go to CASH**, not QQQ — you hold dry powder after profit-booking
    - Cash earns a **configurable APY** while waiting for the next drawdown
    - Entries deploy **from cash** (buying the dip with accumulated dry powder)
    - This creates a cleaner "buy low, sell high, wait in cash" cycle

    ### Example Cycle
    1. DCA accumulates cash weekly
    2. TQQQ crashes -20%, -30% → deploy 25% + 25% of cash into TQQQ
    3. Hold at least 20 trading days (delay gate)
    4. TQQQ recovers +30% → book 50% back to cash
    5. TQQQ hits +50% → book remaining 100% to cash
    6. Cash sits earning 4.5% APY until the next crash → repeat
    """)

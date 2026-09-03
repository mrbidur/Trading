"""
CASH ↔ TQQQ Phased Rotation with Tax, Fees & Delayed QQQ Re-deployment — Streamlit App
=======================================================================================
Three-asset engine: CASH (flat, no yield) ↔ TQQQ (leveraged) ↔ QQQ (baseline).

FLOW:
  - Weekly DCA accumulates into CASH (flat — NO interest / treasury yield)
  - ENTRY:  When TQQQ drops below EMA tiers → deploy CASH into TQQQ
  - EXIT:   When TQQQ surges above EMA tiers → sell TQQQ slice into CASH.
            Capital-gains tax is deducted on the PROFIT portion of that sale.
  - HOLD:   Booked cash waits a configurable period, then is deployed into QQQ
            (delayed re-establishment of a baseline position).
  - FEES:   Brokerage commission (flat $ and/or %) charged on every trade.

Run: streamlit run streamlit_switching_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime
from typing import List, Tuple
from collections import deque
import io

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="CASH/TQQQ/QQQ Rotation Backtest",
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
# BACKTEST ENGINE — CASH ↔ TQQQ ↔ QQQ with Tax, Fees, Delayed QQQ deployment
# =============================================================================
def run_backtest(
    data: pd.DataFrame,
    ema_period: int,
    weekly_dca: float,
    entry_tiers: List[Tuple[float, float]],       # (threshold_pct, weight_pct of CASH)
    entry_reset_threshold: float,
    exit_tiers: List[Tuple[float, float, float]], # (threshold_pct, sell_weight_pct, reset_drop_to_pct)
    hold_delay_days: int,                         # min days in TQQQ before an exit may fire
    cgt_pct: float,                               # capital-gains tax % on realized profit
    cash_hold_days: int,                          # days to hold booked cash before buying QQQ
    fee_flat: float,                              # flat $ fee per trade
    fee_pct: float,                               # % commission of trade notional
) -> pd.DataFrame:
    """
    Three-state rotation with tax + delayed QQQ re-deployment.

    WEEKLY DCA:
      Every new week → add $weekly_dca to CASH balance. Cash is FLAT (no yield).

    ENTRY (CASH → TQQQ):
      When TQQQ dist crosses below each entry tier, deploy that tier's
      weight% of CURRENT CASH into TQQQ (net of fees). Fires once per cycle.
      Resets when dist > entry_reset_threshold. Adds to TQQQ cost basis.

    EXIT / PROFIT-BOOKING (TQQQ → CASH, taxed):
      When TQQQ dist surges above each exit tier (AND hold delay elapsed),
      sell weight% of TQQQ shares. Compute realized profit vs. avg cost basis;
      deduct cgt_pct of the profit as capital-gains tax. Net proceeds go to CASH
      and are tagged with a release date = current + cash_hold_days.

    DELAYED QQQ RE-DEPLOYMENT (CASH → QQQ):
      Each booked-cash bucket that has aged >= cash_hold_days is deployed into
      QQQ (net of fees), establishing a baseline position.

    TIME DELAY (entry side):
      Exits blocked until hold_delay_days trading days pass after a fresh
      TQQQ entry (from zero).
    """
    data = data.copy()
    data["ema"] = data["tqqq"].ewm(span=ema_period, adjust=False).mean()
    data["dist"] = ((data["tqqq"] - data["ema"]) / data["ema"]) * 100.0
    data = data.dropna()

    if data.empty:
        return pd.DataFrame()

    def buy_shares_net(cash_budget: float, price: float) -> Tuple[float, float, float]:
        """
        Deploy `cash_budget` cash into an asset at `price`, net of fees.
        Fee = fee_flat + notional * fee_pct/100, taken out of the budget so that
        (notional + fee) == cash_budget exactly. Returns (shares, notional, fee).
        """
        if cash_budget <= 0 or price <= 0:
            return 0.0, 0.0, 0.0
        # notional*(1+pct) + flat = budget  ->  notional = (budget - flat)/(1+pct)
        notional = (cash_budget - fee_flat) / (1.0 + fee_pct / 100.0)
        if notional <= 0:
            return 0.0, 0.0, 0.0
        fee = fee_flat + notional * fee_pct / 100.0
        return notional / price, notional, fee

    def sell_fee(notional: float) -> float:
        if notional <= 0:
            return 0.0
        return fee_flat + notional * fee_pct / 100.0

    # --- State ---
    cash = 0.0                 # flat cash, no yield
    tqqq_shares = 0.0
    tqqq_cost_basis = 0.0      # total $ cost of current TQQQ shares (for tax)
    qqq_shares = 0.0           # baseline position built from delayed re-deployment
    total_invested = 0.0
    total_fees = 0.0
    total_tax = 0.0
    last_dca_week = None

    # Booked-cash buckets waiting to be deployed into QQQ: list of (release_day_index, amount)
    pending_qqq = deque()
    bar_index = 0

    n_entry = len(entry_tiers)
    entry_triggered = [False] * n_entry
    n_exit = len(exit_tiers)
    exit_triggered = [False] * n_exit

    # Benchmark: pure QQQ DCA (buy QQQ every week net of fees, never sell)
    bench_shares = 0.0

    days_since_entry = -1

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
        fee_today = 0.0
        tax_today = 0.0

        # --- Advance time-delay counter ---
        if tqqq_shares > 0:
            days_since_entry = 0 if days_since_entry < 0 else days_since_entry + 1
        else:
            days_since_entry = -1

        # --- WEEKLY DCA into CASH (flat, no yield) ---
        if is_dca:
            last_dca_week = current_week
            deposit = weekly_dca
            cash += deposit
            total_invested += deposit
            # Benchmark buys QQQ weekly, net of the same friction
            b_sh, _, _ = buy_shares_net(deposit, qp)
            bench_shares += b_sh
            action = "DCA"

        # --- DELAYED QQQ RE-DEPLOYMENT: release aged cash buckets into QQQ ---
        deployed_to_qqq = 0.0
        while pending_qqq and (bar_index - pending_qqq[0][0]) >= cash_hold_days:
            _, amount = pending_qqq.popleft()
            amount = min(amount, cash)  # guard
            if amount <= 0:
                continue
            sh, notional, fee = buy_shares_net(amount, qp)
            if sh > 0:
                qqq_shares += sh
                cash -= amount
                total_fees += fee
                fee_today += fee
                deployed_to_qqq += notional
        if deployed_to_qqq > 0:
            lbl = f"CASH→QQQ deploy (${deployed_to_qqq:,.0f})"
            action = lbl if action == "HOLD" else f"{action} + {lbl}"
            transfer = deployed_to_qqq

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

        # --- PHASED EXIT: TQQQ → CASH (taxed, gated by time-delay) ---
        exit_allowed = (days_since_entry >= hold_delay_days) and (tqqq_shares > 0)
        if exit_allowed:
            for e_idx, (threshold, weight, _) in enumerate(exit_tiers):
                if dist >= threshold and not exit_triggered[e_idx] and tqqq_shares > 0:
                    exit_triggered[e_idx] = True
                    shares_to_sell = tqqq_shares * (weight / 100.0)
                    gross = shares_to_sell * tp

                    # Cost basis of the shares being sold (weighted average)
                    cost_of_sold = tqqq_cost_basis * (shares_to_sell / tqqq_shares) if tqqq_shares > 0 else 0.0
                    fee = sell_fee(gross)
                    realized_profit = gross - cost_of_sold - fee   # profit net of the sell fee
                    tax = max(realized_profit, 0.0) * (cgt_pct / 100.0)

                    net_to_cash = gross - fee - tax
                    cash += max(net_to_cash, 0.0)

                    # Reduce holdings & basis proportionally
                    tqqq_shares -= shares_to_sell
                    tqqq_cost_basis -= cost_of_sold

                    total_fees += fee
                    total_tax += tax
                    fee_today += fee
                    tax_today += tax
                    transfer = gross

                    # Tag the net proceeds to be deployed into QQQ after the holding period
                    if net_to_cash > 0:
                        pending_qqq.append((bar_index, max(net_to_cash, 0.0)))

                    tier_label = f"EXIT→CASH T{e_idx+1} (+{threshold:.0f}%, {weight:.0f}%, tax ${tax:,.0f})"
                    action = tier_label if action == "HOLD" else f"{action} + {tier_label}"
                    break

        # --- PHASED ENTRY: CASH → TQQQ ---
        if dist <= entry_reset_threshold:
            for t_idx, (threshold, weight) in enumerate(entry_tiers):
                if dist <= threshold and not entry_triggered[t_idx] and cash > 0:
                    entry_triggered[t_idx] = True
                    was_flat = (tqqq_shares == 0)
                    deploy_cash = cash * (weight / 100.0)
                    sh, notional, fee = buy_shares_net(deploy_cash, tp)
                    if sh > 0:
                        tqqq_shares += sh
                        tqqq_cost_basis += notional      # basis excludes fee (fee is a cost, not basis)
                        cash -= deploy_cash              # full budget leaves cash (notional + fee)
                        total_fees += fee
                        fee_today += fee
                        transfer = deploy_cash
                        if was_flat:
                            days_since_entry = 0
                        tier_label = f"ENTRY T{t_idx+1} ({threshold:.0f}%, {weight:.0f}%)"
                        action = tier_label if action in ["HOLD", "DCA"] else f"{action} + {tier_label}"
                    break

        # --- VALUATION ---
        tqqq_val = tqqq_shares * tp
        qqq_val = qqq_shares * qp
        total = cash + tqqq_val + qqq_val
        bench_val = bench_shares * qp
        cash_pct = (cash / total * 100) if total > 0 else 100
        tqqq_pct = (tqqq_val / total * 100) if total > 0 else 0
        qqq_pct = (qqq_val / total * 100) if total > 0 else 0
        pending_cash = sum(a for _, a in pending_qqq)

        rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "QQQ Price": round(qp, 4),
            "TQQQ Price": round(tp, 4),
            "TQQQ EMA": round(ema_val, 4),
            "EMA Dist %": round(dist, 2),
            "Cash ($)": round(cash, 2),
            "Pending→QQQ ($)": round(pending_cash, 2),
            "TQQQ Shares": round(tqqq_shares, 4),
            "TQQQ Value ($)": round(tqqq_val, 2),
            "TQQQ Cost Basis ($)": round(tqqq_cost_basis, 2),
            "QQQ Shares": round(qqq_shares, 4),
            "QQQ Value ($)": round(qqq_val, 2),
            "Portfolio ($)": round(total, 2),
            "Benchmark ($)": round(bench_val, 2),
            "Invested ($)": round(total_invested, 2),
            "Cash %": round(cash_pct, 1),
            "TQQQ %": round(tqqq_pct, 1),
            "QQQ %": round(qqq_pct, 1),
            "Days Held": days_since_entry if days_since_entry >= 0 else 0,
            "Entry Tiers": sum(entry_triggered),
            "Exit Tiers": sum(exit_triggered),
            "Action": action,
            "Deposit ($)": round(deposit, 2),
            "Transfer ($)": round(transfer, 2),
            "Fee ($)": round(fee_today, 4),
            "Tax ($)": round(tax_today, 2),
            "Cumulative Fees ($)": round(total_fees, 2),
            "Cumulative Tax ($)": round(total_tax, 2),
        })
        bar_index += 1

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
    qqq_deploys = df["Action"].str.contains("CASH→QQQ").sum()

    total_fees = df["Cumulative Fees ($)"].values[-1] if "Cumulative Fees ($)" in df.columns else 0.0
    total_tax = df["Cumulative Tax ($)"].values[-1] if "Cumulative Tax ($)" in df.columns else 0.0
    n_trades = int((df["Fee ($)"] > 0).sum()) if "Fee ($)" in df.columns else 0
    fee_drag_pct = (total_fees / inv * 100) if inv > 0 else 0.0
    tax_drag_pct = (total_tax / inv * 100) if inv > 0 else 0.0

    return dict(
        invested=inv, final_s=final_s, final_b=final_b,
        roi_s=roi_s, roi_b=roi_b, cagr_s=cagr_s, cagr_b=cagr_b,
        mdd_s=mdd(v), mdd_b=mdd(b), years=yrs,
        alpha=final_s - final_b, alpha_pct=(final_s/final_b-1)*100 if final_b>0 else 0,
        entries=entries, exits=exits, resets=resets, qqq_deploys=qqq_deploys,
        total_fees=total_fees, total_tax=total_tax, n_trades=n_trades,
        fee_drag_pct=fee_drag_pct, tax_drag_pct=tax_drag_pct,
    )


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("💵 Rotation Controls")

st.sidebar.markdown("## 💰 Weekly DCA")
weekly_dca = st.sidebar.slider("Weekly Deposit ($)", 50, 2000, 200, 50,
    help="Added to CASH each week. Cash is FLAT — no interest/yield.")

st.sidebar.markdown("## 📅 Timeline")
col_s, col_e = st.sidebar.columns(2)
start_date = col_s.date_input("Start", value=datetime(2010, 2, 11))
end_date = col_e.date_input("End", value=date.today())

st.sidebar.markdown("## 📈 EMA")
ema_period = st.sidebar.slider("EMA Period (days)", 10, 500, 200, 10)

# --- TRANSACTION COSTS ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 💸 Brokerage / Commission")
st.sidebar.caption("Charged on every buy, sell or shift (DCA, entry, exit, QQQ deploy)")
fee_pct = st.sidebar.slider("Commission (% of trade value)", 0.0, 1.0, 0.0, 0.01,
    help="Percentage commission per trade, e.g. 0.05% or 0.1%")
fee_flat = st.sidebar.number_input("Flat Fee per Trade ($)", 0.0, 100.0, 0.0, 0.5,
    help="Fixed dollar fee charged on each trade")

# --- TAX ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 🧾 Capital Gains Tax")
cgt_pct = st.sidebar.slider("Capital Gains Tax (%)", 0.0, 50.0, 15.0, 1.0,
    help="Deducted from the PROFIT portion of each TQQQ→Cash sale")

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

# --- ENTRY TIME DELAY ---
st.sidebar.markdown("---")
st.sidebar.markdown("## ⏳ EXIT TIME-DELAY")
st.sidebar.caption("Min holding time in TQQQ before profit-booking to cash")
delay_unit = st.sidebar.radio("Delay Unit", ["Trading Days", "Weeks"], index=0, horizontal=True)
delay_value = st.sidebar.slider(f"Hold Delay ({delay_unit})", 0, 260, 20, 1)
hold_delay_days = delay_value * 5 if delay_unit == "Weeks" else delay_value

# --- EXIT ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 📈 EXIT TIERS (TQQQ → CASH, taxed)")
st.sidebar.caption("TQQQ surges X% above EMA → book Y% of TQQQ to CASH. Resets when dist < Z%.")
n_exit_tiers = st.sidebar.number_input("Number of Exit Tiers", 1, 8, 3, 1)
exit_tiers = []
for i in range(int(n_exit_tiers)):
    c1, c2, c3 = st.sidebar.columns(3)
    thresh = c1.slider(f"X{i+1} Trigger", 10, 150, 30 + i*15, 5, key=f"xt_{i}")
    weight = c2.slider(f"X{i+1} to Cash %", 5, 100, 50 if i == 0 else 100, 5, key=f"xw_{i}")
    reset_at = c3.slider(f"X{i+1} Reset", -10, 100, max(0, 30 + i*15 - 10), 5, key=f"xr_{i}")
    exit_tiers.append((thresh, weight, reset_at))

# --- CASH HOLDING BEFORE QQQ DEPLOYMENT ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 🏦 Post-Exit Cash → QQQ")
st.sidebar.caption("Hold booked cash this long, then deploy it into QQQ (baseline)")
qqq_unit = st.sidebar.radio("Hold Unit", ["Trading Days", "Weeks"], index=0, horizontal=True, key="qqq_unit")
qqq_hold_value = st.sidebar.slider(f"Cash Hold Before QQQ ({qqq_unit})", 0, 260, 0, 1, key="qqq_hold")
cash_hold_days = qqq_hold_value * 5 if qqq_unit == "Weeks" else qqq_hold_value

st.sidebar.markdown("---")
run = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")


# =============================================================================
# MAIN
# =============================================================================
st.title("💵 CASH/TQQQ/QQQ Rotation — Tax + Fees + Delayed QQQ Deploy")

entry_str = " | ".join([f"{t:.0f}%→{w:.0f}%" for t, w in entry_tiers])
exit_str = " | ".join([f"+{t:.0f}%→cash {w:.0f}%(reset@{r:.0f}%)" for t, w, r in exit_tiers])
fee_str = f"{fee_pct:.2f}% + ${fee_flat:.2f}/trade"
st.caption(
    f"DCA ${weekly_dca}/wk → CASH (flat) | Fees: {fee_str} | CGT: {cgt_pct:.0f}% | "
    f"Hold-delay: {delay_value} {delay_unit.lower()} | Cash→QQQ after {qqq_hold_value} {qqq_unit.lower()} | "
    f"Entry: [{entry_str}] | Exit→Cash: [{exit_str}]"
)

if run:
    with st.spinner("📡 Fetching QQQ & TQQQ data..."):
        data = fetch_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if data.empty:
        st.error("❌ No data. TQQQ starts Feb 2010.")
    else:
        with st.spinner("⚙️ Running rotation simulation (tax + fees)..."):
            results = run_backtest(
                data, ema_period, weekly_dca,
                entry_tiers, entry_reset, exit_tiers, hold_delay_days,
                cgt_pct, cash_hold_days, fee_flat, fee_pct,
            )

        if results.empty:
            st.error("❌ No results.")
        else:
            m = compute_metrics(results)

            st.markdown("---")
            st.subheader("📊 Performance (Net of Taxes & Fees)")
            r1 = st.columns(4)
            r1[0].metric("Invested", f"${m['invested']:,.0f}")
            r1[1].metric("Strategy (Net)", f"${m['final_s']:,.0f}", f"+{m['roi_s']:.1f}%")
            r1[2].metric("Benchmark (QQQ DCA)", f"${m['final_b']:,.0f}", f"+{m['roi_b']:.1f}%")
            r1[3].metric("Alpha", f"${m['alpha']:+,.0f}", f"{m['alpha_pct']:+.1f}%",
                         delta_color="normal" if m['alpha'] >= 0 else "inverse")
            r2 = st.columns(4)
            r2[0].metric("Net CAGR (Strategy)", f"{m['cagr_s']:.2f}%")
            r2[1].metric("CAGR (Benchmark)", f"{m['cagr_b']:.2f}%")
            r2[2].metric("MDD (Strategy)", f"{m['mdd_s']:.1f}%")
            r2[3].metric("MDD (Benchmark)", f"{m['mdd_b']:.1f}%")
            r3 = st.columns(4)
            r3[0].metric("💸 Total Fees", f"${m['total_fees']:,.2f}", f"-{m['fee_drag_pct']:.2f}%",
                         delta_color="inverse")
            r3[1].metric("🧾 Total Cap-Gains Tax", f"${m['total_tax']:,.2f}", f"-{m['tax_drag_pct']:.2f}%",
                         delta_color="inverse")
            r3[2].metric("Trades Charged", f"{m['n_trades']:,}")
            r3[3].metric("Duration", f"{m['years']:.1f} yrs")
            r4 = st.columns(4)
            r4[0].metric("Entry Switches", m['entries'])
            r4[1].metric("Exit→Cash", m['exits'])
            r4[2].metric("Cash→QQQ Deploys", m['qqq_deploys'])
            r4[3].metric("Resets", m['resets'])

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

            st.subheader("🏦 Allocation ($): Cash vs TQQQ vs QQQ")
            a = results[["Date","Cash ($)","TQQQ Value ($)","QQQ Value ($)"]].copy()
            a["Date"] = pd.to_datetime(a["Date"])
            st.area_chart(a.set_index("Date"), use_container_width=True)

            st.subheader("📊 Allocation %")
            p = results[["Date","Cash %","TQQQ %","QQQ %"]].copy()
            p["Date"] = pd.to_datetime(p["Date"])
            st.area_chart(p.set_index("Date"), use_container_width=True)

            st.subheader("🧾 Cumulative Fees & Taxes")
            f = results[["Date","Cumulative Fees ($)","Cumulative Tax ($)"]].copy()
            f["Date"] = pd.to_datetime(f["Date"])
            st.line_chart(f.set_index("Date"), use_container_width=True)

            st.subheader("📋 Transaction Events")
            ev = results[results["Action"].str.contains("ENTRY|EXIT|RESET|QQQ")]
            st.dataframe(
                ev[["Date","Action","EMA Dist %","Days Held","Transfer ($)","Fee ($)","Tax ($)",
                    "Portfolio ($)","Cash %","TQQQ %","QQQ %"]],
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
    ### 💵 CASH ↔ TQQQ ↔ QQQ Rotation (Tax + Fees + Delayed QQQ Deploy)

    | Asset | Role |
    |-------|------|
    | **CASH** | Dry powder — **flat, no interest/yield** |
    | **TQQQ** | Leveraged position built by deploying cash during drawdowns |
    | **QQQ** | Baseline, re-established from booked cash after a holding pause |

    | Direction | Trigger | Action | Costs / Gate |
    |-----------|---------|--------|--------------|
    | 💰 DCA | Weekly | Add cash to dry powder | fee on benchmark buy |
    | ⬇️ ENTRY | TQQQ below EMA tiers | Deploy % of **CASH** → TQQQ | commission |
    | ⬆️ EXIT | TQQQ above EMA tiers | Book % of TQQQ → **CASH** | **capital-gains tax on profit** + commission + hold-delay gate |
    | 🏦 REDEPLOY | Cash aged past hold period | Booked cash → **QQQ** | commission |

    ### What's new in this version
    - **Cash is flat** — no synthetic treasury/APY yield anymore
    - **Capital-gains tax**: the profit portion of every TQQQ sale is taxed at your
      chosen rate (cost-basis tracked via weighted average)
    - **Delayed QQQ re-deployment**: booked cash waits a configurable period, then
      buys QQQ — test whether pausing beats immediate re-entry
    - **Fees** (flat + %) charged on every trade; all metrics are **net of tax & fees**

    ### Example Cycle
    1. DCA accumulates flat cash weekly
    2. TQQQ crashes -20%, -30% → deploy 25% + 25% of cash into TQQQ (basis recorded)
    3. Hold at least the delay period
    4. TQQQ recovers +30% → sell 50%: pay commission + capital-gains tax on the profit,
       net proceeds land in cash
    5. That cash waits the hold period, then rotates into QQQ (baseline)
    6. Repeat on the next drawdown
    """)

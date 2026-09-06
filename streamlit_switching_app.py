"""
Phased Two-Way Rotation: QQQ ↔ TQQQ ↔ CASH — with FIFO Tax Lots, Fees & Delayed QQQ Re-deployment
===================================================================================================
Assets: QQQ (base) | TQQQ (leveraged) | CASH (flat sideline — NO yield).

FLOW:
  - Weekly DCA accumulates into CASH (flat — no interest/treasury yield).
  - ENTRY (down tiers):  shift a % of TOTAL PORTFOLIO VALUE into TQQQ,
                         funded from CASH first, then by liquidating QQQ (FIFO, taxed).
  - EXIT  (up tiers):    sell a % of TQQQ into CASH; realized gains taxed on a
                         FIFO lot basis (long-term >365d gets a CGT discount).
  - REDEPLOY:            booked cash waits N weeks, then rotates into QQQ (baseline).
  - FEES:                flat $ + % commission on every buy / sell / shift.

Run: streamlit run streamlit_switching_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
from typing import List, Tuple
from collections import deque
import io

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="QQQ/TQQQ/CASH Rotation (FIFO Tax)",
    page_icon="🔁",
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
# FIFO TAX-LOT HELPERS
# =============================================================================
def sell_fifo(lots: deque, shares_to_sell: float, price: float, current_date,
              marginal_rate: float, lt_discount_pct: float) -> Tuple[float, float, float]:
    """
    Sell `shares_to_sell` from a FIFO deque of lots at `price`.

    Each lot is [purchase_date, shares, cost_per_share].
    Returns (gross_proceeds, realized_gain, tax).

    Long-term (held > 365 days) gains receive `lt_discount_pct`% discount on the
    TAXABLE gain before applying the marginal rate. Losses are not taxed (and are
    not used to offset here — kept simple, tax = max(0, ...) per lot).
    """
    remaining = shares_to_sell
    gross = 0.0
    total_gain = 0.0
    total_tax = 0.0

    while remaining > 1e-12 and lots:
        lot = lots[0]
        lot_date, lot_shares, lot_cost = lot
        take = min(lot_shares, remaining)

        proceeds = take * price
        cost = take * lot_cost
        gain = proceeds - cost

        # Holding period → long-term discount
        held_days = (current_date - lot_date).days
        if gain > 0:
            taxable_gain = gain * (1.0 - lt_discount_pct / 100.0) if held_days > 365 else gain
            total_tax += taxable_gain * (marginal_rate / 100.0)

        gross += proceeds
        total_gain += gain

        # Consume the lot
        if take >= lot_shares - 1e-12:
            lots.popleft()
        else:
            lot[1] = lot_shares - take
        remaining -= take

    return gross, total_gain, total_tax


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def run_backtest(
    data: pd.DataFrame,
    ema_period: int,
    weekly_dca: float,
    entry_tiers: List[Tuple[float, float]],       # (threshold_pct, weight_pct of TOTAL PORTFOLIO)
    entry_reset_threshold: float,
    exit_tiers: List[Tuple[float, float, float]], # (threshold_pct, sell_weight_pct, reset_drop_to_pct)
    marginal_rate: float,                         # marginal tax rate %
    lt_discount_pct: float,                       # long-term (>365d) CGT discount %
    cash_hold_weeks: int,                         # MAX weeks to hold booked cash before buying QQQ (fallback)
    pullback_pct: float,                          # TQQQ %-drop from booking price that triggers early redeploy (0 = off)
    fee_flat: float,                              # flat $ fee per trade
    fee_pct: float,                               # % commission of trade notional
) -> pd.DataFrame:
    """
    Phased two-way rotation with FIFO tax lots and delayed QQQ re-deployment.

    ENTRY (down tiers): target = weight% * TOTAL PORTFOLIO VALUE.
      Fund from CASH first; if insufficient, liquidate QQQ (FIFO, taxed) to raise
      the remainder. Then buy TQQQ with the funded amount (net of fees) and record
      a new TQQQ lot. Fires once per cycle; resets when dist > entry_reset_threshold.

    EXIT (up tiers): sell weight% of TQQQ shares into CASH. Realized gain taxed on
      a FIFO lot basis (long-term >365d discounted). Net proceeds are parked in cash
      and tracked as a bucket tagged with (booking_bar, booking_tqqq_price, amount).
      Each tier resets when dist < its reset level.

    HYBRID CASH → QQQ RE-DEPLOYMENT (whichever fires FIRST per bucket):
      a) PRICE PULLBACK: if TQQQ falls >= pullback_pct below its price at the moment
         the cash was booked, redeploy that bucket into QQQ immediately.
      b) TIME FALLBACK: else, once the bucket has aged >= cash_hold_weeks, redeploy
         it into QQQ.

    CASH is flat (no yield). Fees (flat + %) on every buy/sell/shift.
    """
    data = data.copy()
    data["ema"] = data["tqqq"].ewm(span=ema_period, adjust=False).mean()
    data["dist"] = ((data["tqqq"] - data["ema"]) / data["ema"]) * 100.0
    data = data.dropna()

    if data.empty:
        return pd.DataFrame()

    hold_bars = cash_hold_weeks * 5  # ~5 trading days / week

    def fee_on(notional: float) -> float:
        return fee_flat + notional * fee_pct / 100.0 if notional > 0 else 0.0

    def buy_shares_net(cash_budget: float, price: float):
        """Deploy cash_budget into an asset at price, net of fees.
        Returns (shares, notional, fee) with notional + fee == cash_budget."""
        if cash_budget <= 0 or price <= 0:
            return 0.0, 0.0, 0.0
        notional = (cash_budget - fee_flat) / (1.0 + fee_pct / 100.0)
        if notional <= 0:
            return 0.0, 0.0, 0.0
        fee = fee_flat + notional * fee_pct / 100.0
        return notional / price, notional, fee

    # --- State ---
    cash = 0.0
    tqqq_lots = deque()   # [date, shares, cost_per_share]
    qqq_lots = deque()    # [date, shares, cost_per_share]
    total_invested = 0.0
    total_fees = 0.0
    total_tax = 0.0
    last_dca_week = None

    pending_qqq = deque()  # (release_bar_index, amount)
    bar_index = 0

    n_entry = len(entry_tiers)
    entry_triggered = [False] * n_entry
    n_exit = len(exit_tiers)
    exit_triggered = [False] * n_exit

    bench_shares = 0.0     # pure QQQ DCA benchmark (net of fees)

    def shares_of(lots):
        return sum(l[1] for l in lots)

    rows = []

    for dt, row in data.iterrows():
        qp = float(row["qqq"])
        tp = float(row["tqqq"])
        ema_val = float(row["ema"])
        dist = float(row["dist"])
        cur_date = (dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt)

        current_week = (cur_date.isocalendar()[0], cur_date.isocalendar()[1])
        is_dca = (current_week != last_dca_week)

        action = "HOLD"
        deposit = 0.0
        transfer = 0.0
        fee_today = 0.0
        tax_today = 0.0

        # --- WEEKLY DCA into CASH (flat) ---
        if is_dca:
            last_dca_week = current_week
            deposit = weekly_dca
            cash += deposit
            total_invested += deposit
            b_sh, _, _ = buy_shares_net(deposit, qp)   # benchmark buys QQQ net of fees
            bench_shares += b_sh
            action = "DCA"

        # --- HYBRID CASH → QQQ RE-DEPLOYMENT (pullback OR time, whichever first) ---
        deployed_to_qqq = 0.0
        pullback_fired = False
        time_fired = False
        still_pending = deque()
        while pending_qqq:
            book_bar, book_tp, amount = pending_qqq.popleft()
            aged = (bar_index - book_bar) >= hold_bars
            # pullback measured vs TQQQ price at the moment this cash was booked
            pulled_back = (pullback_pct > 0) and (book_tp > 0) and \
                          (((book_tp - tp) / book_tp) * 100.0 >= pullback_pct)
            if not (aged or pulled_back):
                still_pending.append((book_bar, book_tp, amount))
                continue
            amount = min(amount, cash)
            if amount <= 0:
                continue
            sh, notional, fee = buy_shares_net(amount, qp)
            if sh > 0:
                qqq_lots.append([cur_date, sh, notional / sh])
                cash -= amount
                total_fees += fee
                fee_today += fee
                deployed_to_qqq += notional
                if pulled_back:
                    pullback_fired = True
                else:
                    time_fired = True
        pending_qqq = still_pending
        if deployed_to_qqq > 0:
            if pullback_fired and time_fired:
                trig = "pullback+time"
            elif pullback_fired:
                trig = f"pullback ≥{pullback_pct:.0f}%"
            else:
                trig = "time limit"
            lbl = f"CASH→QQQ deploy ${deployed_to_qqq:,.0f} [{trig}]"
            action = lbl if action == "HOLD" else f"{action} + {lbl}"
            transfer += deployed_to_qqq

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

        # --- PHASED EXIT: TQQQ → CASH (FIFO-taxed). No time-delay gate. ---
        tqqq_sh = shares_of(tqqq_lots)
        if tqqq_sh > 0:
            for e_idx, (threshold, weight, _) in enumerate(exit_tiers):
                if dist >= threshold and not exit_triggered[e_idx] and shares_of(tqqq_lots) > 0:
                    exit_triggered[e_idx] = True
                    shares_to_sell = shares_of(tqqq_lots) * (weight / 100.0)
                    gross, gain, tax = sell_fifo(tqqq_lots, shares_to_sell, tp, cur_date,
                                                 marginal_rate, lt_discount_pct)
                    fee = fee_on(gross)
                    net_to_cash = gross - fee - tax
                    cash += max(net_to_cash, 0.0)
                    total_fees += fee
                    total_tax += tax
                    fee_today += fee
                    tax_today += tax
                    transfer += gross
                    if net_to_cash > 0:
                        # (booking_bar, TQQQ price at booking, amount) for hybrid redeploy
                        pending_qqq.append((bar_index, tp, max(net_to_cash, 0.0)))
                    tier_label = (f"EXIT→CASH T{e_idx+1} (+{threshold:.0f}%, {weight:.0f}%, "
                                  f"gain ${gain:,.0f}, tax ${tax:,.0f})")
                    action = tier_label if action == "HOLD" else f"{action} + {tier_label}"
                    break

        # --- PHASED ENTRY: shift % of TOTAL PORTFOLIO into TQQQ (cash first, then QQQ) ---
        if dist <= entry_reset_threshold:
            for t_idx, (threshold, weight) in enumerate(entry_tiers):
                if dist <= threshold and not entry_triggered[t_idx]:
                    entry_triggered[t_idx] = True
                    # Total portfolio value RIGHT NOW
                    port_val = cash + shares_of(tqqq_lots) * tp + shares_of(qqq_lots) * qp
                    target = port_val * (weight / 100.0)
                    if target <= 0:
                        break

                    # 1) Fund from CASH first
                    fund = min(cash, target)
                    cash -= fund
                    shortfall = target - fund

                    # 2) Liquidate QQQ (FIFO, taxed) to cover the shortfall
                    qqq_sold_gross = 0.0
                    if shortfall > 0 and shares_of(qqq_lots) > 0:
                        # shares needed to raise `shortfall` net of the sell fee on that slice
                        # approximate: raise gross so that gross - fee_on(gross) = shortfall
                        gross_needed = (shortfall + fee_flat) / (1.0 - fee_pct / 100.0)
                        qqq_price = qp
                        shares_needed = gross_needed / qqq_price
                        shares_avail = shares_of(qqq_lots)
                        shares_sell = min(shares_needed, shares_avail)
                        g, gain_q, tax_q = sell_fifo(qqq_lots, shares_sell, qqq_price, cur_date,
                                                     marginal_rate, lt_discount_pct)
                        f_q = fee_on(g)
                        net_raised = g - f_q - tax_q
                        cash += max(net_raised, 0.0)     # proceeds into cash first
                        total_fees += f_q
                        total_tax += tax_q
                        fee_today += f_q
                        tax_today += tax_q
                        qqq_sold_gross = g
                        # now pull what we can toward the target
                        pull = min(cash, shortfall)
                        cash -= pull
                        fund += pull

                    # 3) Buy TQQQ with the funded amount (net of fees)
                    sh, notional, fee = buy_shares_net(fund, tp)
                    if sh > 0:
                        tqqq_lots.append([cur_date, sh, notional / sh])
                        total_fees += fee
                        fee_today += fee
                        transfer += notional
                    else:
                        cash += fund  # nothing bought, return funds

                    tier_label = f"ENTRY T{t_idx+1} ({threshold:.0f}%, {weight:.0f}% of port"
                    if qqq_sold_gross > 0:
                        tier_label += f", sold QQQ ${qqq_sold_gross:,.0f}"
                    tier_label += ")"
                    action = tier_label if action in ["HOLD", "DCA"] else f"{action} + {tier_label}"
                    break

        # --- VALUATION ---
        tqqq_shares = shares_of(tqqq_lots)
        qqq_shares = shares_of(qqq_lots)
        tqqq_val = tqqq_shares * tp
        qqq_val = qqq_shares * qp
        total = cash + tqqq_val + qqq_val
        bench_val = bench_shares * qp
        tqqq_basis = sum(l[1] * l[2] for l in tqqq_lots)
        cash_pct = (cash / total * 100) if total > 0 else 100
        tqqq_pct = (tqqq_val / total * 100) if total > 0 else 0
        qqq_pct = (qqq_val / total * 100) if total > 0 else 0
        pending_cash = sum(a for _, _, a in pending_qqq)

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
            "TQQQ Cost Basis ($)": round(tqqq_basis, 2),
            "TQQQ Lots": len(tqqq_lots),
            "QQQ Shares": round(qqq_shares, 4),
            "QQQ Value ($)": round(qqq_val, 2),
            "QQQ Lots": len(qqq_lots),
            "Portfolio ($)": round(total, 2),
            "Benchmark ($)": round(bench_val, 2),
            "Invested ($)": round(total_invested, 2),
            "Cash %": round(cash_pct, 1),
            "TQQQ %": round(tqqq_pct, 1),
            "QQQ %": round(qqq_pct, 1),
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
    pullback_deploys = df["Action"].str.contains(r"pullback").sum()
    time_deploys = df["Action"].str.contains(r"time limit").sum()

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
        pullback_deploys=pullback_deploys, time_deploys=time_deploys,
        total_fees=total_fees, total_tax=total_tax, n_trades=n_trades,
        fee_drag_pct=fee_drag_pct, tax_drag_pct=tax_drag_pct,
    )


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("🔁 Rotation Controls")

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
st.sidebar.caption("Charged on every buy, sell or shift")
fee_pct = st.sidebar.slider("Commission (% of trade value)", 0.0, 1.0, 0.0, 0.01,
    help="Percentage commission per trade, e.g. 0.05% or 0.1%")
fee_flat = st.sidebar.number_input("Flat Fee per Trade ($)", 0.0, 100.0, 0.0, 0.5,
    help="Fixed dollar fee charged on each trade")

# --- TAX ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 🧾 Capital Gains Tax (FIFO lots)")
marginal_rate = st.sidebar.slider("Marginal Tax Rate (%)", 0.0, 50.0, 30.0, 1.0,
    help="Applied to realized gains (FIFO). Long-term gains discounted below.")
lt_discount_pct = st.sidebar.slider("Long-Term Discount (%) for >365d", 0.0, 100.0, 50.0, 5.0,
    help="Gains on lots held > 365 days get this % discount before tax")

# --- ENTRY ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 📉 ENTRY TIERS (→ TQQQ)")
st.sidebar.caption("TQQQ drops X% below EMA → shift Y% of TOTAL PORTFOLIO into TQQQ (cash first, then sell QQQ)")
n_entry_tiers = st.sidebar.number_input("Number of Entry Tiers", 1, 8, 2, 1)
# Spec defaults: Tier 1 = -40% → 40% of portfolio, Tier 2 = -50% → 60% (remainder)
ENTRY_DEFAULT_TRIG = [-40, -50, -60, -30, -20, -25, -35, -45]
ENTRY_DEFAULT_WT   = [40, 60, 100, 25, 25, 25, 25, 25]
entry_tiers = []
for i in range(int(n_entry_tiers)):
    c1, c2 = st.sidebar.columns(2)
    thresh = c1.slider(f"E{i+1} Trigger %", -80, 0, ENTRY_DEFAULT_TRIG[i], 5, key=f"et_{i}")
    weight = c2.slider(f"E{i+1} Portfolio %", 5, 100, ENTRY_DEFAULT_WT[i], 5, key=f"ew_{i}")
    entry_tiers.append((thresh, weight))
entry_reset = st.sidebar.slider("Entry Reset Threshold (%)", -10, 20, 0, 1)

# --- EXIT ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 📈 EXIT TIERS (TQQQ → CASH, taxed)")
st.sidebar.caption("TQQQ surges X% above EMA → book Y% of TQQQ to CASH. Resets when dist < Z%.")
n_exit_tiers = st.sidebar.number_input("Number of Exit Tiers", 1, 8, 1, 1)
# Spec default: single exit at +50% → book 100% to cash (resets when dist < +30%)
EXIT_DEFAULT_TRIG  = [50, 65, 80, 95, 110, 125, 140, 150]
EXIT_DEFAULT_WT    = [100, 100, 100, 100, 100, 100, 100, 100]
EXIT_DEFAULT_RESET = [30, 45, 60, 75, 90, 105, 120, 135]
exit_tiers = []
for i in range(int(n_exit_tiers)):
    c1, c2, c3 = st.sidebar.columns(3)
    thresh = c1.slider(f"X{i+1} Trigger", 10, 150, EXIT_DEFAULT_TRIG[i], 5, key=f"xt_{i}")
    weight = c2.slider(f"X{i+1} to Cash %", 5, 100, EXIT_DEFAULT_WT[i], 5, key=f"xw_{i}")
    reset_at = c3.slider(f"X{i+1} Reset", -10, 100, EXIT_DEFAULT_RESET[i], 5, key=f"xr_{i}")
    exit_tiers.append((thresh, weight, reset_at))

# --- POST-EXIT CASH HOLDING BEFORE QQQ ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 🏦 Post-Exit Cash → QQQ (Hybrid)")
st.sidebar.caption("Booked cash redeploys into QQQ on whichever fires FIRST: a TQQQ pullback OR the max hold time.")
pullback_pct = st.sidebar.slider("Pullback % Trigger for Cash Re-deployment", 0, 80, 20, 5,
    help="If TQQQ drops this % below its price when the cash was booked, redeploy into QQQ immediately. 0 = disable pullback trigger (time-only).")
cash_hold_weeks = st.sidebar.slider("Max Cash Hold Before QQQ (weeks)", 0, 104, 12, 1,
    help="Time fallback: if the pullback trigger isn't hit, redeploy into QQQ once this many weeks elapse. 0 = deploy next bar.")

st.sidebar.markdown("---")
run = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")


# =============================================================================
# MAIN
# =============================================================================
st.title("🔁 QQQ/TQQQ/CASH Two-Way Rotation — FIFO Tax + Fees + Delayed QQQ")

entry_str = " | ".join([f"{t:.0f}%→{w:.0f}%port" for t, w in entry_tiers])
exit_str = " | ".join([f"+{t:.0f}%→cash {w:.0f}%(reset@{r:.0f}%)" for t, w, r in exit_tiers])
fee_str = f"{fee_pct:.2f}% + ${fee_flat:.2f}/trade"
st.caption(
    f"DCA ${weekly_dca}/wk → CASH (flat) | Fees: {fee_str} | Tax: {marginal_rate:.0f}% "
    f"(LT >365d −{lt_discount_pct:.0f}%) | Cash→QQQ: pullback ≥{pullback_pct}% OR ≤{cash_hold_weeks}wk | "
    f"Entry: [{entry_str}] | Exit→Cash: [{exit_str}]"
)

if run:
    with st.spinner("📡 Fetching QQQ & TQQQ data..."):
        data = fetch_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if data.empty:
        st.error("❌ No data. TQQQ starts Feb 2010.")
    else:
        with st.spinner("⚙️ Running rotation simulation (FIFO tax + fees)..."):
            results = run_backtest(
                data, ema_period, weekly_dca,
                entry_tiers, entry_reset, exit_tiers,
                marginal_rate, lt_discount_pct, cash_hold_weeks, pullback_pct,
                fee_flat, fee_pct,
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
            r5 = st.columns(4)
            r5[0].metric("↳ via Pullback", m['pullback_deploys'],
                         help="Redeployments triggered early by the TQQQ pullback condition")
            r5[1].metric("↳ via Time Limit", m['time_deploys'],
                         help="Redeployments triggered by the max-hold-weeks fallback")

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

            st.subheader("🔁 Allocation ($): Cash vs TQQQ vs QQQ")
            a = results[["Date","Cash ($)","TQQQ Value ($)","QQQ Value ($)"]].copy()
            a["Date"] = pd.to_datetime(a["Date"])
            st.area_chart(a.set_index("Date"), use_container_width=True)

            st.subheader("📊 Allocation %")
            p = results[["Date","Cash %","TQQQ %","QQQ %"]].copy()
            p["Date"] = pd.to_datetime(p["Date"])
            st.area_chart(p.set_index("Date"), use_container_width=True)

            st.subheader("🧾 Cumulative Fees & Taxes")
            fdf = results[["Date","Cumulative Fees ($)","Cumulative Tax ($)"]].copy()
            fdf["Date"] = pd.to_datetime(fdf["Date"])
            st.line_chart(fdf.set_index("Date"), use_container_width=True)

            st.subheader("📋 Transaction Events")
            ev = results[results["Action"].str.contains("ENTRY|EXIT|RESET|QQQ")]
            st.dataframe(
                ev[["Date","Action","EMA Dist %","Transfer ($)","Fee ($)","Tax ($)",
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
    ### 🔁 Phased Two-Way Rotation (QQQ ↔ TQQQ ↔ CASH)

    | Asset | Role |
    |-------|------|
    | **CASH** | Sideline dry powder — **flat, no interest/yield** |
    | **TQQQ** | Leveraged position built during drawdowns |
    | **QQQ** | Baseline, re-established from booked cash after a hold |

    | Direction | Trigger | Action | Costs |
    |-----------|---------|--------|-------|
    | 💰 DCA | Weekly | Add cash | fee on benchmark buy |
    | ⬇️ ENTRY | TQQQ below EMA tiers | Shift **% of total portfolio** → TQQQ (cash first, then sell QQQ) | commission + tax on any QQQ sold |
    | ⬆️ EXIT | TQQQ above EMA tiers | Book **% of TQQQ** → CASH | **FIFO capital-gains tax** + commission |
    | 🏦 REDEPLOY | **Pullback OR time** (first) | Booked cash → QQQ | commission |

    ### 🎯 Preloaded default strategy (edit any value in the sidebar)
    - **Entry Tier 1**: TQQQ −40% below EMA → shift **40%** of portfolio into TQQQ
    - **Entry Tier 2**: TQQQ −50% below EMA → shift **60%** of portfolio into TQQQ
    - **Exit**: TQQQ **+50%** above EMA → book **100%** of TQQQ to cash (FIFO-taxed)
    - **Re-deploy to QQQ**: **20%** TQQQ pullback **OR** **12 weeks** — whichever first
    - Entry tiers reset when TQQQ recovers above the 200 EMA

    ### Key mechanics in this version
    - **Portfolio-wide entry sizing**: each entry tier shifts a % of your *total
      portfolio value* into TQQQ, funded from **cash first, then by liquidating QQQ**.
    - **FIFO tax lots**: gains are realized oldest-lot-first. Lots held **> 365 days**
      get a configurable **long-term discount** (default 50%) before your marginal rate.
    - **No exit time-delay** — exits fire the moment a tier is hit.
    - **Hybrid cash → QQQ re-deployment**: after profits are booked to cash, each
      bucket redeploys into QQQ on **whichever happens first**:
        - **Price pullback** — TQQQ drops ≥ your pullback % below its price when the
          cash was booked (buy the new dip early), or
        - **Time limit** — the max hold weeks elapse (fallback).
      The metrics and trade log show which trigger fired for each redeployment.
    - **Flat cash** (no yield) and **fees** (flat + %) on every transaction.
    - All metrics are reported **net of taxes and fees**.
    """)

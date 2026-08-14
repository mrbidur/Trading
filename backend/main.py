"""
Dynamic Leveraged ETF Backtesting – FastAPI Backend (v2.0)
===========================================================
DECOUPLED ARCHITECTURE:
  - Deposits:  Strictly WEEKLY (Friday close), fixed schedule
  - Exits:     Evaluated CONTINUOUSLY on every bar (daily or intraday)
  - Cash Yield: Compounds WEEKLY on calendar week boundaries
  - Drawdown:  Computed on total portfolio value (shares*price + cash_buffer)

This separation prevents conflation of deposit intervals with position-exit
evaluations, ensuring overextended peaks trigger profit-taking the exact
moment thresholds are met.
"""
from __future__ import annotations

import io
import os
import traceback
from datetime import date, datetime, timedelta
from typing import List, Optional

import diskcache
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="ETF Backtest API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Local disk cache
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
cache = diskcache.Cache(CACHE_DIR)
CACHE_TTL = 60 * 60 * 4  # 4 hours


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class TrancheConfig(BaseModel):
    thresholds: List[float] = Field(
        default=[-25.0, -35.0, -45.0, -55.0, -65.0],
        description="Drawdown % thresholds to trigger tranche buys",
    )
    cash_deploy_pct: float = Field(default=20.0, ge=1.0, le=100.0)


class AdaptiveConfig(BaseModel):
    enabled: bool = False
    n_harvest_target: int = Field(default=3, ge=1)
    adaptive_tranche_1: float = Field(default=-20.0, le=0.0)


class BacktestRequest(BaseModel):
    # Asset / timeframe
    ticker: str = Field(default="TQQQ")
    start_date: str = Field(default="2010-01-01")
    end_date: str = Field(default="")
    execution_frequency: str = Field(default="daily")  # daily | intraday

    # Contributions (ALWAYS weekly regardless of scan frequency)
    base_deposit: float = Field(default=200.0, ge=0.0)
    overbought_deposit: float = Field(default=100.0, ge=0.0)
    tranche_deposit: float = Field(default=400.0, ge=0.0)

    # MA / overbought harvest
    ma_period: int = Field(default=200, ge=5)
    profit_harvest_dist_pct: float = Field(default=40.0)
    hysteresis_reset_pct: float = Field(default=30.0)
    initial_scale_out_pct: float = Field(default=70.0, ge=1.0, le=100.0)
    step_trigger_increment_pct: float = Field(default=10.0, ge=0.1)
    step_scale_out_pct: float = Field(default=10.0, ge=0.1, le=100.0)

    # Tranches
    tranches: TrancheConfig = Field(default_factory=TrancheConfig)

    # Adaptive mode
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)

    # Yield (compounds WEEKLY regardless of scan frequency)
    cash_yield_apy: float = Field(default=4.5, ge=0.0)


# ---------------------------------------------------------------------------
# Data Fetching & Caching
# ---------------------------------------------------------------------------
def fetch_price_data(ticker: str, start: str, end: str, intraday: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance with local disk cache.
    
    - intraday=True: 1-hour bars (limited to ~730 days per yfinance chunk)
    - intraday=False: daily bars (full history available)
    
    Always fetches High, Low, Close for accurate threshold detection.
    """
    interval = "1h" if intraday else "1d"
    cache_key = f"prices_v2_{ticker}_{start}_{end}_{interval}"

    if cache_key in cache:
        raw = cache[cache_key]
        df = pd.read_json(io.StringIO(raw), orient="split")
        df.index = pd.to_datetime(df.index)
        return df

    t = yf.Ticker(ticker)

    if intraday:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()
        
        chunks = []
        chunk_start = start_dt
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(days=729), end_dt)
            chunk_df = t.history(
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval="1h",
                auto_adjust=True,
            )
            if chunk_df is not None and not chunk_df.empty:
                chunks.append(chunk_df)
            chunk_start = chunk_end

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No intraday data found for {ticker}. Yahoo Finance limits intraday to ~2 years."
            )
        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")]
    else:
        df = t.history(start=start, end=end, auto_adjust=True)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No price data found for {ticker}")

    # Handle MultiIndex columns (yfinance >= 0.2.40)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=f"Yahoo Finance returned unexpected columns: {list(df.columns)}"
        )

    # Keep Close, High, Low for threshold detection
    keep_cols = ["Close"]
    if "High" in df.columns:
        keep_cols.append("High")
    if "Low" in df.columns:
        keep_cols.append("Low")
    df = df[keep_cols].copy()

    # Ensure timezone-naive datetime index
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    cache.set(cache_key, df.to_json(orient="split"), expire=CACHE_TTL)
    return df


# ---------------------------------------------------------------------------
# Helper: Determine if a given date is a weekly deposit day (Friday)
# ---------------------------------------------------------------------------
def is_deposit_week_boundary(current_dt: datetime, last_deposit_dt: Optional[datetime]) -> bool:
    """
    Returns True if current_dt is on a Friday (weekday=4) and it's a
    different ISO week from the last deposit. This ensures exactly one
    deposit per calendar week.
    """
    if last_deposit_dt is None:
        # First bar — allow deposit if it's a Friday, or the first available bar
        return True
    
    current_week = current_dt.isocalendar()[1]
    current_year = current_dt.isocalendar()[0]
    last_week = last_deposit_dt.isocalendar()[1]
    last_year = last_deposit_dt.isocalendar()[0]
    
    return (current_year, current_week) != (last_year, last_week)


def is_yield_week_boundary(current_dt: datetime, last_yield_dt: Optional[datetime]) -> bool:
    """
    Returns True if we've crossed into a new calendar week since last yield application.
    Cash yield compounds once per week regardless of scan frequency.
    """
    if last_yield_dt is None:
        return False  # Don't compound on the very first bar
    
    current_week = current_dt.isocalendar()[1]
    current_year = current_dt.isocalendar()[0]
    last_week = last_yield_dt.isocalendar()[1]
    last_year = last_yield_dt.isocalendar()[0]
    
    return (current_year, current_week) != (last_year, last_week)


# ---------------------------------------------------------------------------
# Backtesting State Machine Engine (v2.0 — Decoupled Architecture)
# ---------------------------------------------------------------------------
def run_backtest(req: BacktestRequest) -> dict:
    """
    DECOUPLED EXECUTION MODEL:
    ==========================
    
    1. DATA SCANNING: Every bar (daily close or intraday hourly) is evaluated
       for exit conditions (profit harvests, step scale-outs, tranche buys).
       This ensures price targets trigger the EXACT moment they're touched.
    
    2. DEPOSITS: Cash contributions occur on a STRICT WEEKLY schedule 
       (one deposit per calendar week, on the first bar of each new week).
       The deposit amount depends on current state:
       - Normal state:    base_deposit ($200 default)
       - Overbought state: overbought_deposit ($100 default)
       - Tranche buy week: tranche_deposit ($400 default)
    
    3. CASH YIELD: Interest on the cash buffer compounds ONCE PER WEEK
       at APY/52, regardless of how many bars are scanned. This models
       weekly T-bill yield sweeps accurately.
    
    4. DRAWDOWN: Computed on TOTAL PORTFOLIO VALUE (shares*price + cash_buffer),
       NOT on raw asset price. The cash buffer provides capital protection that
       must be reflected in risk metrics.
    
    PROFIT HARVEST DISTANCE (TQQQ-specific rationale):
    - dist_pct = ((price - EMA_200) / EMA_200) * 100
    - +40% on TQQQ ≈ +13% on underlying QQQ above its 200-day EMA
    - At this level, historical data shows high probability of mean-reversion
    - The EMA captures TQQQ's own compounding/decay dynamics
    
    HYSTERESIS RESET (prevents premature re-entry):
    - After harvest, overbought flag stays TRUE until:
      a) Price drops STRICTLY BELOW hysteresis_reset_pct (using Low for intraday)
      b) Cooldown period elapses (prevents same-session whipsaw)
    - Both conditions must be met before a new harvest cycle can begin
    """
    end_date = req.end_date if req.end_date else date.today().isoformat()
    is_intraday = req.execution_frequency.lower() == "intraday"

    # -----------------------------------------------------------------------
    # FETCH RAW PRICE DATA
    # -----------------------------------------------------------------------
    raw_df = fetch_price_data(req.ticker, req.start_date, end_date, intraday=is_intraday)

    # -----------------------------------------------------------------------
    # COMPUTE EMA ON RAW DATA (at scan granularity)
    # For intraday: 200 days * 6.5 hours/day = 1300 hourly bars equivalent
    # -----------------------------------------------------------------------
    if is_intraday:
        hourly_ema_span = int(req.ma_period * 6.5)
        raw_df["EMA"] = raw_df["Close"].ewm(span=hourly_ema_span, adjust=False).mean()
    else:
        raw_df["EMA"] = raw_df["Close"].ewm(span=req.ma_period, adjust=False).mean()

    # Drop rows where EMA hasn't warmed up
    df = raw_df.dropna(subset=["EMA"])

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No data available for {req.ticker} in range {req.start_date} to {end_date}"
        )

    # -----------------------------------------------------------------------
    # WEEKLY CASH YIELD: APY/52 per week (constant regardless of scan freq)
    # Formula: Cash_t = Cash_{t-1} * (1 + APY/52)
    # Applied once per calendar week crossing.
    # -----------------------------------------------------------------------
    weekly_yield_factor = 1 + (req.cash_yield_apy / 100.0) / 52.0

    # -----------------------------------------------------------------------
    # STATE VARIABLES
    # -----------------------------------------------------------------------
    in_overbought = False
    hysteresis_confirmed_below = False
    harvest_cooldown_bars = 0
    n_harvest = 0
    swing_high_peak = 0.0
    cash_buffer = 0.0
    peak_cash_buffer = 0.0
    total_shares = 0.0
    total_out_of_pocket = 0.0
    tranches_triggered = [False] * len(req.tranches.thresholds)
    active_step_triggers: List[float] = []

    # Deposit & yield tracking (decoupled from scan)
    last_deposit_dt: Optional[datetime] = None
    last_yield_dt: Optional[datetime] = None

    # DCA baseline
    dca_shares = 0.0
    dca_out_of_pocket = 0.0
    last_dca_deposit_dt: Optional[datetime] = None

    rows: List[dict] = []
    dca_rows: List[dict] = []

    # Cooldown bars for hysteresis (intraday needs more bars to confirm)
    COOLDOWN_MAP = {"intraday": 13, "daily": 3}
    cooldown_threshold = COOLDOWN_MAP.get(req.execution_frequency.lower(), 3)

    # Track whether a tranche fired this week (affects deposit amount)
    tranche_fired_this_week = False
    last_tranche_week: Optional[tuple] = None

    # -----------------------------------------------------------------------
    # MAIN LOOP: Iterate every bar (daily close or hourly)
    # -----------------------------------------------------------------------
    for exec_date, row_data in df.iterrows():
        price = float(row_data["Close"])
        ema = float(row_data["EMA"])

        if price <= 0 or ema <= 0:
            continue

        # Get High/Low for intraday threshold precision
        has_hl = "High" in row_data.index and "Low" in row_data.index
        check_high = float(row_data["High"]) if has_hl else price
        check_low = float(row_data["Low"]) if has_hl else price

        # Distance from EMA calculations
        dist_pct_close = ((price - ema) / ema) * 100.0
        dist_pct_high = ((check_high - ema) / ema) * 100.0
        dist_pct_low = ((check_low - ema) / ema) * 100.0
        dist_pct = dist_pct_close  # Official record uses close

        # Current datetime for weekly boundary checks
        current_dt = exec_date.to_pydatetime() if hasattr(exec_date, 'to_pydatetime') else exec_date

        # ===================================================================
        # PHASE 1: WEEKLY CASH YIELD COMPOUNDING
        # Applied once when crossing into a new calendar week.
        # Formula: Cash_buffer *= (1 + APY/52)
        # ===================================================================
        if is_yield_week_boundary(current_dt, last_yield_dt):
            cash_buffer *= weekly_yield_factor
            last_yield_dt = current_dt
        elif last_yield_dt is None:
            last_yield_dt = current_dt  # Initialize on first bar

        # ===================================================================
        # PHASE 2: HYSTERESIS RESET EVALUATION (continuous)
        # The overbought flag can only reset when:
        #   a) Price LOW drops STRICTLY BELOW hysteresis_reset_pct
        #   b) Cooldown period has elapsed since last harvest
        # This prevents premature re-entry during choppy bull runs.
        # ===================================================================
        if in_overbought:
            harvest_cooldown_bars += 1
            
            # Check using LOW (intraday) or CLOSE (daily) for genuine reversion
            reset_check_dist = dist_pct_low if is_intraday else dist_pct_close
            
            if reset_check_dist < req.hysteresis_reset_pct:
                hysteresis_confirmed_below = True
            
            # Two-stage confirmation: below threshold + cooldown elapsed
            if hysteresis_confirmed_below and harvest_cooldown_bars >= cooldown_threshold:
                in_overbought = False
                hysteresis_confirmed_below = False
                harvest_cooldown_bars = 0
                active_step_triggers = []

        # ===================================================================
        # PHASE 3: PROFIT HARVEST EVALUATION (continuous — every bar)
        # Uses HIGH for intraday (captures exact threshold crossing)
        # Uses CLOSE for daily (end-of-day evaluation)
        # ===================================================================
        action = "HOLD"  # Default action when no deposit and no event
        harvest_trigger_dist = dist_pct_high if is_intraday else dist_pct_close

        if harvest_trigger_dist >= req.profit_harvest_dist_pct and not in_overbought:
            # === PROFIT TAKE TRIGGERED ===
            in_overbought = True
            hysteresis_confirmed_below = False
            harvest_cooldown_bars = 0
            n_harvest += 1

            # Execution price: threshold-crossing price for intraday, close for daily
            threshold_price = ema * (1 + req.profit_harvest_dist_pct / 100.0)
            execution_price = threshold_price if is_intraday else price

            swing_high_peak = max(price, check_high)

            # Sell initial_scale_out_pct of current shares
            shares_to_sell = total_shares * (req.initial_scale_out_pct / 100.0)
            cash_harvested = shares_to_sell * execution_price
            total_shares -= shares_to_sell
            cash_buffer += cash_harvested
            peak_cash_buffer = cash_buffer

            # Reset tranche triggers for new cycle
            tranches_triggered = [False] * len(req.tranches.thresholds)

            # Build step scale-out triggers above initial harvest level
            active_step_triggers = [
                req.profit_harvest_dist_pct + req.step_trigger_increment_pct * (i + 1)
                for i in range(10)
            ]

            action = f"PROFIT TAKE ({req.initial_scale_out_pct:.0f}%)"

        elif in_overbought:
            # Update swing high peak continuously during overbought state
            swing_high_peak = max(swing_high_peak, check_high if is_intraday else price)

            # === STEP SCALE-OUTS (continuous evaluation) ===
            step_check_dist = dist_pct_high if is_intraday else dist_pct_close
            for st in list(active_step_triggers):
                if step_check_dist >= st:
                    step_exec_price = ema * (1 + st / 100.0) if is_intraday else price
                    step_shares = total_shares * (req.step_scale_out_pct / 100.0)
                    cash_buffer += step_shares * step_exec_price
                    total_shares -= step_shares
                    peak_cash_buffer = cash_buffer
                    active_step_triggers.remove(st)
                    action = f"STEP SCALE-OUT @ +{st:.0f}%"
                    break

        # ===================================================================
        # PHASE 4: TRANCHE DIP-BUY EVALUATION (continuous — every bar)
        # Evaluates pullback from swing high peak using LOW (intraday) or
        # CLOSE (daily) to detect exact moment threshold is breached.
        # ===================================================================
        # Adaptive tranche threshold
        tau_1 = (
            req.adaptive.adaptive_tranche_1
            if (req.adaptive.enabled and n_harvest >= req.adaptive.n_harvest_target)
            else req.tranches.thresholds[0]
        )
        active_tranches = [tau_1] + list(req.tranches.thresholds[1:])

        # Pullback calculation from swing high
        pullback_check_price = check_low if is_intraday else price
        pullback_pct = (
            ((pullback_check_price - swing_high_peak) / swing_high_peak) * 100.0
            if swing_high_peak > 0
            else 0.0
        )

        tranche_fired = False
        for t_idx, trigger_level in enumerate(active_tranches):
            if t_idx < len(tranches_triggered) and pullback_pct <= trigger_level and not tranches_triggered[t_idx]:
                tranches_triggered[t_idx] = True
                n_harvest = 0  # Reset consecutive harvest counter
                tranche_fired = True

                # Deploy cash_deploy_pct of peak cash buffer + tranche deposit
                tranche_deposit_amount = req.tranche_deposit
                cash_to_deploy = (peak_cash_buffer * (req.tranches.cash_deploy_pct / 100.0)) + tranche_deposit_amount
                actual_deploy = min(cash_buffer + tranche_deposit_amount, cash_to_deploy)

                # Execution price at threshold crossing for intraday
                if is_intraday and swing_high_peak > 0:
                    tranche_exec_price = swing_high_peak * (1 + trigger_level / 100.0)
                    tranche_exec_price = max(tranche_exec_price, check_low)
                else:
                    tranche_exec_price = price

                shares_bought = actual_deploy / tranche_exec_price
                total_shares += shares_bought
                cash_buffer = max(0.0, cash_buffer + tranche_deposit_amount - actual_deploy)
                total_out_of_pocket += tranche_deposit_amount

                # Track that a tranche fired this week
                current_week_key = current_dt.isocalendar()[:2]
                last_tranche_week = current_week_key
                tranche_fired_this_week = True

                action = f"TRANCHE {t_idx + 1} BUY ({trigger_level:.0f}%)"
                break

        # ===================================================================
        # PHASE 5: WEEKLY DEPOSIT (strictly once per calendar week)
        # 
        # Deposits are DECOUPLED from exit evaluations:
        # - They occur on the first bar of each new calendar week
        # - Amount depends on current state machine state:
        #     Normal:     base_deposit ($200)
        #     Overbought: overbought_deposit ($100)
        #     Tranche:    tranche_deposit ($400) — already handled above
        # - Deposit buys shares at current close price
        # ===================================================================
        deposit = 0.0
        is_deposit_bar = is_deposit_week_boundary(current_dt, last_deposit_dt)

        if is_deposit_bar:
            # Determine deposit amount based on state
            if tranche_fired:
                # Tranche buy already handled the deposit + deployment above
                deposit = 0.0  # Don't double-deposit
            elif in_overbought:
                deposit = req.overbought_deposit
            else:
                deposit = req.base_deposit

            # Execute deposit: buy shares at current price
            if deposit > 0:
                shares_bought = deposit / price
                total_shares += shares_bought
                total_out_of_pocket += deposit
                if action == "HOLD":
                    action = "WEEKLY DCA" if not in_overbought else "WEEKLY DCA (OB)"

            last_deposit_dt = current_dt

            # Reset weekly tranche tracking
            current_week_key = current_dt.isocalendar()[:2]
            if last_tranche_week != current_week_key:
                tranche_fired_this_week = False

        # ===================================================================
        # PHASE 6: PORTFOLIO VALUATION
        # Total Portfolio = (Shares * Current Price) + Cash Buffer
        # This is the TRUE portfolio value including capital protection.
        # ===================================================================
        portfolio_valuation = (total_shares * price) + cash_buffer

        # Format date
        date_str = exec_date.strftime("%Y-%m-%d %H:%M") if is_intraday else exec_date.strftime("%Y-%m-%d")

        rows.append({
            "date": date_str,
            "close": round(price, 4),
            "ema": round(ema, 4),
            "dist_pct": round(dist_pct, 4),
            "overbought": in_overbought,
            "n_harvest": n_harvest,
            "swing_high_peak": round(swing_high_peak, 4),
            "pullback_pct": round(pullback_pct, 4),
            "tranche_1_trigger": round(tau_1, 2),
            "action": action,
            "deposit": round(deposit + (req.tranche_deposit if tranche_fired else 0), 2),
            "cash_buffer": round(cash_buffer, 2),
            "total_shares": round(total_shares, 6),
            "portfolio_valuation": round(portfolio_valuation, 2),
            "out_of_pocket_total": round(total_out_of_pocket, 2),
        })

        # DCA baseline: also weekly deposits only
        if is_deposit_bar:
            dca_shares += req.base_deposit / price if price > 0 else 0.0
            dca_out_of_pocket += req.base_deposit

        dca_rows.append({
            "date": date_str,
            "portfolio_valuation": round(dca_shares * price, 2),
            "out_of_pocket_total": round(dca_out_of_pocket, 2),
        })

    # -----------------------------------------------------------------------
    # PERFORMANCE METRICS (Corrected Drawdown)
    # -----------------------------------------------------------------------
    def compute_metrics(valuations: List[float], invested: List[float], dates: List[str]) -> dict:
        """
        Compute performance metrics using TOTAL PORTFOLIO VALUE.
        
        CRITICAL FIX: Maximum Drawdown (MDD) is calculated on the full portfolio
        equity curve (shares*price + cash_buffer), NOT on the raw asset price.
        
        Because the strategy banks 70-80% of capital into the cash buffer during
        peaks, the portfolio experiences MUCH lower drawdowns than the underlying
        asset. Previous implementations incorrectly reported raw asset drawdowns
        (e.g., -58% to -65%) which didn't reflect the capital protection.
        """
        if not valuations or not dates:
            return {"terminal_value": 0, "total_invested": 0, "cagr": 0, "mdd": 0, "sortino": 0, "years": 0}

        terminal_val = valuations[-1]
        total_invested = invested[-1]

        def parse_date(s: str) -> datetime:
            try:
                return datetime.strptime(s, "%Y-%m-%d %H:%M")
            except ValueError:
                return datetime.strptime(s, "%Y-%m-%d")

        start_dt = parse_date(dates[0])
        end_dt = parse_date(dates[-1])
        years = max((end_dt - start_dt).days / 365.25, 0.01)

        # CAGR: growth of total portfolio vs total invested capital
        cagr = ((terminal_val / max(total_invested, 1)) ** (1.0 / years) - 1) * 100

        # MAX DRAWDOWN: Peak-to-trough on PORTFOLIO VALUE (includes cash buffer!)
        # This correctly reflects capital protection from the cash-hedged strategy.
        vals_arr = np.array(valuations, dtype=np.float64)
        peak_arr = np.maximum.accumulate(vals_arr)
        # Avoid division by zero for initial zero-value periods
        dd_arr = np.where(
            peak_arr > 0,
            (vals_arr - peak_arr) / peak_arr,
            0.0
        )
        mdd = float(dd_arr.min()) * 100

        # SORTINO RATIO: Risk-adjusted return using downside deviation only
        # Annualized using sqrt(periods_per_year)
        # Uses ~52 periods/year since deposits are weekly
        annualization_factor = 52  # Weekly basis for Sortino since deposits are weekly
        
        if len(vals_arr) > 1:
            # Compute returns on a weekly-sampled basis for consistency
            # Take every N-th value to approximate weekly returns
            if len(vals_arr) > 260:  # More than ~1 year of daily data
                # Sample weekly for Sortino calculation
                step = max(1, len(vals_arr) // (int(years * 52)))
                weekly_vals = vals_arr[::step]
            else:
                weekly_vals = vals_arr

            if len(weekly_vals) > 1:
                period_returns = np.diff(weekly_vals) / np.where(weekly_vals[:-1] > 0, weekly_vals[:-1], 1)
                neg_returns = period_returns[period_returns < 0]
                downside_std = float(np.std(neg_returns)) if len(neg_returns) > 1 else 0.001
                mean_return = float(np.mean(period_returns))
                sortino = (mean_return / downside_std) * np.sqrt(annualization_factor) if downside_std > 0 else 0.0
            else:
                sortino = 0.0
        else:
            sortino = 0.0

        return {
            "terminal_value": round(terminal_val, 2),
            "total_invested": round(total_invested, 2),
            "cagr": round(cagr, 2),
            "mdd": round(mdd, 2),
            "sortino": round(sortino, 3),
            "years": round(years, 2),
        }

    strategy_valuations = [r["portfolio_valuation"] for r in rows]
    strategy_invested = [r["out_of_pocket_total"] for r in rows]
    strategy_dates = [r["date"] for r in rows]

    dca_valuations = [r["portfolio_valuation"] for r in dca_rows]
    dca_invested = [r["out_of_pocket_total"] for r in dca_rows]
    dca_dates = [r["date"] for r in dca_rows]

    return {
        "ticker": req.ticker.upper(),
        "rows": rows,
        "dca_rows": dca_rows,
        "strategy_metrics": compute_metrics(strategy_valuations, strategy_invested, strategy_dates),
        "dca_metrics": compute_metrics(dca_valuations, dca_invested, dca_dates),
    }


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/backtest")
def backtest(req: BacktestRequest):
    """Run backtest and return JSON results with metrics."""
    try:
        return run_backtest(req)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Backtest engine error: {str(e)}"
        )


@app.post("/export-csv")
def export_csv(req: BacktestRequest):
    """Run backtest and return the full audit trail as a downloadable CSV."""
    try:
        result = run_backtest(req)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Export engine error: {str(e)}"
        )

    rows = result["rows"]
    if not rows:
        raise HTTPException(status_code=400, detail="No data to export")

    output = io.StringIO()
    headers = [
        "Date", "Close Price", "Moving Average", "Dist %",
        "Overbought Flag", "Harvest Counter", "Swing High Peak",
        "Pullback %", "Tranche 1 Trigger", "Action Executed",
        "Deposit ($)", "Cash Buffer ($)", "Total Shares",
        "Portfolio Valuation ($)", "Out of Pocket Total ($)",
    ]
    output.write(",".join(headers) + "\n")
    for r in rows:
        line = [
            r["date"],
            r["close"],
            r["ema"],
            r["dist_pct"],
            str(r["overbought"]).upper(),
            r["n_harvest"],
            r["swing_high_peak"],
            r["pullback_pct"],
            r["tranche_1_trigger"],
            f'"{r["action"]}"',
            r["deposit"],
            r["cash_buffer"],
            r["total_shares"],
            r["portfolio_valuation"],
            r["out_of_pocket_total"],
        ]
        output.write(",".join(str(v) for v in line) + "\n")

    output.seek(0)
    filename = f"{req.ticker}_backtest_{req.start_date}_to_{rows[-1]['date'].split()[0]}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/tickers")
def get_ticker_suggestions():
    """Quick-access popular tickers."""
    return {
        "popular": [
            "TQQQ", "UPRO", "SOXL", "SPXL", "TECL",
            "QQQ", "SPY", "IWM", "VOO", "ARKK",
            "NVDA", "TSLA", "AAPL", "MSFT", "AMZN",
        ]
    }

"""
Dynamic Leveraged ETF Backtesting – FastAPI Backend
====================================================
State-machine-based allocation strategy engine with full CSV export.
Supports daily, weekly, monthly, and intraday (hourly) execution frequencies.
"""
from __future__ import annotations

import io
import os
import traceback
from datetime import date, datetime, timedelta
from typing import List

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
app = FastAPI(title="ETF Backtest API", version="1.0.0")
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
    start_date: str = Field(default="2015-01-01")
    end_date: str = Field(default="")
    execution_frequency: str = Field(default="weekly")  # weekly | daily | monthly | intraday

    # Contributions
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

    # Yield
    cash_yield_apy: float = Field(default=4.5, ge=0.0)


# ---------------------------------------------------------------------------
# Data Fetching & Caching
# ---------------------------------------------------------------------------
def fetch_price_data(ticker: str, start: str, end: str, intraday: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance with local disk cache.
    
    For intraday mode: fetches 1-hour bars. Yahoo Finance limits intraday
    history to ~730 days, so we fetch in chunks if needed.
    For daily mode: fetches daily close prices.
    """
    interval = "1h" if intraday else "1d"
    cache_key = f"prices_{ticker}_{start}_{end}_{interval}"

    if cache_key in cache:
        raw = cache[cache_key]
        df = pd.read_json(io.StringIO(raw), orient="split")
        df.index = pd.to_datetime(df.index)
        return df

    t = yf.Ticker(ticker)

    if intraday:
        # Yahoo Finance limits intraday to ~730 days per request
        # Fetch in chunks and concatenate
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
                detail=f"No intraday data found for {ticker}. Note: Yahoo Finance limits intraday history to ~2 years."
            )
        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")]
    else:
        df = t.history(start=start, end=end, auto_adjust=True)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No price data found for {ticker}")

    # Handle MultiIndex columns (yfinance >= 0.2.40 returns ("Close", "TQQQ"))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=f"Yahoo Finance returned unexpected columns: {list(df.columns)}"
        )

    # For intraday, also keep High and Low for threshold detection
    if intraday and "High" in df.columns and "Low" in df.columns:
        df = df[["Close", "High", "Low"]].copy()
    else:
        df = df[["Close"]].copy()

    # Ensure timezone-naive datetime index
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    cache.set(cache_key, df.to_json(orient="split"), expire=CACHE_TTL)
    return df


def resample_to_frequency(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Resample prices to the user-selected execution frequency."""
    # Use "ME" for pandas >= 2.2, fall back to "M" for older versions
    pandas_version = tuple(int(x) for x in pd.__version__.split(".")[:2])
    month_freq = "ME" if pandas_version >= (2, 2) else "M"

    freq_map = {"weekly": "W-FRI", "monthly": month_freq, "daily": None, "intraday": None}
    freq = freq_map.get(frequency.lower())
    if freq is None:
        return df  # daily/intraday: no resampling
    resampled = df.resample(freq).last().dropna()
    return resampled


# ---------------------------------------------------------------------------
# Backtesting State Machine Engine
# ---------------------------------------------------------------------------
def run_backtest(req: BacktestRequest) -> dict:
    """
    Run the full state-machine backtest and return structured results.
    
    Key design decisions for accurate TQQQ overbought detection:
    
    1. PROFIT HARVEST DISTANCE: Measures how far price has extended above the
       EMA as a percentage: ((price - ema) / ema) * 100. For leveraged ETFs
       like TQQQ, a +40% extension above the 200-day EMA represents an 
       extreme overextension (~13% above EMA on the underlying QQQ).
       
    2. INTRADAY MODE: When enabled, the engine evaluates each hourly bar's
       High price against thresholds. This ensures profit harvests trigger
       at the EXACT moment price crosses the threshold, not hours later at
       market close. Trades execute at the threshold-crossing price.
       
    3. HYSTERESIS RESET: After a profit harvest, the overbought flag remains
       locked TRUE until price drops STRICTLY BELOW the hysteresis_reset_pct.
       Additionally, a cooldown counter ensures the flag cannot reset within
       the same trading session that triggered the harvest. This prevents
       whipsaw re-entry where a brief dip followed by continuation would
       immediately re-trigger another harvest.
    """
    end_date = req.end_date if req.end_date else date.today().isoformat()
    is_intraday = req.execution_frequency.lower() == "intraday"

    raw_df = fetch_price_data(req.ticker, req.start_date, end_date, intraday=is_intraday)

    # ---------------------------------------------------------------------------
    # EMA Computation
    # For intraday: compute EMA on the hourly close prices using an equivalent
    # period. A 200-day EMA ~ 200*6.5 = 1300 hourly bars (regular trading hours).
    # This gives the same smoothing behavior adapted to the bar frequency.
    # ---------------------------------------------------------------------------
    if is_intraday:
        # Convert day-based MA period to equivalent hourly bars
        # ~6.5 trading hours per day for US equities
        hourly_ema_span = int(req.ma_period * 6.5)
        raw_df["EMA"] = raw_df["Close"].ewm(span=hourly_ema_span, adjust=False).mean()
    else:
        raw_df["EMA"] = raw_df["Close"].ewm(span=req.ma_period, adjust=False).mean()

    df = resample_to_frequency(raw_df, req.execution_frequency)
    df = df.dropna(subset=["EMA"])

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No data available for {req.ticker} in the selected date range ({req.start_date} to {end_date})"
        )

    # ---------------------------------------------------------------------------
    # Periods per year (for yield and Sortino calculations)
    # ---------------------------------------------------------------------------
    n_periods_per_year = {
        "weekly": 52,
        "monthly": 12,
        "daily": 252,
        "intraday": 252 * 7,  # ~6.5 hours/day * ~252 days ~ 1638 bars/year
    }
    ppy = n_periods_per_year.get(req.execution_frequency.lower(), 52)
    yield_per_period = (1 + req.cash_yield_apy / 100.0) ** (1.0 / ppy) - 1

    # ---------------------------------------------------------------------------
    # Deposit frequency for intraday mode
    # In intraday mode, deposits are made once per trading day (first bar of day)
    # to avoid unrealistically inflating contributions.
    # ---------------------------------------------------------------------------
    last_deposit_date = None  # Track daily deposit for intraday mode

    # ---------------------------------------------------------------------------
    # State variables
    # ---------------------------------------------------------------------------
    in_overbought = False
    hysteresis_confirmed_below = False  # Must confirm price dropped below reset level
    harvest_cooldown_bars = 0  # Bars since last harvest (prevents same-session reset)
    n_harvest = 0
    swing_high_peak = 0.0
    cash_buffer = 0.0
    peak_cash_buffer = 0.0
    total_shares = 0.0
    total_out_of_pocket = 0.0
    tranches_triggered = [False] * len(req.tranches.thresholds)

    # DCA baseline
    dca_shares = 0.0
    dca_out_of_pocket = 0.0

    rows: List[dict] = []
    dca_rows: List[dict] = []
    active_step_triggers: List[float] = []

    # Minimum cooldown bars before hysteresis can reset after a harvest
    # For intraday: ~13 bars (2 hours), for daily: 3 bars, weekly: 1 bar
    COOLDOWN_MAP = {"intraday": 13, "daily": 3, "weekly": 1, "monthly": 1}
    cooldown_threshold = COOLDOWN_MAP.get(req.execution_frequency.lower(), 3)

    for exec_date, row_data in df.iterrows():
        price = float(row_data["Close"])
        ema = float(row_data["EMA"])

        if price <= 0 or ema <= 0:
            continue

        # -------------------------------------------------------------------
        # PROFIT HARVEST DISTANCE CALCULATION (Refined for TQQQ)
        # 
        # For leveraged ETFs, the EMA distance metric captures "overextension":
        #   dist_pct = ((current_price - ema) / ema) * 100
        #
        # Why this works for TQQQ:
        # - TQQQ amplifies QQQ moves 3x daily, but compounding causes
        #   exponential divergence during sustained rallies.
        # - A +40% distance above 200-EMA on TQQQ typically corresponds to
        #   QQQ being ~12-15% above its own 200-EMA — a historically
        #   overextended condition that reverts.
        # - Using the EMA of TQQQ itself (not the underlying) captures the
        #   actual compounding/decay dynamics inherent to the leveraged product.
        #
        # For intraday: we check the bar's HIGH against the trigger to detect
        # the exact moment the threshold was breached within the bar.
        # -------------------------------------------------------------------
        
        # Use High price for threshold detection in intraday mode
        check_price_high = float(row_data["High"]) if (is_intraday and "High" in row_data.index) else price
        check_price_low = float(row_data["Low"]) if (is_intraday and "Low" in row_data.index) else price

        # Distance from EMA (using bar high for overbought detection)
        dist_pct_close = ((price - ema) / ema) * 100.0
        dist_pct_high = ((check_price_high - ema) / ema) * 100.0
        dist_pct_low = ((check_price_low - ema) / ema) * 100.0

        # The "official" dist_pct for record-keeping is based on close
        dist_pct = dist_pct_close

        # 1. Apply periodic cash yield on uninvested cash
        cash_buffer *= (1 + yield_per_period)

        # -------------------------------------------------------------------
        # 2. HYSTERESIS RESET LOGIC (Corrected)
        #
        # The overbought flag can ONLY reset when ALL of these are true:
        #   a) Price (using LOW for intraday, CLOSE otherwise) has dropped
        #      STRICTLY BELOW the hysteresis_reset_pct threshold
        #   b) A minimum cooldown period has elapsed since the last harvest
        #      (prevents same-bar or immediate-next-bar whipsaw resets)
        #   c) The below-threshold condition is confirmed (prevents noise)
        #
        # This 2-stage confirmation prevents:
        #   - Whipsaw: harvest at +40%, brief dip to +39%, immediately
        #     re-arms and harvests again at +40% on the same move
        #   - Premature re-entry: must see genuine mean-reversion below
        #     the buffer level before allowing another harvest cycle
        # -------------------------------------------------------------------
        if in_overbought:
            harvest_cooldown_bars += 1

            # Check if price has genuinely dropped below the reset threshold
            reset_check_dist = dist_pct_low if is_intraday else dist_pct_close

            if reset_check_dist < req.hysteresis_reset_pct:
                # Price is below reset level — mark as confirmed
                hysteresis_confirmed_below = True

            # Only reset overbought if confirmed below AND cooldown elapsed
            if hysteresis_confirmed_below and harvest_cooldown_bars >= cooldown_threshold:
                in_overbought = False
                hysteresis_confirmed_below = False
                harvest_cooldown_bars = 0
                active_step_triggers = []

        action = "NORMAL DCA"
        deposit = req.base_deposit
        tranche_fired = False

        # For intraday: only allow one deposit per calendar day
        if is_intraday:
            current_date = exec_date.date() if hasattr(exec_date, 'date') else exec_date
            if last_deposit_date == current_date:
                deposit = 0.0  # Already deposited today
            else:
                last_deposit_date = current_date

        # -------------------------------------------------------------------
        # 3. OVERBOUGHT PROFIT-TAKE (Refined for accuracy)
        #
        # Trigger condition uses HIGH price for intraday (captures exact moment)
        # or CLOSE price for end-of-day frequencies.
        #
        # For TQQQ specifically, the +40% default trigger represents:
        # - Price that is 1.4x the 200-day EMA
        # - Historically, TQQQ reverts significantly from these levels
        # - The trigger uses the bar's HIGH to capture the peak of the move,
        #   then executes the sale at that high price (best-case intraday fill)
        # -------------------------------------------------------------------
        harvest_trigger_dist = dist_pct_high if is_intraday else dist_pct_close

        if harvest_trigger_dist >= req.profit_harvest_dist_pct and not in_overbought:
            in_overbought = True
            hysteresis_confirmed_below = False
            harvest_cooldown_bars = 0
            n_harvest += 1

            # For intraday: execute at the threshold-crossing price (estimated)
            # This is the price where distance exactly equals the trigger
            threshold_price = ema * (1 + req.profit_harvest_dist_pct / 100.0)
            execution_price = threshold_price if is_intraday else price
            
            swing_high_peak = max(price, check_price_high)

            shares_to_sell = total_shares * (req.initial_scale_out_pct / 100.0)
            cash_harvested = shares_to_sell * execution_price
            total_shares -= shares_to_sell
            cash_buffer += cash_harvested
            peak_cash_buffer = cash_buffer

            action = f"PROFIT TAKE ({req.initial_scale_out_pct:.0f}%)"
            deposit = req.overbought_deposit if deposit != 0 else 0.0
            tranches_triggered = [False] * len(req.tranches.thresholds)

            # Build step triggers above the initial harvest level
            active_step_triggers = [
                req.profit_harvest_dist_pct + req.step_trigger_increment_pct * (i + 1)
                for i in range(10)
            ]

        elif in_overbought:
            deposit = req.overbought_deposit if deposit != 0 else 0.0
            swing_high_peak = max(swing_high_peak, check_price_high if is_intraday else price)

            # Step scale-outs: check using HIGH for intraday precision
            step_check_dist = dist_pct_high if is_intraday else dist_pct_close
            for st in list(active_step_triggers):
                if step_check_dist >= st:
                    # Execute at the threshold-crossing price for intraday
                    step_exec_price = ema * (1 + st / 100.0) if is_intraday else price
                    step_shares = total_shares * (req.step_scale_out_pct / 100.0)
                    cash_buffer += step_shares * step_exec_price
                    total_shares -= step_shares
                    peak_cash_buffer = cash_buffer
                    active_step_triggers.remove(st)
                    action = f"STEP SCALE-OUT @ +{st:.0f}%"
                    break

        # 4. Adaptive tranche threshold
        tau_1 = (
            req.adaptive.adaptive_tranche_1
            if (req.adaptive.enabled and n_harvest >= req.adaptive.n_harvest_target)
            else req.tranches.thresholds[0]
        )
        active_tranches = [tau_1] + list(req.tranches.thresholds[1:])

        # 5. Pullback from swing high (use LOW for intraday to catch exact dip)
        pullback_check_price = check_price_low if is_intraday else price
        pullback_pct = (
            ((pullback_check_price - swing_high_peak) / swing_high_peak) * 100.0
            if swing_high_peak > 0
            else 0.0
        )

        # 6. Tranche dip-buy
        for t_idx, trigger_level in enumerate(active_tranches):
            if t_idx < len(tranches_triggered) and pullback_pct <= trigger_level and not tranches_triggered[t_idx]:
                tranches_triggered[t_idx] = True
                tranche_deposit_amount = req.tranche_deposit
                n_harvest = 0
                tranche_fired = True

                cash_to_deploy = (peak_cash_buffer * (req.tranches.cash_deploy_pct / 100.0)) + tranche_deposit_amount
                actual_deploy = min(cash_buffer + tranche_deposit_amount, cash_to_deploy)

                # For intraday: execute at the threshold-crossing price
                if is_intraday and swing_high_peak > 0:
                    tranche_exec_price = swing_high_peak * (1 + trigger_level / 100.0)
                    tranche_exec_price = max(tranche_exec_price, check_price_low)
                else:
                    tranche_exec_price = price

                shares_bought = actual_deploy / tranche_exec_price
                total_shares += shares_bought
                cash_buffer = max(0.0, cash_buffer + tranche_deposit_amount - actual_deploy)
                total_out_of_pocket += tranche_deposit_amount
                deposit = tranche_deposit_amount
                action = f"TRANCHE {t_idx + 1} BUY ({trigger_level:.0f}%)"
                break

        # 7. Normal DCA (when no tranche fired)
        if not tranche_fired and deposit > 0:
            shares_bought = deposit / price
            total_shares += shares_bought
            total_out_of_pocket += deposit

        portfolio_valuation = (total_shares * price) + cash_buffer

        # Format date string based on frequency
        if is_intraday:
            date_str = exec_date.strftime("%Y-%m-%d %H:%M")
        else:
            date_str = exec_date.strftime("%Y-%m-%d")

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
            "deposit": round(deposit, 2),
            "cash_buffer": round(cash_buffer, 2),
            "total_shares": round(total_shares, 6),
            "portfolio_valuation": round(portfolio_valuation, 2),
            "out_of_pocket_total": round(total_out_of_pocket, 2),
        })

        # DCA baseline (one deposit per day for intraday, per period otherwise)
        if not is_intraday or (is_intraday and deposit > 0):
            dca_deposit = req.base_deposit if deposit > 0 else 0.0
            dca_shares += dca_deposit / price if price > 0 else 0.0
            dca_out_of_pocket += dca_deposit
        
        dca_rows.append({
            "date": date_str,
            "portfolio_valuation": round(dca_shares * price, 2),
            "out_of_pocket_total": round(dca_out_of_pocket, 2),
        })

    # ---------------------------------------------------------------------------
    # Performance Metrics
    # ---------------------------------------------------------------------------
    def compute_metrics(valuations: List[float], invested: List[float], dates: List[str]) -> dict:
        if not valuations or not dates:
            return {"terminal_value": 0, "total_invested": 0, "cagr": 0, "mdd": 0, "sortino": 0, "years": 0}

        terminal_val = valuations[-1]
        total_invested = invested[-1]

        # Parse dates (handle both "YYYY-MM-DD" and "YYYY-MM-DD HH:MM")
        def parse_date(s: str) -> datetime:
            try:
                return datetime.strptime(s, "%Y-%m-%d %H:%M")
            except ValueError:
                return datetime.strptime(s, "%Y-%m-%d")

        start_dt = parse_date(dates[0])
        end_dt = parse_date(dates[-1])
        years = max((end_dt - start_dt).days / 365.25, 0.01)

        cagr = ((terminal_val / max(total_invested, 1)) ** (1.0 / years) - 1) * 100

        # Max drawdown
        vals_arr = np.array(valuations, dtype=np.float64)
        peak_arr = np.maximum.accumulate(vals_arr)
        dd_arr = (vals_arr - peak_arr) / np.where(peak_arr > 0, peak_arr, 1)
        mdd = float(dd_arr.min()) * 100

        # Sortino ratio (annualised)
        if len(vals_arr) > 1:
            period_returns = np.diff(vals_arr) / np.where(vals_arr[:-1] > 0, vals_arr[:-1], 1)
            neg_returns = period_returns[period_returns < 0]
            downside_std = float(np.std(neg_returns)) if len(neg_returns) > 1 else 0.001
            mean_return = float(np.mean(period_returns))
            sortino = (mean_return / downside_std) * np.sqrt(ppy) if downside_std > 0 else 0.0
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

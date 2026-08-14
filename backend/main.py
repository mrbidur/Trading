"""
Dynamic Leveraged ETF Backtesting – FastAPI Backend (v2.2)
===========================================================
WEEKLY FRIDAY CLOSE EXECUTION MODEL (Matches Research PDF):
  - ALL logic evaluated on Friday close only: deposits, harvests, tranches
  - EMA computed daily, sampled on Friday close for evaluation
  - No mid-week threshold catching — strict weekly evaluation cadence
  - Cash yield compounds weekly (applied each Friday)
  - MDD on total portfolio value (shares*price + cash_buffer)

This aligns exactly with the research PDF's 860-week evaluation methodology
where every decision is made using Friday close prices.
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
app = FastAPI(title="ETF Backtest API", version="2.2.0")
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
    execution_frequency: str = Field(default="weekly")  # weekly | daily | intraday

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
    Always fetches daily data for EMA computation.
    For intraday mode, fetches hourly bars.
    """
    interval = "1h" if intraday else "1d"
    cache_key = f"prices_v22_{ticker}_{start}_{end}_{interval}"

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

    keep_cols = ["Close"]
    if "High" in df.columns:
        keep_cols.append("High")
    if "Low" in df.columns:
        keep_cols.append("Low")
    df = df[keep_cols].copy()

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    cache.set(cache_key, df.to_json(orient="split"), expire=CACHE_TTL)
    return df


# ---------------------------------------------------------------------------
# Backtesting State Machine Engine (v2.2 — Weekly Friday Close Evaluation)
# ---------------------------------------------------------------------------
def run_backtest(req: BacktestRequest) -> dict:
    """
    WEEKLY FRIDAY CLOSE EXECUTION MODEL
    ====================================
    
    Matches the research PDF methodology exactly:
    
    1. DAILY EMA COMPUTATION: The 200-day EMA is computed on every trading day
       using all daily close prices (continuous calculation from Day 1).
    
    2. FRIDAY-ONLY EVALUATION: ALL strategy logic — deposits, profit harvests,
       step scale-outs, tranche buys, hysteresis resets — is evaluated
       exclusively on Friday close bars (resampled weekly W-FRI).
       This means NO mid-week threshold catching. If TQQQ spikes +45% above
       EMA on a Tuesday but closes Friday at +38%, no harvest fires.
    
    3. WEEKLY CASH YIELD: Cash buffer compounds once per week at APY/52,
       applied on each Friday evaluation bar.
    
    4. DEPOSIT LOGIC: One deposit per Friday bar. Amount determined by state:
       - Normal:     $200 (base_deposit)
       - Overbought: $100 (overbought_deposit)  
       - Tranche:    $400 (tranche_deposit) — replaces normal deposit
    
    5. PORTFOLIO MDD: Computed on total portfolio value (shares*price + cash)
       with warm-up exclusion to avoid artificial tiny-portfolio drawdowns.
    
    KEY FIXES PRESERVED:
    - FIX 1: peak_cash_buffer anchored ONLY at initial harvest
    - FIX 2: n_harvest resets ONLY on tranche buy execution
    - FIX 3: MDD on portfolio value, not raw asset price
    - FIX 4: Step scale-outs don't update peak_cash_buffer
    """
    end_date = req.end_date if req.end_date else date.today().isoformat()
    is_intraday = req.execution_frequency.lower() == "intraday"
    is_weekly = req.execution_frequency.lower() == "weekly"

    # -----------------------------------------------------------------------
    # FETCH DAILY PRICE DATA (always daily for EMA computation)
    # -----------------------------------------------------------------------
    raw_df = fetch_price_data(req.ticker, req.start_date, end_date, intraday=is_intraday)

    # -----------------------------------------------------------------------
    # COMPUTE EMA ON DAILY DATA
    # The 200-day EMA is calculated from Day 1 using all daily closes.
    # For intraday: 200 days * 6.5 hours = 1300 hourly bars equivalent.
    # -----------------------------------------------------------------------
    if is_intraday:
        hourly_ema_span = int(req.ma_period * 6.5)
        raw_df["EMA"] = raw_df["Close"].ewm(span=hourly_ema_span, adjust=False).mean()
    else:
        raw_df["EMA"] = raw_df["Close"].ewm(span=req.ma_period, adjust=False).mean()

    # -----------------------------------------------------------------------
    # RESAMPLE TO EXECUTION FREQUENCY
    # For weekly: resample to Friday close (W-FRI) — this is the core change
    # that aligns with the research PDF's 860-week evaluation schedule.
    # -----------------------------------------------------------------------
    if is_weekly:
        # Resample to weekly Friday close
        df = raw_df.resample("W-FRI").last().dropna()
    elif is_intraday:
        df = raw_df.dropna(subset=["EMA"])
    else:
        # Daily: no resampling
        df = raw_df.dropna(subset=["EMA"])

    df = df.dropna(subset=["EMA"])

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No data available for {req.ticker} in range {req.start_date} to {end_date}"
        )

    # -----------------------------------------------------------------------
    # WEEKLY CASH YIELD: APY/52 per evaluation period
    # Applied once per evaluation bar (weekly = once per Friday)
    # -----------------------------------------------------------------------
    weekly_yield_factor = 1 + (req.cash_yield_apy / 100.0) / 52.0

    # -----------------------------------------------------------------------
    # STATE VARIABLES
    # -----------------------------------------------------------------------
    in_overbought = False
    n_harvest = 0                    # FIX 2: only resets on tranche buy
    swing_high_peak = 0.0
    cash_buffer = 0.0
    peak_cash_buffer = 0.0           # FIX 1: only set at initial harvest
    total_shares = 0.0
    total_out_of_pocket = 0.0
    tranches_triggered = [False] * len(req.tranches.thresholds)
    active_step_triggers: List[float] = []

    # DCA baseline
    dca_shares = 0.0
    dca_out_of_pocket = 0.0

    rows: List[dict] = []
    dca_rows: List[dict] = []

    # -----------------------------------------------------------------------
    # MAIN LOOP: Iterate each evaluation bar
    # For weekly mode: each bar IS a Friday close (one bar per week)
    # ALL logic — deposits, harvests, tranches — evaluated on same bar
    # -----------------------------------------------------------------------
    for exec_date, row_data in df.iterrows():
        price = float(row_data["Close"])
        ema = float(row_data["EMA"])

        if price <= 0 or ema <= 0:
            continue

        # Distance from EMA (using Friday close price)
        dist_pct = ((price - ema) / ema) * 100.0

        # ===================================================================
        # STEP 1: WEEKLY CASH YIELD COMPOUNDING
        # Applied every evaluation bar (= once per week for weekly mode)
        # Formula: Cash_buffer *= (1 + APY/52)
        # ===================================================================
        cash_buffer *= weekly_yield_factor

        # ===================================================================
        # STEP 2: HYSTERESIS RESET CHECK
        # If dist_pct drops below hysteresis_reset_pct, reset overbought flag.
        # Per the PDF: "inOverbought remains locked at True until Dist_Pct < +30%"
        # n_harvest is NOT reset here (FIX 2).
        # ===================================================================
        if in_overbought and dist_pct < req.hysteresis_reset_pct:
            in_overbought = False
            active_step_triggers = []
            # NOTE: n_harvest NOT reset (persists until tranche fires)
            # NOTE: peak_cash_buffer NOT cleared (remains anchored)

        # ===================================================================
        # STEP 3: PROFIT HARVEST EVALUATION
        # Condition: dist_pct >= +40% AND in_overbought == False
        # Action: Sell 70% of shares, bank to cash buffer
        # ===================================================================
        action = "NORMAL DCA"
        deposit = req.base_deposit
        tranche_fired = False

        if dist_pct >= req.profit_harvest_dist_pct and not in_overbought:
            # === PROFIT TAKE TRIGGERED ===
            in_overbought = True
            n_harvest += 1  # FIX 2: increment globally

            swing_high_peak = price

            # Sell initial_scale_out_pct of current shares
            shares_to_sell = total_shares * (req.initial_scale_out_pct / 100.0)
            cash_harvested = shares_to_sell * price
            total_shares -= shares_to_sell
            cash_buffer += cash_harvested

            # FIX 1: Anchor peak_cash_buffer at this exact moment (LOCKED)
            peak_cash_buffer = cash_buffer

            # Reset tranche triggers for new cycle
            tranches_triggered = [False] * len(req.tranches.thresholds)

            # Build step scale-out triggers above harvest level
            active_step_triggers = [
                req.profit_harvest_dist_pct + req.step_trigger_increment_pct * (i + 1)
                for i in range(10)
            ]

            action = f"PROFIT TAKE ({req.initial_scale_out_pct:.0f}%)"
            deposit = req.overbought_deposit

        elif in_overbought:
            # Update swing high peak during overbought state
            swing_high_peak = max(swing_high_peak, price)
            deposit = req.overbought_deposit

            # === STEP SCALE-OUTS ===
            # FIX 4: Do NOT update peak_cash_buffer
            for st in list(active_step_triggers):
                if dist_pct >= st:
                    step_shares = total_shares * (req.step_scale_out_pct / 100.0)
                    cash_buffer += step_shares * price
                    total_shares -= step_shares
                    active_step_triggers.remove(st)
                    # peak_cash_buffer NOT updated (FIX 4)
                    action = f"STEP SCALE-OUT @ +{st:.0f}%"
                    break

        # ===================================================================
        # STEP 4: ADAPTIVE TRANCHE THRESHOLD CALCULATION
        # If n_harvest >= 3: tau_1 shifts from -25% to -20%
        # ===================================================================
        tau_1 = (
            req.adaptive.adaptive_tranche_1
            if (req.adaptive.enabled and n_harvest >= req.adaptive.n_harvest_target)
            else req.tranches.thresholds[0]
        )
        active_tranches = [tau_1] + list(req.tranches.thresholds[1:])

        # ===================================================================
        # STEP 5: TRANCHE DIP-BUY EVALUATION
        # Pullback measured from swing_high_peak
        # FIX 1: Deployment uses ANCHORED peak_cash_buffer
        # FIX 2: n_harvest resets to 0 ONLY here
        # ===================================================================
        pullback_pct = (
            ((price - swing_high_peak) / swing_high_peak) * 100.0
            if swing_high_peak > 0
            else 0.0
        )

        for t_idx, trigger_level in enumerate(active_tranches):
            if t_idx < len(tranches_triggered) and pullback_pct <= trigger_level and not tranches_triggered[t_idx]:
                tranches_triggered[t_idx] = True
                tranche_fired = True

                # FIX 2: Reset n_harvest ONLY on tranche buy
                n_harvest = 0

                # FIX 1: Deploy 20% of ANCHORED peak_cash_buffer
                # Formula from PDF: Deploy_Amt = min(0.20 * Cash_Buffer_Peak, Cash_Buffer)
                deploy_from_buffer = min(
                    peak_cash_buffer * (req.tranches.cash_deploy_pct / 100.0),
                    cash_buffer
                )

                # Buy shares with deployed cash + tranche deposit
                tranche_deposit_amount = req.tranche_deposit
                total_buy_amount = deploy_from_buffer + tranche_deposit_amount

                shares_bought = total_buy_amount / price
                total_shares += shares_bought
                cash_buffer -= deploy_from_buffer
                total_out_of_pocket += tranche_deposit_amount

                deposit = tranche_deposit_amount
                action = f"TRANCHE {t_idx + 1} BUY ({trigger_level:.0f}%)"
                break

        # ===================================================================
        # STEP 6: WEEKLY DEPOSIT (buy shares with out-of-pocket contribution)
        # This happens every Friday bar. The deposit amount is determined by:
        #   - $400 if tranche fired (already handled above)
        #   - $100 if in overbought state
        #   - $200 otherwise (normal DCA)
        # ===================================================================
        if not tranche_fired:
            # Normal or overbought deposit — buy shares
            shares_bought = deposit / price
            total_shares += shares_bought
            total_out_of_pocket += deposit

        # ===================================================================
        # STEP 7: PORTFOLIO VALUATION
        # Total Portfolio = (Shares * Friday Close Price) + Cash Buffer
        # ===================================================================
        portfolio_valuation = (total_shares * price) + cash_buffer

        # Format date
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

        # DCA baseline: same weekly $200 deposit
        dca_shares += req.base_deposit / price if price > 0 else 0.0
        dca_out_of_pocket += req.base_deposit

        dca_rows.append({
            "date": date_str,
            "portfolio_valuation": round(dca_shares * price, 2),
            "out_of_pocket_total": round(dca_out_of_pocket, 2),
        })

    # -----------------------------------------------------------------------
    # PERFORMANCE METRICS
    # FIX 3: MDD on portfolio value with warm-up exclusion
    # -----------------------------------------------------------------------
    def compute_metrics(valuations: List[float], invested: List[float], dates: List[str]) -> dict:
        """
        Performance metrics computed on TOTAL PORTFOLIO VALUE.
        
        MDD reflects the cash buffer's capital protection shield:
        - During peaks, 70%+ of capital sits in cash
        - During bears, cash shields portfolio from full asset drawdown
        - Portfolio MDD is dramatically lower than raw TQQQ MDD
        
        Warm-up exclusion: first 52 bars skipped (portfolio too small for
        meaningful percentage drawdowns).
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

        # CAGR: growth of portfolio vs total invested capital
        cagr = ((terminal_val / max(total_invested, 1)) ** (1.0 / years) - 1) * 100

        # FIX 3: MDD on portfolio value with warm-up exclusion
        vals_arr = np.array(valuations, dtype=np.float64)
        warmup_bars = min(52, len(vals_arr) // 4)
        if len(vals_arr) > warmup_bars:
            mdd_vals = vals_arr[warmup_bars:]
        else:
            mdd_vals = vals_arr

        peak_arr = np.maximum.accumulate(mdd_vals)
        dd_arr = np.where(peak_arr > 0, (mdd_vals - peak_arr) / peak_arr, 0.0)
        mdd = float(dd_arr.min()) * 100

        # Sortino ratio (annualized, weekly periods)
        annualization_factor = 52
        if len(vals_arr) > 1:
            period_returns = np.diff(vals_arr) / np.where(vals_arr[:-1] > 0, vals_arr[:-1], 1)
            neg_returns = period_returns[period_returns < 0]
            downside_std = float(np.std(neg_returns)) if len(neg_returns) > 1 else 0.001
            mean_return = float(np.mean(period_returns))
            sortino = (mean_return / downside_std) * np.sqrt(annualization_factor) if downside_std > 0 else 0.0
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

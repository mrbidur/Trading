"""
Dynamic Leveraged ETF Backtesting – FastAPI Backend (v2.3)
===========================================================
HYBRID EXECUTION MODEL:
  - DAILY evaluation: Profit harvests, step scale-outs, tranche buys, hysteresis
  - FRIDAY ONLY: Regular DCA deposits ($200 normal, $100 overbought, $400 tranche)
  - Cash Yield: Compounds WEEKLY (once per Friday bar)
  - MDD: On total portfolio value (shares*price + cash_buffer)

HYSTERESIS LOGIC (confirmed):
  After a profit harvest fires at +40%, the in_overbought flag is set to True.
  Another harvest at +40% can ONLY fire after:
    1. Price pulls back so dist_pct drops BELOW +30% on a daily close
    2. At that point, in_overbought resets to False
    3. Only then can the next +40% crossing trigger a new harvest
  This prevents re-selling during top consolidation where price oscillates
  between +35% and +45% without genuinely reverting to the mean.
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
app = FastAPI(title="ETF Backtest API", version="2.3.0")
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

    # Contributions (deposited on Fridays only)
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

    # Yield (compounds weekly on Fridays)
    cash_yield_apy: float = Field(default=4.5, ge=0.0)


# ---------------------------------------------------------------------------
# Data Fetching & Caching
# ---------------------------------------------------------------------------
def fetch_price_data(ticker: str, start: str, end: str, intraday: bool = False) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance with local disk cache."""
    interval = "1h" if intraday else "1d"
    cache_key = f"prices_v23_{ticker}_{start}_{end}_{interval}"

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
            raise HTTPException(status_code=404, detail=f"No intraday data for {ticker}.")
        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")]
    else:
        df = t.history(start=start, end=end, auto_adjust=True)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No price data found for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise HTTPException(status_code=500, detail=f"Unexpected columns: {list(df.columns)}")

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
# Backtesting State Machine Engine (v2.3 — Hybrid Daily/Friday Model)
# ---------------------------------------------------------------------------
def run_backtest(req: BacktestRequest) -> dict:
    """
    HYBRID EXECUTION MODEL
    =======================
    
    EVALUATED DAILY (immediate execution on signal day):
      - Profit harvest (+40% EMA distance)
      - Step scale-outs (+50%, +60%, +70%, ...)
      - Tranche dip-buys (pullback from swing high)
      - Hysteresis reset (dist < +30%)
    
    EVALUATED FRIDAY ONLY (weekly DCA deposits):
      - $200 base deposit (normal market)
      - $100 overbought deposit (during harvest regime)
      - $400 tranche deposit (on tranche signal week — deposited on Friday)
    
    CASH YIELD:
      - Compounds once per week (on Friday bars) at APY/52
    
    HYSTERESIS RE-ENTRY LOGIC (CONFIRMED):
    ========================================
    Q: Can another +40% harvest fire immediately after the first one?
    A: NO. Here is the exact sequence required:
    
      1. Price crosses +40% above EMA → PROFIT TAKE fires
         in_overbought = True (LOCKED)
      
      2. While in_overbought == True:
         - Another +40% crossing is IGNORED (guard: "and not in_overbought")
         - Step scale-outs at +50%, +60% etc. can still fire
         - Swing high peak continues to update
      
      3. Price must pull back so that dist_pct < +30% on a daily close
         → in_overbought resets to False
      
      4. ONLY NOW can the next +40% crossing trigger a new harvest
    
    This prevents whipsaw re-selling during top consolidation where price
    oscillates between +35% and +45% without genuine mean reversion.
    
    TRANCHE EXECUTION:
      When a pullback threshold is hit on ANY daily close, the tranche buy
      executes immediately at that day's close price. The $400 out-of-pocket
      tranche deposit is counted on that day (not deferred to Friday).
    """
    end_date = req.end_date if req.end_date else date.today().isoformat()
    is_intraday = req.execution_frequency.lower() == "intraday"

    # -----------------------------------------------------------------------
    # FETCH DAILY PRICE DATA
    # -----------------------------------------------------------------------
    raw_df = fetch_price_data(req.ticker, req.start_date, end_date, intraday=is_intraday)

    # -----------------------------------------------------------------------
    # COMPUTE EMA ON DAILY DATA (continuous from Day 1)
    # -----------------------------------------------------------------------
    if is_intraday:
        hourly_ema_span = int(req.ma_period * 6.5)
        raw_df["EMA"] = raw_df["Close"].ewm(span=hourly_ema_span, adjust=False).mean()
    else:
        raw_df["EMA"] = raw_df["Close"].ewm(span=req.ma_period, adjust=False).mean()

    df = raw_df.dropna(subset=["EMA"])

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No data available for {req.ticker} in range {req.start_date} to {end_date}"
        )

    # Weekly cash yield factor
    weekly_yield_factor = 1 + (req.cash_yield_apy / 100.0) / 52.0

    # -----------------------------------------------------------------------
    # STATE VARIABLES
    # -----------------------------------------------------------------------
    in_overbought = False
    n_harvest = 0                    # Only resets on tranche buy
    swing_high_peak = 0.0
    cash_buffer = 0.0
    peak_cash_buffer = 0.0           # Only set at initial harvest (locked)
    total_shares = 0.0
    total_out_of_pocket = 0.0
    tranches_triggered = [False] * len(req.tranches.thresholds)
    active_step_triggers: List[float] = []

    # Weekly tracking
    last_deposit_week: Optional[tuple] = None  # (year, week_number)
    last_yield_week: Optional[tuple] = None

    # Track if a tranche fired this week (to adjust Friday deposit)
    tranche_fired_this_week = False
    current_tranche_week: Optional[tuple] = None

    # DCA baseline
    dca_shares = 0.0
    dca_out_of_pocket = 0.0

    rows: List[dict] = []
    dca_rows: List[dict] = []

    # -----------------------------------------------------------------------
    # MAIN LOOP: Evaluate every daily bar
    # -----------------------------------------------------------------------
    for exec_date, row_data in df.iterrows():
        price = float(row_data["Close"])
        ema = float(row_data["EMA"])

        if price <= 0 or ema <= 0:
            continue

        # Distance from EMA
        dist_pct = ((price - ema) / ema) * 100.0

        current_dt = exec_date.to_pydatetime() if hasattr(exec_date, 'to_pydatetime') else exec_date
        weekday = current_dt.weekday()  # 0=Mon, 4=Fri
        is_friday = (weekday == 4)
        current_week_key = (current_dt.isocalendar()[0], current_dt.isocalendar()[1])

        # Reset weekly tranche flag on new week
        if current_tranche_week != current_week_key:
            tranche_fired_this_week = False
            current_tranche_week = current_week_key

        action = "HOLD"
        deposit = 0.0
        tranche_fired_today = False

        # ===================================================================
        # PHASE 1: WEEKLY CASH YIELD (applied on Fridays only)
        # Cash_buffer *= (1 + APY/52)
        # ===================================================================
        if is_friday and last_yield_week != current_week_key:
            cash_buffer *= weekly_yield_factor
            last_yield_week = current_week_key

        # ===================================================================
        # PHASE 2: HYSTERESIS RESET (evaluated DAILY)
        #
        # CONFIRMED LOGIC:
        # After a profit harvest, in_overbought = True stays LOCKED.
        # It ONLY resets when dist_pct drops BELOW +30% on a daily close.
        # Until this happens, NO new +40% harvest can fire (the condition
        # "and not in_overbought" blocks it).
        #
        # This means: if price goes +40% → harvest → stays at +42% → drops
        # to +38% → rises back to +44%, NO second harvest fires because
        # price never dropped below +30%. Only genuine mean-reversion unlocks.
        # ===================================================================
        if in_overbought and dist_pct < req.hysteresis_reset_pct:
            in_overbought = False
            active_step_triggers = []
            # n_harvest NOT reset (persists until tranche fires)

        # ===================================================================
        # PHASE 3: PROFIT HARVEST (evaluated DAILY — immediate execution)
        #
        # Condition: dist_pct >= +40% AND in_overbought == False
        #
        # The "not in_overbought" guard ensures this can ONLY fire after
        # a hysteresis reset (dist dropped below +30% first). After the
        # initial harvest, in_overbought = True prevents any re-fire until
        # the price genuinely pulls back below the +30% buffer.
        # ===================================================================
        if dist_pct >= req.profit_harvest_dist_pct and not in_overbought:
            in_overbought = True
            n_harvest += 1

            swing_high_peak = price

            # Sell initial_scale_out_pct of current shares
            shares_to_sell = total_shares * (req.initial_scale_out_pct / 100.0)
            cash_harvested = shares_to_sell * price
            total_shares -= shares_to_sell
            cash_buffer += cash_harvested

            # Anchor peak_cash_buffer (LOCKED for this cycle)
            peak_cash_buffer = cash_buffer

            # Reset tranche triggers for new cycle
            tranches_triggered = [False] * len(req.tranches.thresholds)

            # Build step triggers
            active_step_triggers = [
                req.profit_harvest_dist_pct + req.step_trigger_increment_pct * (i + 1)
                for i in range(10)
            ]

            action = f"PROFIT TAKE ({req.initial_scale_out_pct:.0f}%)"

        elif in_overbought:
            # Update swing high peak continuously
            swing_high_peak = max(swing_high_peak, price)

            # STEP SCALE-OUTS (evaluated DAILY — immediate execution)
            for st in list(active_step_triggers):
                if dist_pct >= st:
                    step_shares = total_shares * (req.step_scale_out_pct / 100.0)
                    cash_buffer += step_shares * price
                    total_shares -= step_shares
                    active_step_triggers.remove(st)
                    # peak_cash_buffer NOT updated
                    action = f"STEP SCALE-OUT @ +{st:.0f}%"
                    break

        # ===================================================================
        # PHASE 4: ADAPTIVE TRANCHE THRESHOLD
        # If n_harvest >= 3: τ₁ shifts from -25% to -20%
        # n_harvest only resets when a tranche actually fires
        # ===================================================================
        tau_1 = (
            req.adaptive.adaptive_tranche_1
            if (req.adaptive.enabled and n_harvest >= req.adaptive.n_harvest_target)
            else req.tranches.thresholds[0]
        )
        active_tranches = [tau_1] + list(req.tranches.thresholds[1:])

        # ===================================================================
        # PHASE 5: TRANCHE DIP-BUY (evaluated DAILY — immediate execution)
        #
        # When a pullback threshold is met on ANY daily close, the tranche
        # executes immediately at that day's close price. The $400 deposit
        # is contributed on signal day (not deferred to Friday).
        # ===================================================================
        pullback_pct = (
            ((price - swing_high_peak) / swing_high_peak) * 100.0
            if swing_high_peak > 0
            else 0.0
        )

        for t_idx, trigger_level in enumerate(active_tranches):
            if t_idx < len(tranches_triggered) and pullback_pct <= trigger_level and not tranches_triggered[t_idx]:
                tranches_triggered[t_idx] = True
                tranche_fired_today = True
                tranche_fired_this_week = True

                # Reset n_harvest ONLY here
                n_harvest = 0

                # Deploy 20% of ANCHORED peak_cash_buffer (capped at available cash)
                deploy_from_buffer = min(
                    peak_cash_buffer * (req.tranches.cash_deploy_pct / 100.0),
                    cash_buffer
                )

                # Tranche deposit ($400) is out-of-pocket on signal day
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
        # PHASE 6: FRIDAY DCA DEPOSIT (only regular weekly contributions)
        #
        # Friday entries are STRICTLY limited to regular DCA purchases:
        #   - $200 if normal state
        #   - $100 if overbought state
        #   - $0 if a tranche already fired this week (already deposited $400)
        #
        # Profit harvests and tranche buys are NOT evaluated here — they
        # execute daily in Phases 3 and 5 above.
        # ===================================================================
        if is_friday and last_deposit_week != current_week_key:
            last_deposit_week = current_week_key

            if tranche_fired_this_week:
                # Tranche already contributed $400 this week, no additional DCA
                friday_deposit = 0.0
            elif in_overbought:
                friday_deposit = req.overbought_deposit
            else:
                friday_deposit = req.base_deposit

            if friday_deposit > 0:
                shares_bought = friday_deposit / price
                total_shares += shares_bought
                total_out_of_pocket += friday_deposit
                deposit += friday_deposit
                if action == "HOLD":
                    action = "WEEKLY DCA" if not in_overbought else "WEEKLY DCA (OB)"

            # DCA baseline: always $200/week on Friday
            dca_shares += req.base_deposit / price if price > 0 else 0.0
            dca_out_of_pocket += req.base_deposit

        # ===================================================================
        # PHASE 7: PORTFOLIO VALUATION
        # ===================================================================
        portfolio_valuation = (total_shares * price) + cash_buffer

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
            "deposit": round(deposit, 2),
            "cash_buffer": round(cash_buffer, 2),
            "total_shares": round(total_shares, 6),
            "portfolio_valuation": round(portfolio_valuation, 2),
            "out_of_pocket_total": round(total_out_of_pocket, 2),
        })

        dca_rows.append({
            "date": date_str,
            "portfolio_valuation": round(dca_shares * price, 2),
            "out_of_pocket_total": round(dca_out_of_pocket, 2),
        })

    # -----------------------------------------------------------------------
    # PERFORMANCE METRICS
    # -----------------------------------------------------------------------
    def compute_metrics(valuations: List[float], invested: List[float], dates: List[str]) -> dict:
        """
        MDD on total portfolio value (shares*price + cash_buffer).
        Warm-up exclusion: first 52 bars skipped.
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

        cagr = ((terminal_val / max(total_invested, 1)) ** (1.0 / years) - 1) * 100

        # MDD with warm-up exclusion
        vals_arr = np.array(valuations, dtype=np.float64)
        warmup_bars = min(252, len(vals_arr) // 4)  # ~1 year of daily bars
        if len(vals_arr) > warmup_bars:
            mdd_vals = vals_arr[warmup_bars:]
        else:
            mdd_vals = vals_arr

        peak_arr = np.maximum.accumulate(mdd_vals)
        dd_arr = np.where(peak_arr > 0, (mdd_vals - peak_arr) / peak_arr, 0.0)
        mdd = float(dd_arr.min()) * 100

        # Sortino (weekly-sampled for consistency)
        annualization_factor = 52
        if len(vals_arr) > 5:
            # Sample to weekly for Sortino
            step = max(1, len(vals_arr) // (int(years * 52) + 1))
            weekly_vals = vals_arr[::step]
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
        raise HTTPException(status_code=500, detail=f"Backtest engine error: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Export engine error: {str(e)}")

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
            r["date"], r["close"], r["ema"], r["dist_pct"],
            str(r["overbought"]).upper(), r["n_harvest"],
            r["swing_high_peak"], r["pullback_pct"], r["tranche_1_trigger"],
            f'"{r["action"]}"', r["deposit"], r["cash_buffer"],
            r["total_shares"], r["portfolio_valuation"], r["out_of_pocket_total"],
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

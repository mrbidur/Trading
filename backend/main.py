"""
Dynamic Leveraged ETF Backtesting – FastAPI Backend
====================================================
State-machine-based allocation strategy engine with full CSV export.
"""
from __future__ import annotations

import io
import os
from datetime import date, datetime
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
    execution_frequency: str = Field(default="weekly")

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
def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV data from Yahoo Finance with local disk cache."""
    cache_key = f"prices_{ticker}_{start}_{end}"
    if cache_key in cache:
        raw = cache[cache_key]
        df = pd.read_json(io.StringIO(raw), orient="split")
        df.index = pd.to_datetime(df.index)
        return df

    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No price data found for {ticker}")

    df = df[["Close"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    cache.set(cache_key, df.to_json(orient="split"), expire=CACHE_TTL)
    return df


def resample_to_frequency(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Resample daily prices to the user-selected execution frequency."""
    freq_map = {"weekly": "W-FRI", "monthly": "ME", "daily": None}
    freq = freq_map.get(frequency.lower())
    if freq is None:
        return df
    resampled = df.resample(freq).last().dropna()
    return resampled


# ---------------------------------------------------------------------------
# Backtesting State Machine Engine
# ---------------------------------------------------------------------------
def run_backtest(req: BacktestRequest) -> dict:
    """Run the full state-machine backtest and return structured results."""
    end_date = req.end_date if req.end_date else date.today().isoformat()
    raw_df = fetch_price_data(req.ticker, req.start_date, end_date)

    # Compute EMA on daily data, then resample
    raw_df["EMA"] = raw_df["Close"].ewm(span=req.ma_period, adjust=False).mean()
    df = resample_to_frequency(raw_df, req.execution_frequency)
    df = df.dropna(subset=["EMA"])

    # Yield per period
    n_periods_per_year = {"weekly": 52, "monthly": 12, "daily": 252}
    ppy = n_periods_per_year.get(req.execution_frequency.lower(), 52)
    yield_per_period = (1 + req.cash_yield_apy / 100.0) ** (1.0 / ppy) - 1

    # State variables
    in_overbought = False
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

    for exec_date, row_data in df.iterrows():
        price = float(row_data["Close"])
        ema = float(row_data["EMA"])

        if price <= 0 or ema <= 0:
            continue

        dist_pct = ((price - ema) / ema) * 100.0

        # 1. Apply periodic cash yield on uninvested cash
        cash_buffer *= (1 + yield_per_period)

        # 2. Hysteresis reset
        if dist_pct < req.hysteresis_reset_pct:
            in_overbought = False
            active_step_triggers = []

        action = "NORMAL DCA"
        deposit = req.base_deposit
        tranche_fired = False

        # 3. Overbought profit-take
        if dist_pct >= req.profit_harvest_dist_pct and not in_overbought:
            in_overbought = True
            n_harvest += 1
            swing_high_peak = price

            shares_to_sell = total_shares * (req.initial_scale_out_pct / 100.0)
            cash_harvested = shares_to_sell * price
            total_shares -= shares_to_sell
            cash_buffer += cash_harvested
            peak_cash_buffer = cash_buffer

            action = f"PROFIT TAKE ({req.initial_scale_out_pct:.0f}%)"
            deposit = req.overbought_deposit
            tranches_triggered = [False] * len(req.tranches.thresholds)

            # Build step triggers
            active_step_triggers = [
                req.profit_harvest_dist_pct + req.step_trigger_increment_pct * (i + 1)
                for i in range(10)
            ]

        elif in_overbought:
            deposit = req.overbought_deposit
            swing_high_peak = max(swing_high_peak, price)

            # Step scale-outs
            for st in list(active_step_triggers):
                if dist_pct >= st:
                    step_shares = total_shares * (req.step_scale_out_pct / 100.0)
                    cash_buffer += step_shares * price
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

        # 5. Pullback from swing high
        pullback_pct = (
            ((price - swing_high_peak) / swing_high_peak) * 100.0
            if swing_high_peak > 0
            else 0.0
        )

        # 6. Tranche dip-buy
        for t_idx, trigger_level in enumerate(active_tranches):
            if t_idx < len(tranches_triggered) and pullback_pct <= trigger_level and not tranches_triggered[t_idx]:
                tranches_triggered[t_idx] = True
                deposit = req.tranche_deposit
                n_harvest = 0
                tranche_fired = True

                cash_to_deploy = (peak_cash_buffer * (req.tranches.cash_deploy_pct / 100.0)) + deposit
                actual_deploy = min(cash_buffer + deposit, cash_to_deploy)

                shares_bought = actual_deploy / price
                total_shares += shares_bought
                cash_buffer = max(0.0, cash_buffer + deposit - actual_deploy)
                total_out_of_pocket += deposit
                action = f"TRANCHE {t_idx + 1} BUY ({trigger_level:.0f}%)"
                break

        # 7. Normal DCA (when no tranche fired)
        if not tranche_fired:
            shares_bought = deposit / price
            total_shares += shares_bought
            total_out_of_pocket += deposit

        portfolio_valuation = (total_shares * price) + cash_buffer

        rows.append({
            "date": exec_date.strftime("%Y-%m-%d"),
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

        # DCA baseline
        dca_shares += req.base_deposit / price
        dca_out_of_pocket += req.base_deposit
        dca_rows.append({
            "date": exec_date.strftime("%Y-%m-%d"),
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
        start_dt = datetime.strptime(dates[0], "%Y-%m-%d")
        end_dt = datetime.strptime(dates[-1], "%Y-%m-%d")
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
    return run_backtest(req)


@app.post("/export-csv")
def export_csv(req: BacktestRequest):
    """Run backtest and return the full audit trail as a downloadable CSV."""
    result = run_backtest(req)
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
    filename = f"{req.ticker}_backtest_{req.start_date}_to_{rows[-1]['date']}.csv"
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

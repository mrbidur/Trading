"""
QQQ/TQQQ Dynamic Asset-Switching Backtesting Engine
=====================================================
Strategy: Multi-tier drawdown switching from QQQ to TQQQ based on
200-day EMA distance. When QQQ falls below defined % thresholds from
its 200 EMA, systematically transfer capital from QQQ into TQQQ.
Reset when QQQ recovers above the 200 EMA.

Author: Quantitative Trading Engine
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import os


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class StrategyConfig:
    """All configurable parameters for the backtest."""
    
    # Assets
    base_asset: str = "QQQ"
    leveraged_asset: str = "TQQQ"
    
    # Timeline
    start_date: str = "2010-02-12"  # TQQQ inception
    end_date: str = ""              # Empty = today
    
    # Capital
    initial_capital: float = 100_000.0
    
    # EMA
    ema_period: int = 200
    
    # Drawdown tiers: distance below 200 EMA (%) that triggers a transfer
    # Each tier transfers a slice of QQQ holdings into TQQQ
    drawdown_tiers: List[float] = field(default_factory=lambda: [-20.0, -30.0, -40.0, -50.0])
    
    # Transfer percentage: how much of remaining QQQ to move per tier
    # "proportional" = 25% of current QQQ per tier (4 tiers = up to 100%)
    transfer_mode: str = "equal_slice"  # "equal_slice" or "proportional"
    transfer_pct_per_tier: float = 25.0  # % of QQQ holdings to transfer per tier
    
    # Reset: when EMA distance goes above this, reset all tier flags
    reset_threshold: float = 0.0  # 0% = above EMA
    
    # Output
    output_csv: str = "strategy_backtest_results.csv"


# =============================================================================
# DATA FETCHER
# =============================================================================
class DataFetcher:
    """Fetches and aligns daily price data for multiple tickers."""
    
    @staticmethod
    def fetch(tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch daily close prices for all tickers and align on common dates.
        
        Returns DataFrame with columns: '{TICKER}_close' for each ticker.
        """
        end_date = end if end else date.today().isoformat()
        
        all_data = {}
        for ticker in tickers:
            print(f"  Fetching {ticker} data...")
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end_date, auto_adjust=True)
            
            if df is None or df.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Handle MultiIndex columns (yfinance >= 0.2.40)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure timezone-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            all_data[f"{ticker}_close"] = df["Close"]
        
        # Combine into single DataFrame aligned on dates
        combined = pd.DataFrame(all_data)
        combined.index = pd.to_datetime(combined.index)
        combined.index.name = "Date"
        
        # Drop any rows where either asset has NaN (alignment)
        combined = combined.dropna()
        
        print(f"  Data aligned: {len(combined)} trading days ({combined.index[0].strftime('%Y-%m-%d')} to {combined.index[-1].strftime('%Y-%m-%d')})")
        return combined


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================
class BacktestEngine:
    """
    Core backtesting engine for QQQ/TQQQ dynamic switching strategy.
    
    Strategy Logic:
    1. Start 100% in QQQ
    2. Calculate daily EMA distance: ((QQQ_price - EMA) / EMA) * 100
    3. When distance crosses below each drawdown tier threshold,
       transfer a slice of QQQ holdings into TQQQ
    4. When QQQ recovers above EMA (distance > 0), reset all tier flags
    5. Track portfolio value daily: (QQQ_shares * QQQ_price) + (TQQQ_shares * TQQQ_price)
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
    
    def run(self) -> pd.DataFrame:
        """Execute the full backtest and return results DataFrame."""
        print("\n" + "=" * 70)
        print("QQQ/TQQQ DYNAMIC SWITCHING BACKTEST ENGINE")
        print("=" * 70)
        
        # Step 1: Fetch data
        print("\n[1/4] Fetching price data...")
        self.data = DataFetcher.fetch(
            tickers=[self.config.base_asset, self.config.leveraged_asset],
            start=self.config.start_date,
            end=self.config.end_date,
        )
        
        # Step 2: Calculate EMA
        print("[2/4] Computing 200-day EMA...")
        base_col = f"{self.config.base_asset}_close"
        self.data["EMA_200"] = self.data[base_col].ewm(
            span=self.config.ema_period, adjust=False
        ).mean()
        
        # Step 3: Calculate EMA distance
        self.data["EMA_dist_pct"] = (
            (self.data[base_col] - self.data["EMA_200"]) / self.data["EMA_200"]
        ) * 100.0
        
        # Step 4: Run strategy simulation
        print("[3/4] Running strategy simulation...")
        self.results = self._simulate()
        
        # Step 5: Export
        print("[4/4] Exporting results...")
        self._export_csv()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _simulate(self) -> pd.DataFrame:
        """Run the day-by-day simulation."""
        cfg = self.config
        base_col = f"{cfg.base_asset}_close"
        lev_col = f"{cfg.leveraged_asset}_close"
        
        # Initial state: all capital in QQQ
        first_price = float(self.data[base_col].iloc[0])
        qqq_shares = cfg.initial_capital / first_price
        tqqq_shares = 0.0
        
        # Tier tracking: which tiers have been triggered in current cycle
        n_tiers = len(cfg.drawdown_tiers)
        tiers_triggered = [False] * n_tiers
        
        # Results accumulator
        rows = []
        
        for i, (dt, row) in enumerate(self.data.iterrows()):
            qqq_price = float(row[base_col])
            tqqq_price = float(row[lev_col])
            ema_200 = float(row["EMA_200"])
            ema_dist_pct = float(row["EMA_dist_pct"])
            
            action = "HOLD"
            transfer_amount = 0.0
            
            # ----------------------------------------------------------
            # RESET: If QQQ is above EMA, reset all tier flags
            # ----------------------------------------------------------
            if ema_dist_pct > cfg.reset_threshold:
                if any(tiers_triggered):
                    action = "RESET"
                tiers_triggered = [False] * n_tiers
            
            # ----------------------------------------------------------
            # CHECK DRAWDOWN TIERS (only fire each once per cycle)
            # ----------------------------------------------------------
            else:
                for t_idx, tier_threshold in enumerate(cfg.drawdown_tiers):
                    if ema_dist_pct <= tier_threshold and not tiers_triggered[t_idx]:
                        # Tier triggered — transfer QQQ to TQQQ
                        tiers_triggered[t_idx] = True
                        
                        if qqq_shares > 0:
                            # Calculate transfer amount
                            if cfg.transfer_mode == "equal_slice":
                                # Transfer fixed % of ORIGINAL QQQ position per tier
                                shares_to_transfer = qqq_shares * (cfg.transfer_pct_per_tier / 100.0)
                            else:  # proportional
                                shares_to_transfer = qqq_shares * (cfg.transfer_pct_per_tier / 100.0)
                            
                            # Sell QQQ shares, buy TQQQ with proceeds
                            cash_from_sale = shares_to_transfer * qqq_price
                            tqqq_bought = cash_from_sale / tqqq_price
                            
                            qqq_shares -= shares_to_transfer
                            tqqq_shares += tqqq_bought
                            transfer_amount = cash_from_sale
                            
                            action = f"TIER {t_idx + 1} SWITCH ({tier_threshold:.0f}%)"
                        break  # Only one tier per day
            
            # ----------------------------------------------------------
            # CALCULATE PORTFOLIO VALUE
            # ----------------------------------------------------------
            qqq_value = qqq_shares * qqq_price
            tqqq_value = tqqq_shares * tqqq_price
            total_value = qqq_value + tqqq_value
            
            # Benchmark: 100% QQQ buy-and-hold
            benchmark_shares = cfg.initial_capital / float(self.data[base_col].iloc[0])
            benchmark_value = benchmark_shares * qqq_price
            
            # Record
            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "qqq_price": round(qqq_price, 4),
                "tqqq_price": round(tqqq_price, 4),
                "ema_200": round(ema_200, 4),
                "ema_dist_pct": round(ema_dist_pct, 4),
                "qqq_shares": round(qqq_shares, 6),
                "tqqq_shares": round(tqqq_shares, 6),
                "qqq_value": round(qqq_value, 2),
                "tqqq_value": round(tqqq_value, 2),
                "total_portfolio": round(total_value, 2),
                "benchmark_value": round(benchmark_value, 2),
                "tiers_active": sum(tiers_triggered),
                "action": action,
                "transfer_amount": round(transfer_amount, 2),
            })
        
        return pd.DataFrame(rows)
    
    def _export_csv(self):
        """Export results to CSV."""
        output_path = os.path.join(os.path.dirname(__file__), self.config.output_csv)
        self.results.to_csv(output_path, index=False)
        print(f"  Exported: {output_path} ({len(self.results)} rows)")
    
    def _print_summary(self):
        """Print performance summary."""
        if self.results is None or self.results.empty:
            return
        
        first = self.results.iloc[0]
        last = self.results.iloc[-1]
        
        initial = self.config.initial_capital
        final_strategy = last["total_portfolio"]
        final_benchmark = last["benchmark_value"]
        
        years = (pd.to_datetime(last["date"]) - pd.to_datetime(first["date"])).days / 365.25
        
        # CAGR
        cagr_strategy = ((final_strategy / initial) ** (1 / years) - 1) * 100
        cagr_benchmark = ((final_benchmark / initial) ** (1 / years) - 1) * 100
        
        # Max Drawdown (on portfolio value)
        portfolio_values = self.results["total_portfolio"].values
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        mdd_strategy = drawdown.min() * 100
        
        benchmark_values = self.results["benchmark_value"].values
        peak_b = np.maximum.accumulate(benchmark_values)
        drawdown_b = (benchmark_values - peak_b) / peak_b
        mdd_benchmark = drawdown_b.min() * 100
        
        # Count events
        switches = self.results[self.results["action"].str.contains("TIER")].shape[0]
        resets = self.results[self.results["action"] == "RESET"].shape[0]
        
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"\n{'Metric':<30} {'Strategy':<20} {'Benchmark (QQQ)':<20}")
        print("-" * 70)
        print(f"{'Initial Capital':<30} ${initial:>15,.2f}   ${initial:>15,.2f}")
        print(f"{'Final Value':<30} ${final_strategy:>15,.2f}   ${final_benchmark:>15,.2f}")
        print(f"{'Total Return':<30} {((final_strategy/initial)-1)*100:>14.2f}%   {((final_benchmark/initial)-1)*100:>14.2f}%")
        print(f"{'CAGR':<30} {cagr_strategy:>14.2f}%   {cagr_benchmark:>14.2f}%")
        print(f"{'Max Drawdown':<30} {mdd_strategy:>14.2f}%   {mdd_benchmark:>14.2f}%")
        print(f"{'Duration':<30} {years:>14.1f} yrs")
        print(f"\n{'Tier Switches':<30} {switches}")
        print(f"{'Reset Events':<30} {resets}")
        print(f"{'Final QQQ Shares':<30} {last['qqq_shares']:,.2f}")
        print(f"{'Final TQQQ Shares':<30} {last['tqqq_shares']:,.2f}")
        print(f"{'Final QQQ Allocation':<30} {last['qqq_value']/last['total_portfolio']*100:.1f}%")
        print(f"{'Final TQQQ Allocation':<30} {last['tqqq_value']/last['total_portfolio']*100:.1f}%")
        print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Run the backtest with default configuration."""
    
    # Create configuration (all parameters adjustable here)
    config = StrategyConfig(
        base_asset="QQQ",
        leveraged_asset="TQQQ",
        start_date="2010-02-12",
        end_date="",  # Today
        initial_capital=100_000.0,
        ema_period=200,
        drawdown_tiers=[-20.0, -30.0, -40.0, -50.0],
        transfer_mode="equal_slice",
        transfer_pct_per_tier=25.0,  # 25% of QQQ per tier (4 tiers = up to 100%)
        reset_threshold=0.0,  # Reset when QQQ > EMA
        output_csv="strategy_backtest_results.csv",
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run()
    
    print(f"\nResults saved to: {config.output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()

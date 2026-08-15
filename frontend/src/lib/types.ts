export interface BacktestConfig {
  ticker: string;
  start_date: string;
  end_date: string;
  execution_frequency: string;
  base_deposit: number;
  overbought_deposit: number;
  tranche_deposit: number;
  ma_period: number;
  profit_harvest_dist_pct: number;
  hysteresis_reset_pct: number;
  initial_scale_out_pct: number;
  step_trigger_increment_pct: number;
  step_scale_out_pct: number;
  tranches: {
    thresholds: number[];
    cash_deploy_pct: number;
  };
  adaptive: {
    enabled: boolean;
    n_harvest_target: number;
    adaptive_tranche_1: number;
  };
  cash_yield_apy: number;
}

export interface BacktestRow {
  date: string;
  close: number;
  ema: number;
  dist_pct: number;
  overbought: boolean;
  n_harvest: number;
  swing_high_peak: number;
  pullback_pct: number;
  tranche_1_trigger: number;
  action: string;
  deposit: number;
  cash_buffer: number;
  total_shares: number;
  portfolio_valuation: number;
  out_of_pocket_total: number;
}

export interface DCARow {
  date: string;
  portfolio_valuation: number;
  out_of_pocket_total: number;
}

export interface Metrics {
  terminal_value: number;
  total_invested: number;
  cagr: number;
  mdd: number;
  sortino: number;
  years: number;
}

export interface BacktestResult {
  ticker: string;
  rows: BacktestRow[];
  dca_rows: DCARow[];
  strategy_metrics: Metrics;
  dca_metrics: Metrics;
}

export const DEFAULT_CONFIG: BacktestConfig = {
  ticker: "TQQQ",
  start_date: "2010-02-12",
  end_date: "",
  execution_frequency: "daily",
  base_deposit: 200,
  overbought_deposit: 100,
  tranche_deposit: 400,
  ma_period: 200,
  profit_harvest_dist_pct: 40,
  hysteresis_reset_pct: 30,
  initial_scale_out_pct: 70,
  step_trigger_increment_pct: 10,
  step_scale_out_pct: 10,
  tranches: {
    thresholds: [-25, -35, -45, -55, -65],
    cash_deploy_pct: 20,
  },
  adaptive: {
    enabled: true,
    n_harvest_target: 3,
    adaptive_tranche_1: -20,
  },
  cash_yield_apy: 4.5,
};

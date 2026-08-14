"use client";

import React from "react";
import { BacktestConfig } from "@/lib/types";
import { Settings, TrendingUp, Shield, Layers, Zap, Percent, Info } from "lucide-react";

interface ConfigPanelProps {
  config: BacktestConfig;
  onChange: (config: BacktestConfig) => void;
  onRun: () => void;
  loading: boolean;
}

function InputField({
  label,
  value,
  onChange,
  type = "text",
  step,
  min,
  max,
  suffix,
  hint,
}: {
  label: string;
  value: string | number;
  onChange: (val: string) => void;
  type?: string;
  step?: number;
  min?: number;
  max?: number;
  suffix?: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-muted-foreground font-medium">{label}</label>
      <div className="relative">
        <input
          type={type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          step={step}
          min={min}
          max={max}
          className="w-full bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 pr-8"
        />
        {suffix && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
            {suffix}
          </span>
        )}
      </div>
      {hint && <span className="text-[10px] text-muted-foreground/70 leading-tight">{hint}</span>}
    </div>
  );
}

function SectionHeader({ icon: Icon, title }: { icon: React.ElementType; title: string }) {
  return (
    <div className="flex items-center gap-2 pt-4 pb-2 border-b border-border">
      <Icon className="w-4 h-4 text-primary" />
      <h3 className="text-sm font-semibold text-foreground">{title}</h3>
    </div>
  );
}

function InfoBanner({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-2 bg-primary/5 border border-primary/20 rounded-md p-2 mt-1">
      <Info className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
      <p className="text-[10px] text-muted-foreground leading-relaxed">{children}</p>
    </div>
  );
}

export default function ConfigPanel({ config, onChange, onRun, loading }: ConfigPanelProps) {
  const update = (partial: Partial<BacktestConfig>) => {
    onChange({ ...config, ...partial });
  };

  const presetDate = (years: number | "max") => {
    if (years === "max") {
      update({ start_date: "2010-01-01", end_date: "" });
    } else {
      const start = new Date();
      start.setFullYear(start.getFullYear() - years);
      update({ start_date: start.toISOString().split("T")[0], end_date: "" });
    }
  };

  return (
    <aside className="w-80 min-w-80 h-full overflow-y-auto border-r border-border bg-card p-4 flex flex-col gap-2">
      <div className="flex items-center gap-2 pb-2 border-b border-border">
        <Settings className="w-5 h-5 text-primary" />
        <h2 className="text-base font-bold">Strategy Configuration</h2>
      </div>

      {/* ====== ASSET & TIMEFRAME ====== */}
      <SectionHeader icon={TrendingUp} title="Asset & Timeframe" />
      <InputField
        label="Ticker Symbol"
        value={config.ticker}
        onChange={(v) => update({ ticker: v.toUpperCase() })}
      />
      <div className="grid grid-cols-2 gap-2">
        <InputField
          label="Start Date"
          type="date"
          value={config.start_date}
          onChange={(v) => update({ start_date: v })}
        />
        <InputField
          label="End Date"
          type="date"
          value={config.end_date}
          onChange={(v) => update({ end_date: v })}
        />
      </div>
      <div className="flex flex-wrap gap-1 mt-1">
        {[1, 3, 5, 10].map((y) => (
          <button
            key={y}
            onClick={() => presetDate(y)}
            className="text-xs px-2 py-0.5 rounded bg-secondary hover:bg-primary/20 transition"
          >
            {y}Y
          </button>
        ))}
        <button
          onClick={() => presetDate("max")}
          className="text-xs px-2 py-0.5 rounded bg-secondary hover:bg-primary/20 transition"
        >
          Max
        </button>
      </div>

      {/* SCAN FREQUENCY (decoupled from deposits) */}
      <div className="flex flex-col gap-1">
        <label className="text-xs text-muted-foreground font-medium">
          Price Scan Frequency
        </label>
        <select
          value={config.execution_frequency}
          onChange={(e) => update({ execution_frequency: e.target.value })}
          className="bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
        >
          <option value="daily">Daily Close (exits evaluated daily)</option>
          <option value="intraday">Intraday Hourly (exits evaluated every hour)</option>
        </select>
      </div>
      <InfoBanner>
        <strong>Decoupled model:</strong> Deposits are always weekly (Friday).
        Profit harvests &amp; tranche buys trigger continuously based on scan frequency.
        Cash yield compounds weekly at APY/52.
      </InfoBanner>

      {/* ====== WEEKLY CONTRIBUTIONS ====== */}
      <SectionHeader icon={Layers} title="Weekly Contributions" />
      <InputField
        label="Base Weekly Deposit"
        type="number"
        value={config.base_deposit}
        onChange={(v) => update({ base_deposit: parseFloat(v) || 0 })}
        step={50}
        min={0}
        suffix="$"
        hint="Deposited every Friday during normal market conditions"
      />
      <InputField
        label="Overbought Weekly Deposit"
        type="number"
        value={config.overbought_deposit}
        onChange={(v) => update({ overbought_deposit: parseFloat(v) || 0 })}
        step={50}
        min={0}
        suffix="$"
        hint="Reduced deposit during overbought (harvesting) state"
      />
      <InputField
        label="Tranche Buy Deposit"
        type="number"
        value={config.tranche_deposit}
        onChange={(v) => update({ tranche_deposit: parseFloat(v) || 0 })}
        step={50}
        min={0}
        suffix="$"
        hint="Extra capital deployed when dip thresholds are hit"
      />

      {/* ====== MA & HARVEST (continuous evaluation) ====== */}
      <SectionHeader icon={Zap} title="Profit Harvest (Continuous)" />
      <InputField
        label="EMA Period (days)"
        type="number"
        value={config.ma_period}
        onChange={(v) => update({ ma_period: parseInt(v) || 200 })}
        min={5}
        hint="200-day EMA. For intraday, auto-converts to hourly equivalent."
      />
      <InputField
        label="Profit Harvest Distance"
        type="number"
        value={config.profit_harvest_dist_pct}
        onChange={(v) => update({ profit_harvest_dist_pct: parseFloat(v) || 40 })}
        step={5}
        suffix="%"
        hint="+40% above EMA triggers profit take. ~13% on underlying QQQ."
      />
      <InputField
        label="Hysteresis Reset Buffer"
        type="number"
        value={config.hysteresis_reset_pct}
        onChange={(v) => update({ hysteresis_reset_pct: parseFloat(v) || 30 })}
        step={5}
        suffix="%"
        hint="Must drop below this level before harvest can re-arm (prevents whipsaw)"
      />
      <InputField
        label="Initial Scale-Out Share %"
        type="number"
        value={config.initial_scale_out_pct}
        onChange={(v) => update({ initial_scale_out_pct: parseFloat(v) || 70 })}
        step={5}
        min={1}
        max={100}
        suffix="%"
        hint="Percentage of shares sold on first profit harvest"
      />
      <InputField
        label="Step Trigger Increment"
        type="number"
        value={config.step_trigger_increment_pct}
        onChange={(v) => update({ step_trigger_increment_pct: parseFloat(v) || 10 })}
        step={5}
        suffix="%"
        hint="Each +10% above harvest level triggers another scale-out"
      />
      <InputField
        label="Step Scale-Out Share %"
        type="number"
        value={config.step_scale_out_pct}
        onChange={(v) => update({ step_scale_out_pct: parseFloat(v) || 10 })}
        step={5}
        min={1}
        max={100}
        suffix="%"
        hint="Additional shares sold at each step level"
      />

      {/* ====== TRANCHES (continuous evaluation) ====== */}
      <SectionHeader icon={Shield} title="Tranche Dip-Buy (Continuous)" />
      <div className="flex flex-col gap-1">
        <label className="text-xs text-muted-foreground font-medium">
          Drawdown Thresholds from Swing High (%)
        </label>
        <input
          value={config.tranches.thresholds.join(", ")}
          onChange={(e) => {
            const thresholds = e.target.value
              .split(",")
              .map((v) => parseFloat(v.trim()))
              .filter((v) => !isNaN(v));
            update({ tranches: { ...config.tranches, thresholds } });
          }}
          className="w-full bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          placeholder="-25, -35, -45, -55, -65"
        />
        <span className="text-[10px] text-muted-foreground/70">
          Tiered pullback levels from peak that deploy cash buffer
        </span>
      </div>
      <InputField
        label="Cash Deploy per Tranche"
        type="number"
        value={config.tranches.cash_deploy_pct}
        onChange={(v) =>
          update({ tranches: { ...config.tranches, cash_deploy_pct: parseFloat(v) || 20 } })
        }
        step={5}
        min={1}
        max={100}
        suffix="%"
        hint="% of peak cash buffer deployed at each tranche level"
      />

      {/* ====== ADAPTIVE MODE ====== */}
      <div className="flex items-center gap-2 mt-3">
        <input
          type="checkbox"
          id="adaptive-toggle"
          checked={config.adaptive.enabled}
          onChange={(e) =>
            update({ adaptive: { ...config.adaptive, enabled: e.target.checked } })
          }
          className="w-4 h-4 rounded border-border bg-secondary accent-primary"
        />
        <label htmlFor="adaptive-toggle" className="text-xs text-muted-foreground font-medium cursor-pointer">
          Adaptive Threshold Mode
        </label>
      </div>
      {config.adaptive.enabled && (
        <div className="ml-4 flex flex-col gap-2 border-l-2 border-primary/30 pl-3">
          <InputField
            label="Consecutive Harvest Target (N)"
            type="number"
            value={config.adaptive.n_harvest_target}
            onChange={(v) =>
              update({
                adaptive: { ...config.adaptive, n_harvest_target: parseInt(v) || 3 },
              })
            }
            min={1}
            hint="After N harvests without a dip, shift Tranche 1 deeper"
          />
          <InputField
            label="Adaptive Tranche 1 Trigger"
            type="number"
            value={config.adaptive.adaptive_tranche_1}
            onChange={(v) =>
              update({
                adaptive: { ...config.adaptive, adaptive_tranche_1: parseFloat(v) || -20 },
              })
            }
            step={5}
            suffix="%"
            hint="Adjusted T1 threshold after N consecutive harvests"
          />
        </div>
      )}

      {/* ====== CASH YIELD ====== */}
      <SectionHeader icon={Percent} title="Cash Yield (Weekly Compound)" />
      <InputField
        label="Cash Buffer APY"
        type="number"
        value={config.cash_yield_apy}
        onChange={(v) => update({ cash_yield_apy: parseFloat(v) || 0 })}
        step={0.5}
        min={0}
        suffix="%"
        hint="Compounds weekly: Cash *= (1 + APY/52). Models T-bill sweep yield."
      />

      {/* ====== RUN BUTTON ====== */}
      <button
        onClick={onRun}
        disabled={loading}
        className="mt-4 w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-2.5 rounded-lg transition disabled:opacity-50 flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Running Backtest...
          </>
        ) : (
          <>
            <TrendingUp className="w-4 h-4" />
            Run Backtest
          </>
        )}
      </button>
    </aside>
  );
}

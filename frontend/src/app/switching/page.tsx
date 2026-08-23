"use client";

import React, { useState, useCallback } from "react";
import { Activity, Download, TrendingUp, TrendingDown, BarChart3, Shield } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ComposedChart, Area,
} from "recharts";

interface SwitchingConfig {
  base_asset: string;
  leveraged_asset: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  ema_period: number;
  drawdown_tiers: number[];
  transfer_pct_per_tier: number;
  reset_threshold: number;
}

const DEFAULT_CONFIG: SwitchingConfig = {
  base_asset: "QQQ",
  leveraged_asset: "TQQQ",
  start_date: "2010-02-12",
  end_date: "",
  initial_capital: 100000,
  ema_period: 200,
  drawdown_tiers: [-20, -30, -40, -50],
  transfer_pct_per_tier: 25,
  reset_threshold: 0,
};

function formatDollar(v: number) {
  if (v >= 1000000) return `$${(v / 1000000).toFixed(1)}M`;
  if (v >= 1000) return `$${(v / 1000).toFixed(0)}K`;
  return `$${v.toFixed(0)}`;
}

export default function SwitchingPage() {
  const [config, setConfig] = useState<SwitchingConfig>(DEFAULT_CONFIG);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/switching-backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [config]);

  const update = (p: Partial<SwitchingConfig>) => setConfig({ ...config, ...p });

  // Chart data (sampled for performance)
  const chartData = result?.rows
    ? (() => {
        const step = Math.max(1, Math.floor(result.rows.length / 400));
        return result.rows.filter((_: any, i: number) => i % step === 0 || i === result.rows.length - 1);
      })()
    : [];

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      {/* Sidebar */}
      <aside className="w-80 min-w-80 h-full overflow-y-auto border-r border-border bg-card p-4 flex flex-col gap-3">
        <div className="flex items-center gap-2 pb-2 border-b border-border">
          <Activity className="w-5 h-5 text-primary" />
          <h2 className="text-base font-bold">QQQ/TQQQ Switching</h2>
        </div>

        <label className="text-xs text-muted-foreground font-medium">Base Asset</label>
        <input value={config.base_asset} onChange={(e) => update({ base_asset: e.target.value.toUpperCase() })}
          className="bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm" />

        <label className="text-xs text-muted-foreground font-medium">Leveraged Asset</label>
        <input value={config.leveraged_asset} onChange={(e) => update({ leveraged_asset: e.target.value.toUpperCase() })}
          className="bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm" />

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-xs text-muted-foreground font-medium">Start Date</label>
            <input type="date" value={config.start_date} onChange={(e) => update({ start_date: e.target.value })}
              className="w-full bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground font-medium">End Date</label>
            <input type="date" value={config.end_date} onChange={(e) => update({ end_date: e.target.value })}
              className="w-full bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm" />
          </div>
        </div>

        <label className="text-xs text-muted-foreground font-medium">Initial Capital ($)</label>
        <input type="number" value={config.initial_capital} step={10000}
          onChange={(e) => update({ initial_capital: parseFloat(e.target.value) || 100000 })}
          className="bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm" />

        <label className="text-xs text-muted-foreground font-medium">EMA Period (days)</label>
        <input type="number" value={config.ema_period} min={5}
          onChange={(e) => update({ ema_period: parseInt(e.target.value) || 200 })}
          className="bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm" />

        <label className="text-xs text-muted-foreground font-medium">
          Drawdown Tiers (%, comma-separated)
        </label>
        <input value={config.drawdown_tiers.join(", ")}
          onChange={(e) => {
            const tiers = e.target.value.split(",").map((v) => parseFloat(v.trim())).filter((v) => !isNaN(v));
            update({ drawdown_tiers: tiers });
          }}
          className="bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm"
          placeholder="-20, -30, -40, -50" />

        <label className="text-xs text-muted-foreground font-medium">Transfer % per Tier</label>
        <input type="number" value={config.transfer_pct_per_tier} min={1} max={100} step={5}
          onChange={(e) => update({ transfer_pct_per_tier: parseFloat(e.target.value) || 25 })}
          className="bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm" />

        <label className="text-xs text-muted-foreground font-medium">Reset Threshold (% above EMA)</label>
        <input type="number" value={config.reset_threshold} step={5}
          onChange={(e) => update({ reset_threshold: parseFloat(e.target.value) || 0 })}
          className="bg-secondary/50 border border-border rounded-md px-3 py-1.5 text-sm" />

        <div className="mt-2 p-2 bg-primary/5 border border-primary/20 rounded-md">
          <p className="text-[10px] text-muted-foreground leading-relaxed">
            When {config.base_asset} drops below each tier from its 200 EMA,
            transfer {config.transfer_pct_per_tier}% of {config.base_asset} into {config.leveraged_asset}.
            Reset when {config.base_asset} recovers above EMA.
          </p>
        </div>

        <button onClick={run} disabled={loading}
          className="mt-3 w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-2.5 rounded-lg transition disabled:opacity-50 flex items-center justify-center gap-2">
          {loading ? (
            <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Running...</>
          ) : (
            <><TrendingUp className="w-4 h-4" /> Run Backtest</>
          )}
        </button>

        <a href="/" className="text-xs text-center text-muted-foreground hover:text-primary transition mt-2">
          ← Back to TQQQ DCA Strategy
        </a>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">QQQ/TQQQ Dynamic Switching Backtest</h1>
            <p className="text-xs text-muted-foreground">
              EMA-distance multi-tier drawdown switching from {config.base_asset} to {config.leveraged_asset}
            </p>
          </div>
        </div>

        {error && (
          <div className="bg-destructive/10 border border-destructive/30 text-red-400 rounded-lg p-3 text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}

        {!result && !loading && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center max-w-md text-muted-foreground">
              <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-20" />
              <h2 className="text-lg font-semibold text-foreground mb-2">Configure & Run</h2>
              <p className="text-sm">Set your switching tiers in the sidebar and click Run Backtest.</p>
            </div>
          </div>
        )}

        {loading && (
          <div className="flex-1 flex items-center justify-center">
            <div className="w-12 h-12 border-4 border-primary/30 border-t-primary rounded-full animate-spin" />
          </div>
        )}

        {result && !loading && (
          <>
            {/* KPI Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
              {[
                { label: "Strategy Final", value: formatDollar(result.strategy_metrics.terminal_value), sub: `${result.strategy_metrics.total_return}%` },
                { label: "Benchmark Final", value: formatDollar(result.benchmark_metrics.terminal_value), sub: `${result.benchmark_metrics.total_return}%` },
                { label: "Strategy CAGR", value: `${result.strategy_metrics.cagr}%`, sub: `${result.strategy_metrics.years}yr` },
                { label: "Strategy MDD", value: `${result.strategy_metrics.mdd}%`, sub: "peak-to-trough" },
                { label: "Benchmark MDD", value: `${result.benchmark_metrics.mdd}%`, sub: "peak-to-trough" },
              ].map((card, i) => (
                <div key={i} className="bg-card border border-border rounded-xl p-4">
                  <div className="text-xs text-muted-foreground mb-1">{card.label}</div>
                  <div className="text-lg font-bold">{card.value}</div>
                  <div className="text-xs text-muted-foreground">{card.sub}</div>
                </div>
              ))}
            </div>

            {/* Portfolio Growth Chart */}
            <div className="bg-card border border-border rounded-xl p-4">
              <h3 className="text-sm font-semibold text-muted-foreground mb-3">Portfolio Growth: Strategy vs Benchmark</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ComposedChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217,33%,17%)" />
                  <XAxis dataKey="date" tick={{ fill: "hsl(215,20%,65%)", fontSize: 10 }} stroke="hsl(217,33%,17%)"
                    tickFormatter={(d) => new Date(d).toLocaleDateString("en-US", { month: "short", year: "2-digit" })} />
                  <YAxis tickFormatter={formatDollar} tick={{ fill: "hsl(215,20%,65%)", fontSize: 10 }} stroke="hsl(217,33%,17%)" />
                  <Tooltip contentStyle={{ backgroundColor: "hsl(222,47%,8%)", border: "1px solid hsl(217,33%,17%)", borderRadius: "8px", fontSize: "12px" }}
                    formatter={(v: any, name: string) => [formatDollar(v), name]} />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <Line type="monotone" dataKey="total_portfolio" stroke="#22c55e" strokeWidth={2} dot={false} name="Strategy" />
                  <Line type="monotone" dataKey="benchmark" stroke="#f59e0b" strokeWidth={1.5} dot={false} strokeDasharray="5 5" name="Benchmark (100% QQQ)" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* EMA Distance Chart */}
            <div className="bg-card border border-border rounded-xl p-4">
              <h3 className="text-sm font-semibold text-muted-foreground mb-3">EMA Distance % (Tier Triggers)</h3>
              <ResponsiveContainer width="100%" height={200}>
                <ComposedChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217,33%,17%)" />
                  <XAxis dataKey="date" tick={{ fill: "hsl(215,20%,65%)", fontSize: 10 }} stroke="hsl(217,33%,17%)"
                    tickFormatter={(d) => new Date(d).toLocaleDateString("en-US", { month: "short", year: "2-digit" })} />
                  <YAxis tick={{ fill: "hsl(215,20%,65%)", fontSize: 10 }} stroke="hsl(217,33%,17%)" />
                  <Tooltip contentStyle={{ backgroundColor: "hsl(222,47%,8%)", border: "1px solid hsl(217,33%,17%)", borderRadius: "8px", fontSize: "12px" }} />
                  <Area type="monotone" dataKey="ema_dist_pct" fill="hsl(217,91%,60%)" fillOpacity={0.1} stroke="hsl(217,91%,60%)" strokeWidth={1} name="EMA Dist %" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* Allocation Chart */}
            <div className="bg-card border border-border rounded-xl p-4">
              <h3 className="text-sm font-semibold text-muted-foreground mb-3">Asset Allocation ($)</h3>
              <ResponsiveContainer width="100%" height={200}>
                <ComposedChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217,33%,17%)" />
                  <XAxis dataKey="date" tick={{ fill: "hsl(215,20%,65%)", fontSize: 10 }} stroke="hsl(217,33%,17%)"
                    tickFormatter={(d) => new Date(d).toLocaleDateString("en-US", { month: "short", year: "2-digit" })} />
                  <YAxis tickFormatter={formatDollar} tick={{ fill: "hsl(215,20%,65%)", fontSize: 10 }} stroke="hsl(217,33%,17%)" />
                  <Tooltip contentStyle={{ backgroundColor: "hsl(222,47%,8%)", border: "1px solid hsl(217,33%,17%)", borderRadius: "8px", fontSize: "12px" }}
                    formatter={(v: any, name: string) => [formatDollar(v), name]} />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <Area type="monotone" dataKey="base_value" stackId="1" fill="#3b82f6" fillOpacity={0.6} stroke="#3b82f6" name={config.base_asset} />
                  <Area type="monotone" dataKey="lev_value" stackId="1" fill="#8b5cf6" fillOpacity={0.6} stroke="#8b5cf6" name={config.leveraged_asset} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

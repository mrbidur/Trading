"use client";

import React, { useState, useCallback } from "react";
import ConfigPanel from "@/components/ConfigPanel";
import MetricsCards from "@/components/MetricsCards";
import Charts from "@/components/Charts";
import DataTable from "@/components/DataTable";
import { BacktestConfig, BacktestResult, DEFAULT_CONFIG } from "@/lib/types";
import { Download, BarChart3, Activity } from "lucide-react";

export default function HomePage() {
  const [config, setConfig] = useState<BacktestConfig>(DEFAULT_CONFIG);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runBacktest = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => null);
        const detail = err?.detail || `Server returned HTTP ${res.status}. Make sure the backend is running on port 8000.`;
        throw new Error(detail);
      }
      const data: BacktestResult = await res.json();
      setResult(data);
    } catch (e: any) {
      if (e.message?.includes("fetch")) {
        setError("Cannot connect to backend server. Make sure the backend is running (port 8000).");
      } else {
        setError(e.message || "Failed to run backtest");
      }
    } finally {
      setLoading(false);
    }
  }, [config]);

  const exportCSV = useCallback(async () => {
    try {
      const res = await fetch("/api/export-csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!res.ok) throw new Error("Export failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${config.ticker}_backtest.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e.message || "CSV export failed");
    }
  }, [config]);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar Configuration Panel */}
      <ConfigPanel config={config} onChange={setConfig} onRun={runBacktest} loading={loading} />

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
        {/* Header Bar */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <Activity className="w-7 h-7 text-primary" />
            <div>
              <h1 className="text-xl font-bold text-foreground">
                Dynamic ETF Backtest Engine
              </h1>
              <p className="text-xs text-muted-foreground">
                State-machine allocation strategy &middot; Leveraged ETF backtesting with configurable parameters
              </p>
            </div>
          </div>
          {result && (
            <button
              onClick={exportCSV}
              className="flex items-center gap-2 bg-secondary hover:bg-secondary/80 text-foreground text-sm font-medium py-2 px-4 rounded-lg transition border border-border"
            >
              <Download className="w-4 h-4" />
              Export CSV
            </button>
          )}
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bg-destructive/10 border border-destructive/30 text-red-400 rounded-lg p-3 text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Empty State */}
        {!result && !loading && !error && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center max-w-md">
              <BarChart3 className="w-20 h-20 mx-auto mb-4 text-muted-foreground opacity-20" />
              <h2 className="text-lg font-semibold text-foreground mb-2">
                Configure & Run a Backtest
              </h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Set your strategy parameters in the sidebar. Adjust the ticker, timeframe,
                contribution amounts, EMA distance triggers, tranche thresholds, and yield
                settings. Click <strong>&quot;Run Backtest&quot;</strong> to generate full
                performance analytics, interactive charts, and an exportable CSV audit trail.
              </p>
            </div>
          </div>
        )}

        {/* Loading Spinner */}
        {loading && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <div className="w-14 h-14 border-4 border-primary/30 border-t-primary rounded-full animate-spin mx-auto mb-4" />
              <p className="text-sm text-muted-foreground">
                Fetching market data & running backtest engine...
              </p>
            </div>
          </div>
        )}

        {/* Results Dashboard */}
        {result && !loading && (
          <>
            {/* KPI Summary Cards */}
            <MetricsCards
              strategyMetrics={result.strategy_metrics}
              dcaMetrics={result.dca_metrics}
              ticker={result.ticker}
            />

            {/* Interactive Charts */}
            <Charts rows={result.rows} dcaRows={result.dca_rows} ticker={result.ticker} />

            {/* Full Execution Table */}
            <DataTable rows={result.rows} />
          </>
        )}
      </main>
    </div>
  );
}

"use client";

import React from "react";
import { Metrics } from "@/lib/types";
import { TrendingUp, TrendingDown, BarChart3, Shield, DollarSign } from "lucide-react";

interface MetricsCardsProps {
  strategyMetrics: Metrics;
  dcaMetrics: Metrics;
  ticker: string;
}

function MetricCard({
  label,
  stratValue,
  dcaValue,
  format,
  icon: Icon,
  betterIfHigher = true,
}: {
  label: string;
  stratValue: number;
  dcaValue: number;
  format: (v: number) => string;
  icon: React.ElementType;
  betterIfHigher?: boolean;
}) {
  const diff = stratValue - dcaValue;
  const isBetter = betterIfHigher ? diff > 0 : diff < 0;

  return (
    <div className="bg-card border border-border rounded-xl p-4 flex flex-col gap-2">
      <div className="flex items-center gap-2 text-muted-foreground">
        <Icon className="w-4 h-4" />
        <span className="text-xs font-medium uppercase tracking-wide">{label}</span>
      </div>
      <div className="flex items-end gap-3">
        <div>
          <div className="text-lg font-bold text-foreground">{format(stratValue)}</div>
          <div className="text-[10px] text-muted-foreground uppercase">Strategy</div>
        </div>
        <div className="text-muted-foreground text-sm pb-1">vs</div>
        <div>
          <div className="text-base font-semibold text-muted-foreground">{format(dcaValue)}</div>
          <div className="text-[10px] text-muted-foreground uppercase">DCA</div>
        </div>
      </div>
      <div
        className={`text-xs font-semibold flex items-center gap-1 ${
          isBetter ? "text-green-400" : "text-red-400"
        }`}
      >
        {isBetter ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
        {diff > 0 ? "+" : ""}
        {format(diff)} vs DCA
      </div>
    </div>
  );
}

export default function MetricsCards({ strategyMetrics, dcaMetrics, ticker }: MetricsCardsProps) {
  const fmtDollar = (v: number) =>
    `$${Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 })}${v < 0 ? " (loss)" : ""}`;
  const fmtPct = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
  const fmtRatio = (v: number) => v.toFixed(3);

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
          Performance Summary
        </h2>
        <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded font-medium">
          {ticker} &middot; {strategyMetrics.years}yr
        </span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        <MetricCard
          label="Terminal Value"
          stratValue={strategyMetrics.terminal_value}
          dcaValue={dcaMetrics.terminal_value}
          format={fmtDollar}
          icon={BarChart3}
        />
        <MetricCard
          label="Total Invested"
          stratValue={strategyMetrics.total_invested}
          dcaValue={dcaMetrics.total_invested}
          format={fmtDollar}
          icon={DollarSign}
          betterIfHigher={false}
        />
        <MetricCard
          label="CAGR"
          stratValue={strategyMetrics.cagr}
          dcaValue={dcaMetrics.cagr}
          format={fmtPct}
          icon={TrendingUp}
        />
        <MetricCard
          label="Max Drawdown"
          stratValue={strategyMetrics.mdd}
          dcaValue={dcaMetrics.mdd}
          format={fmtPct}
          icon={Shield}
          betterIfHigher={false}
        />
        <MetricCard
          label="Sortino Ratio"
          stratValue={strategyMetrics.sortino}
          dcaValue={dcaMetrics.sortino}
          format={fmtRatio}
          icon={BarChart3}
        />
      </div>
    </div>
  );
}

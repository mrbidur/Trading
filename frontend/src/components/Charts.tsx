"use client";

import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Area,
  ReferenceDot,
} from "recharts";
import { BacktestRow, DCARow } from "@/lib/types";

interface ChartsProps {
  rows: BacktestRow[];
  dcaRows: DCARow[];
  ticker: string;
}

function formatDollar(value: number) {
  if (value >= 1000000) return `$${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `$${(value / 1000).toFixed(0)}K`;
  return `$${value.toFixed(0)}`;
}

function formatDate(dateStr: string) {
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
}

export default function Charts({ rows, dcaRows, ticker }: ChartsProps) {
  // Merge and sample data for portfolio growth chart
  const portfolioData = useMemo(() => {
    const step = Math.max(1, Math.floor(rows.length / 300));
    const sampled: any[] = [];
    for (let i = 0; i < rows.length; i += step) {
      const r = rows[i];
      const d = dcaRows[i];
      sampled.push({
        date: r.date,
        strategy: r.portfolio_valuation,
        dca: d?.portfolio_valuation ?? 0,
        cash_buffer: r.cash_buffer,
        invested: r.out_of_pocket_total,
      });
    }
    // Always include last data point
    const last = rows[rows.length - 1];
    const lastDca = dcaRows[dcaRows.length - 1];
    if (sampled.length === 0 || sampled[sampled.length - 1].date !== last?.date) {
      sampled.push({
        date: last.date,
        strategy: last.portfolio_valuation,
        dca: lastDca?.portfolio_valuation ?? 0,
        cash_buffer: last.cash_buffer,
        invested: last.out_of_pocket_total,
      });
    }
    return sampled;
  }, [rows, dcaRows]);

  // Price/EMA overlay chart
  const priceData = useMemo(() => {
    const step = Math.max(1, Math.floor(rows.length / 300));
    const sampled: any[] = [];
    for (let i = 0; i < rows.length; i += step) {
      const r = rows[i];
      sampled.push({
        date: r.date,
        price: r.close,
        ema: r.ema,
      });
    }
    return sampled;
  }, [rows]);

  // Trade markers (buy = green, sell = red)
  const tradeMarkers = useMemo(() => {
    return rows
      .filter(
        (r) =>
          r.action.includes("PROFIT TAKE") ||
          r.action.includes("TRANCHE") ||
          r.action.includes("STEP SCALE")
      )
      .slice(0, 150)
      .map((r) => ({
        date: r.date,
        price: r.close,
        action: r.action,
        isBuy: r.action.includes("TRANCHE"),
      }));
  }, [rows]);

  const tooltipStyle = {
    backgroundColor: "hsl(222, 47%, 8%)",
    border: "1px solid hsl(217, 33%, 17%)",
    borderRadius: "8px",
    fontSize: "12px",
    color: "hsl(210, 40%, 95%)",
  };

  return (
    <div className="flex flex-col gap-6 w-full">
      {/* Portfolio Growth Chart */}
      <div className="bg-card border border-border rounded-xl p-4">
        <h3 className="text-sm font-semibold text-muted-foreground mb-4 uppercase tracking-wide">
          Portfolio Growth: Strategy vs. Standard DCA
        </h3>
        <ResponsiveContainer width="100%" height={340}>
          <ComposedChart data={portfolioData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217,33%,17%)" />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              tick={{ fill: "hsl(215,20%,65%)", fontSize: 10 }}
              stroke="hsl(217,33%,17%)"
            />
            <YAxis
              tickFormatter={formatDollar}
              tick={{ fill: "hsl(215,20%,65%)", fontSize: 10 }}
              stroke="hsl(217,33%,17%)"
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelFormatter={(l) => `Date: ${l}`}
              formatter={(v: any, name: string) => [formatDollar(v), name]}
            />
            <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }} />
            <Area
              type="monotone"
              dataKey="cash_buffer"
              fill="hsl(217,91%,60%)"
              fillOpacity={0.08}
              stroke="hsl(217,91%,60%)"
              strokeWidth={1}
              strokeDasharray="2 2"
              name="Cash Buffer"
            />
            <Line
              type="monotone"
              dataKey="strategy"
              stroke="#22c55e"
              strokeWidth={2.5}
              dot={false}
              name="Strategy"
            />
            <Line
              type="monotone"
              dataKey="dca"
              stroke="#f59e0b"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="5 5"
              name="Standard DCA"
            />
            <Line
              type="monotone"
              dataKey="invested"
              stroke="hsl(215,20%,45%)"
              strokeWidth={1}
              dot={false}
              strokeDasharray="2 2"
              name="Total Invested"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Price / EMA Overlay */}
      <div className="bg-card border border-border rounded-xl p-4">
        <h3 className="text-sm font-semibold text-muted-foreground mb-4 uppercase tracking-wide">
          {ticker} Price & {`EMA`} with Trade Markers
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={priceData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217,33%,17%)" />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              tick={{ fill: "hsl(215,20%,65%)", fontSize: 10 }}
              stroke="hsl(217,33%,17%)"
            />
            <YAxis
              tickFormatter={(v) => `$${v.toFixed(0)}`}
              tick={{ fill: "hsl(215,20%,65%)", fontSize: 10 }}
              stroke="hsl(217,33%,17%)"
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelFormatter={(l) => `Date: ${l}`}
              formatter={(v: any, name: string) => [`$${Number(v).toFixed(2)}`, name]}
            />
            <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }} />
            <Line
              type="monotone"
              dataKey="price"
              stroke="#8b5cf6"
              strokeWidth={1.5}
              dot={false}
              name="Price"
            />
            <Line
              type="monotone"
              dataKey="ema"
              stroke="#f97316"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="4 4"
              name="EMA"
            />
          </LineChart>
        </ResponsiveContainer>
        {/* Trade marker legend */}
        <div className="flex items-center gap-6 mt-3 text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span>Tranche Buy ({tradeMarkers.filter((m) => m.isBuy).length})</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span>Profit Take / Scale-Out ({tradeMarkers.filter((m) => !m.isBuy).length})</span>
          </div>
        </div>
      </div>
    </div>
  );
}

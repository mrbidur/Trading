"use client";

import React, { useState } from "react";
import { BacktestRow } from "@/lib/types";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface DataTableProps {
  rows: BacktestRow[];
}

export default function DataTable({ rows }: DataTableProps) {
  const [page, setPage] = useState(0);
  const perPage = 25;
  const totalPages = Math.ceil(rows.length / perPage);
  const pageRows = rows.slice(page * perPage, (page + 1) * perPage);

  const actionColor = (action: string) => {
    if (action.includes("PROFIT TAKE") || action.includes("STEP SCALE"))
      return "text-red-400 font-semibold";
    if (action.includes("TRANCHE")) return "text-green-400 font-semibold";
    return "text-muted-foreground";
  };

  return (
    <div className="bg-card border border-border rounded-xl p-4 overflow-hidden">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
          Execution Audit Log ({rows.length} entries)
        </h3>
        <div className="flex items-center gap-2">
          <button
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
            className="p-1 rounded bg-secondary disabled:opacity-30 hover:bg-primary/20 transition"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-xs text-muted-foreground min-w-[60px] text-center">
            {page + 1} / {totalPages}
          </span>
          <button
            disabled={page >= totalPages - 1}
            onClick={() => setPage(page + 1)}
            className="p-1 rounded bg-secondary disabled:opacity-30 hover:bg-primary/20 transition"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs whitespace-nowrap">
          <thead>
            <tr className="border-b border-border text-muted-foreground">
              <th className="text-left py-2 px-2">Date</th>
              <th className="text-right py-2 px-2">Close</th>
              <th className="text-right py-2 px-2">EMA</th>
              <th className="text-right py-2 px-2">Dist%</th>
              <th className="text-center py-2 px-2">OB</th>
              <th className="text-right py-2 px-2">Harv#</th>
              <th className="text-right py-2 px-2">Peak</th>
              <th className="text-right py-2 px-2">PB%</th>
              <th className="text-left py-2 px-2">Action</th>
              <th className="text-right py-2 px-2">Deposit</th>
              <th className="text-right py-2 px-2">Cash</th>
              <th className="text-right py-2 px-2">Shares</th>
              <th className="text-right py-2 px-2">Valuation</th>
              <th className="text-right py-2 px-2">OOP Total</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map((r, i) => (
              <tr
                key={i}
                className="border-b border-border/30 hover:bg-secondary/30 transition-colors"
              >
                <td className="py-1.5 px-2 font-mono">{r.date}</td>
                <td className="text-right py-1.5 px-2">${r.close.toFixed(2)}</td>
                <td className="text-right py-1.5 px-2">${r.ema.toFixed(2)}</td>
                <td className="text-right py-1.5 px-2">
                  <span className={r.dist_pct > 0 ? "text-green-400" : "text-red-400"}>
                    {r.dist_pct.toFixed(1)}%
                  </span>
                </td>
                <td className="text-center py-1.5 px-2">
                  <span
                    className={`inline-block w-2.5 h-2.5 rounded-full ${
                      r.overbought ? "bg-red-400 shadow-sm shadow-red-400/50" : "bg-gray-700"
                    }`}
                  />
                </td>
                <td className="text-right py-1.5 px-2">{r.n_harvest}</td>
                <td className="text-right py-1.5 px-2">
                  {r.swing_high_peak > 0 ? `$${r.swing_high_peak.toFixed(2)}` : "-"}
                </td>
                <td className="text-right py-1.5 px-2">
                  {r.pullback_pct !== 0 ? `${r.pullback_pct.toFixed(1)}%` : "-"}
                </td>
                <td className={`py-1.5 px-2 ${actionColor(r.action)}`}>{r.action}</td>
                <td className="text-right py-1.5 px-2">${r.deposit}</td>
                <td className="text-right py-1.5 px-2">
                  ${r.cash_buffer.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                </td>
                <td className="text-right py-1.5 px-2">{r.total_shares.toFixed(2)}</td>
                <td className="text-right py-1.5 px-2 font-semibold text-foreground">
                  ${r.portfolio_valuation.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                </td>
                <td className="text-right py-1.5 px-2">
                  ${r.out_of_pocket_total.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const res = await fetch(`${BACKEND_URL}/export-csv`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({ detail: "Export failed" }));
      return NextResponse.json(data, { status: res.status });
    }

    const csvText = await res.text();
    const headers = new Headers();
    headers.set("Content-Type", "text/csv");
    headers.set(
      "Content-Disposition",
      res.headers.get("Content-Disposition") || 'attachment; filename="backtest.csv"'
    );

    return new NextResponse(csvText, { status: 200, headers });
  } catch (error: any) {
    console.error("[API Proxy] Failed to reach backend:", error.message);
    return NextResponse.json(
      {
        detail: `Cannot connect to backend at ${BACKEND_URL}. Error: ${error.message}`,
      },
      { status: 502 }
    );
  }
}

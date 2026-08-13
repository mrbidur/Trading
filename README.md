# Dynamic Leveraged ETF Backtesting Web Application

A full-stack, customizable web application for backtesting dynamic state-machine allocation strategies on leveraged ETFs (TQQQ, UPRO, SOXL, etc.) and standard assets (QQQ, SPY, etc.).

![Architecture](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square) ![Frontend](https://img.shields.io/badge/Frontend-Next.js_14-000000?style=flat-square) ![Charts](https://img.shields.io/badge/Charts-Recharts-8884d8?style=flat-square) ![Data](https://img.shields.io/badge/Data-Yahoo_Finance-7B1FA2?style=flat-square)

---

## Features

- **Fully Configurable Strategy Parameters** — Every parameter is adjustable via the UI (no hardcoded values)
- **State Machine Engine** — EMA-distance-based overbought detection, profit harvesting, hysteresis resets, step scale-outs, and multi-tranche dip buying
- **Adaptive Thresholds** — Dynamically adjusts tranche triggers based on consecutive harvest counts
- **Cash Buffer Yield** — Models APY on uninvested cash (T-bill sweep)
- **Interactive Charts** — Portfolio growth (Strategy vs. DCA), Price/EMA overlay with trade markers
- **Performance Metrics** — Terminal value, CAGR, Max Drawdown, Sortino Ratio with side-by-side DCA comparison
- **CSV Export** — 15-column audit trail export for offline analysis
- **Multi-Asset Support** — Any Yahoo Finance ticker (TQQQ, QQQ, SPY, UPRO, NVDA, etc.)
- **Flexible Timeframes** — Custom date ranges, 1Y/3Y/5Y/10Y/Max presets
- **Execution Frequency** — Weekly (Friday), Daily, or Monthly

---

## Architecture

```
Trading/
├── backend/                 # FastAPI Python backend
│   ├── main.py              # API server + backtest engine
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # Backend container
│   └── .cache/              # yfinance disk cache (auto-created)
├── frontend/                # Next.js React frontend
│   ├── src/
│   │   ├── app/             # Next.js app router
│   │   ├── components/      # React components
│   │   │   ├── ConfigPanel.tsx   # Parameter sidebar
│   │   │   ├── MetricsCards.tsx  # KPI comparison cards
│   │   │   ├── Charts.tsx        # Recharts visualizations
│   │   │   └── DataTable.tsx     # Paginated audit log
│   │   └── lib/
│   │       ├── types.ts     # TypeScript interfaces
│   │       └── utils.ts     # Utility functions
│   ├── package.json
│   ├── tailwind.config.ts
│   └── Dockerfile           # Frontend container
├── docker-compose.yml       # One-command deployment
└── README.md
```

---

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
docker-compose up --build
```

Access the app at [http://localhost:3000](http://localhost:3000).

### Option 2: Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Access at [http://localhost:3000](http://localhost:3000).

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/backtest` | Run backtest, returns JSON with metrics + row data |
| `POST` | `/export-csv` | Run backtest, returns downloadable CSV file |
| `GET` | `/tickers` | Popular ticker suggestions |

---

## Configurable Parameters

### Core Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| Ticker | `TQQQ` | Any Yahoo Finance symbol |
| Timeframe | 2015–Present | Custom or preset ranges |
| Frequency | Weekly | Weekly / Daily / Monthly |

### Contributions
| Parameter | Default | Description |
|-----------|---------|-------------|
| Base Deposit | $200 | Normal market deposit |
| Overbought Deposit | $100 | Deposit during overbought state |
| Tranche Deposit | $400 | Deposit during dip-buy executions |

### EMA & Harvest Logic
| Parameter | Default | Description |
|-----------|---------|-------------|
| MA Period | 200 | EMA lookback window (days) |
| Profit Harvest Distance | +40% | Distance above EMA to trigger harvest |
| Hysteresis Reset | +30% | Distance below which overbought resets |
| Initial Scale-Out | 70% | Shares sold on first harvest |
| Step Increment | +10% | Distance step for additional scale-outs |
| Step Scale-Out | 10% | Shares sold per step |

### Tranche Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| Thresholds | -25, -35, -45, -55, -65% | Pullback levels from swing high |
| Cash Deploy | 20% | Peak cash deployed per tranche |
| Adaptive Mode | Off | Adjusts thresholds after N harvests |

### Yield
| Parameter | Default | Description |
|-----------|---------|-------------|
| Cash APY | 4.5% | Interest on uninvested cash buffer |

---

## CSV Export Schema

The exported CSV contains a 15-column audit trail:

| Col | Header | Description |
|-----|--------|-------------|
| A | Date | Execution date (YYYY-MM-DD) |
| B | Close Price | Asset price ($) |
| C | Moving Average | EMA value ($) |
| D | Dist % | Distance from EMA (%) |
| E | Overbought Flag | TRUE / FALSE |
| F | Harvest Counter | Consecutive harvest count |
| G | Swing High Peak | Peak price anchor ($) |
| H | Pullback % | Drawdown from peak (%) |
| I | Tranche 1 Trigger | Active threshold (%) |
| J | Action Executed | Event string |
| K | Deposit ($) | Cash contribution |
| L | Cash Buffer ($) | Uninvested cash |
| M | Total Shares | Cumulative shares |
| N | Portfolio Valuation ($) | Total portfolio value |
| O | Out of Pocket Total ($) | Cumulative deposits |

---

## State Machine Logic

The engine processes each execution period through these phases:

1. **Cash Yield** — Apply periodic interest on cash buffer
2. **Hysteresis Reset** — Reset overbought flag if price drops below threshold
3. **Profit Take** — Sell shares if EMA distance exceeds trigger (+ step scale-outs)
4. **Adaptive Threshold** — Adjust T1 trigger after N consecutive harvests
5. **Tranche Dip Buy** — Deploy cash + deposit on drawdown from swing high
6. **Normal DCA** — Standard periodic investment when no special events fire

---

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Pandas, NumPy, yfinance, diskcache
- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS, Recharts, Lucide Icons
- **Deployment**: Docker Compose
- **Data**: Yahoo Finance API with 4-hour local disk cache

---

## License

MIT

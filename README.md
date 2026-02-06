# MT5 Trading System

Automated crypto portfolio rebalancing bot for MetaTrader 5. Runs scheduled sessions that update market data, aggregate signals from multiple pre-fitted strategy models, and adjust MT5 positions. Designed for FTMO and similar prop-trading environments.

## Features

- **Multi-strategy ensemble** — 10+ strategies (trend, momentum, breakout, reversal, flow, alpha) aggregated per asset
- **Volatility targeting** — Rogers-Satchell vol, Weibull regime tilt, strategy decay
- **Data sources** — MT5 1H bars (primary), Binance klines (optional)
- **Hourly rebalancing** — Configurable via scheduler; aligns to market close when needed

## Quick Start

```bash
# Clone, venv, install
git clone <repo-url>
cd MT5
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Configure (copy .env.example → .env, add MT5 credentials)
copy .env.example .env

# One-time: init DB, fit models
python -m src.db_load
python src/fit_models.py

# Run rebalance (MT5 terminal must be running & logged in)
python src/main.py
```

See [docs/SETUP.md](docs/SETUP.md) for full setup, MetaTrader config, and troubleshooting.

## Project Structure

```
MT5/
├── src/
│   ├── main.py          # Entry point: update data → rebalance
│   ├── fit_models.py    # Offline model fitting per symbol
│   ├── port_rebalance.py# Rebalance logic, signal aggregation
│   ├── data_update.py   # Fetch MT5 + Binance → SQLite
│   ├── api_mt5.py       # MT5 init, bars, orders, positions
│   ├── api_bnb.py       # Binance klines, trades
│   ├── db_load.py       # Save to DB
│   ├── db_read.py       # Read from DB
│   ├── strat_models.py  # All strategy classes
│   ├── strat_io.py      # Save/load joblib models
│   └── ...
├── docs/
│   ├── ARCHITECTURE.md  # Data flow, operational flow, strategies
│   └── SETUP.md         # Environment, MT5, DB, scheduling
├── data/                # SQLite DB (gitignored)
├── models/              # Fitted models per symbol (gitignored)
├── .env.example         # Env template
└── requirements.txt
```

## Strategies (Summary)

| Strategy | Type | Signal |
|----------|------|--------|
| EmaVolStrategy | Trend | EMA crossover × vol tilt |
| AccelVolStrategy | Momentum | EMA acceleration (momentum-of-momentum) |
| BreakVolStrategy | Breakout | Range breakout smoothed |
| BlockVolStrategy | Trend | Block momentum (HH + HL) |
| WedThuStrategy | Calendar | Long Wed / Short Thu |
| RevStrategy | Reversal | Short-term mean reversion |
| OrthAlphaStrategy | Alpha | Momentum-orthogonal residuals |
| VolScaleStrategy | Vol | Inverse-vol baseline |
| BuySellVolStrategy | Flow | Taker buy/sell ratio (Binance) |

Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Environment

| Variable | Required | Description |
|----------|----------|-------------|
| `MT5_SERVER` | Yes | MT5 server (e.g. `FTMO-Demo`) |
| `MT5_LOGIN` | Yes | Account number |
| `MT5_PASSWORD` | Yes | Account password |
| `DB_PATH` | No | SQLite path (default: `data/trading.db`) |
| `MODEL_DIR` | No | Models path (default: `models`) |

Copy `.env.example` to `.env` and fill in credentials.

## Scheduling

Run `main.py` via Windows Task Scheduler (or cron) at your desired interval (e.g. hourly). The script aligns to the hourly boundary and can offset for market close.

## Docs

- **[SETUP.md](docs/SETUP.md)** — Prerequisites, install, `.env`, MetaTrader setup, DB init, model fitting, scheduling, troubleshooting
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Data flow, operational flow, strategy details, file map, diagrams

## Requirements

- Python 3.9+
- Windows (MetaTrader5 package needs MT5 desktop terminal)
- MetaTrader 5 terminal with Algo Trading enabled

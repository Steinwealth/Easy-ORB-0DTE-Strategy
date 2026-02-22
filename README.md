# Easy ORB 0DTE Strategy

**Automated dual-strategy system:** Opening Range Breakout (ORB) for ETFs + selective **0DTE options overlay** for convex amplification — built on **broker-only data**, multi-layer risk controls, and full position monitoring.

> **Positioning:** This repo is part of the "Easy Trading Software" suite — production-grade strategy execution + monitoring designed for reliability, reproducibility, and real-world constraints (rate limits, slippage, missing candles, broker quirks).

---

## Evidence (validated baseline)

**Historical validation (11 trading days, October 2024):**

- **Weekly return:** **+73.69%**
- **Winning days:** **10/11 (91%)**
- **Max drawdown:** **-0.84%** (reported as ~96% reduction vs baseline)
- **Profit factor:** **194.00**
- **Monthly projection (compounded):** **+508%**

> Notes: Validation results reflect the documented test window and system configuration used at the time. Real-world performance will vary due to fills, spreads, latency, volatility regimes, and broker constraints. This project is not financial advice.

---

## What it does

This repository contains:

1. **Easy ORB 0DTE Strategy** — Main application (ORB capture → signal collection → execution → monitoring).  
   Deploys as a **single Cloud Run service**.

2. **easy0DTE** — Options overlay module (convex eligibility + execution) used only on the highest-conviction setups.

3. **easyCollector** — Standalone market snapshot collector (ORB / SIGNAL / OUTCOME) for US 0DTE + crypto (separate deploy).

**Core philosophy:**

- ORB captures breakouts from the first 15 minutes of market action.
- 0DTE overlay is selective: **"Gamma > Leverage."**
- Systems-first: guardrails, monitoring, repeatable execution, clean alerts.

---

## End-to-end flow (one minute)

Typical flow:

**OAuth → Good Morning → ORB Capture → validation candle (7:00/7:15) → Signal Collection → execution → monitoring → EOD**

Key timing (PT):

- **6:30–6:45** ORB capture window  
- **7:00 / 7:15** validation candle + signal collection window  
- **7:30** batch execution  
- **All day** monitoring + exits + reporting

---

## Guardrails (high level)

Designed to reduce failure modes common in automated trading:

- **Risk engine & position sizing controls** (caps, redistribution, safety floors)
- **Market condition / "red day" filtering** (portfolio + signal level controls)
- **Duplicate prevention + trade persistence**
- **Exit system + trailing/breakeven logic** with monitoring loop
- **Holiday and low-liquidity day avoidance** (config-driven)

For full detail: see **[docs/Risk.md](docs/Risk.md)** and **[docs/Strategy.md](docs/Strategy.md)**.

---

## Repository structure

```text
easyORBStrategy/
├── main.py                  # Application entry (local)
├── cloud_run_entry.py       # Cloud Run entry point
├── requirements.txt
├── Dockerfile
├── modules/                 # ORB strategy: data, risk, alerts, execution, GCS
├── configs/                 # Strategy configs + templates (*.template, strategies.env, modes/, etc.)
├── data/
│   ├── watchlist/           # core_list.csv, 0dte_list.csv, sentiment/orb mappings
│   ├── holidays_custom.json
│   └── holidays_future_proof.json
├── priority_optimizer/      # 89-point data collection and priority optimization
├── easy0DTE/                # Easy 0DTE Strategy (options overlay)
└── easyCollector/           # Market data collector (separate deploy)
```

---

## Quick start

**Config**

- Copy config templates from `configs/` and `easy0DTE/configs/`.
- Create your `.env` files with broker credentials and API keys.
- Do not commit secrets. See [.env.example](.env.example) for placeholder keys.

**Install**

```bash
pip install -r requirements.txt
```

(If using the overlay) install `easy0DTE/requirements.txt`.

**Run locally**

```bash
python main.py
```

(Cloud entry) `python cloud_run_entry.py`

**Collector**

See [easyCollector/README.md](easyCollector/README.md) for collector run + deploy (Polygon/Coinbase/Firestore).

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture + module organization |
| [docs/Strategy.md](docs/Strategy.md) | ORB strategy timing, validation, performance context |
| [docs/Risk.md](docs/Risk.md) | Risk controls, sizing, constraints |
| [docs/OAuth.md](docs/OAuth.md) | E*TRADE OAuth token management |
| [docs/Data.md](docs/Data.md) | Watchlists, ORB capture, data handling |
| [docs/Alerts.md](docs/Alerts.md) | Alert types + formatting |
| [docs/Cloud.md](docs/Cloud.md) | GCP Cloud Run deployment steps |
| **easyCollector** | [easyCollector/README.md](easyCollector/README.md), [easyCollector/EXECUTIVE_SUMMARY.md](easyCollector/EXECUTIVE_SUMMARY.md) |

---

## Deployment

- **ORB + 0DTE:** Build with root `Dockerfile`; deploy to Cloud Run using `cloud_run_entry.py` as entry.
- Set `GCP_PROJECT_ID`, region, and inject secrets using your secret manager.
- See [docs/Cloud.md](docs/Cloud.md).

---

## Disclaimer

This repository is provided for **educational and research purposes only** and does not constitute financial advice. Trading involves substantial risk and may result in significant losses. Always validate thoroughly in a simulation environment before any live usage.

---

## License

See [LICENSE](LICENSE) in this repository.

---

*Public repo. Set `GCP_PROJECT_ID` and your own secrets for deployment. Do not commit credentials. Revision history: [CHANGELOG.md](CHANGELOG.md).*

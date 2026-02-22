# Easy ORB 0DTE Strategy

**Automated dual-strategy system: Opening Range Breakout (ORB) for ETFs plus 0DTE options overlay for selective convex amplification—with broker-only data, multi-layer risk management, and full position monitoring.**

---

### New here?

| Goal | Where to go |
|------|-------------|
| **Run or deploy the app** | [Quick Start](#-quick-start) |
| **Understand the system** | [System Overview](#-system-overview) · [Repository structure](#repository-structure) |
| **Full documentation** | [Documentation](#-documentation) · [docs/](docs/) |
| **Cloud deploy** | [docs/Cloud.md](docs/Cloud.md) |

---

## System overview

This repository contains:

1. **Easy ORB 0DTE Strategy** — Main application (ORB capture, signal collection, execution, monitoring). Deploys as a single Cloud Run service.
2. **easy0dte** — Easy 0DTE Strategy module (options signal generation, convex eligibility, execution).
3. **easyCollector** — Standalone market data collection service (ORB/SIGNAL/OUTCOME snapshots for US 0DTE and crypto). Deploys separately.

**Core philosophy:** ORB Strategy trades breakouts from the first 15 minutes of market action. 0DTE Strategy applies only to the highest-conviction setups (selective convex amplification). **Easy 0DTE = Gamma > Leverage.**

**Typical flow:** OAuth → Good Morning → ORB Capture → validation candle (7:00/7:15) → Signal Collection → execution → position monitoring → EOD.

---

## Repository structure

```
easyORBStrategy/
├── main.py                  # Application entry (local)
├── cloud_run_entry.py       # Cloud Run entry point
├── requirements.txt
├── Dockerfile
├── modules/                 # ORB strategy: data, risk, alerts, execution, GCS
├── configs/                 # Strategy configs + templates (*.template, strategies.env, modes/, etc.)
├── data/
│   ├── watchlist/           # core_list.csv, 0dte_list.csv, sentiment/orb mappings
│   ├── holidays_custom.json # Holiday filter (custom)
│   └── holidays_future_proof.json # Holiday filter (future-proof)
├── priority_optimizer/      # 89-point data collection and priority optimization
│   ├── *.py                 # Collection and recovery scripts
│   ├── priority_optimizer/  # Package
│   ├── docs/                # Reference docs
│   └── daily_data/, 0dte_data/, ... # Generated (gitignored)
├── easy0DTE/                # Easy 0DTE Strategy (options overlay)
│   ├── modules/             # 0DTE signal manager, options API, execution, exit
│   ├── configs/
│   ├── requirements.txt
│   └── VERSION.txt
└── easyCollector/           # Market data collector (separate deploy)
    ├── backend/             # FastAPI app, clients, services, storage
    ├── scripts/             # Validation, smoke tests
    ├── README.md            # Collector-specific docs
    └── EXECUTIVE_SUMMARY.md # Collector production details
```

---

## Quick start

1. **Config**: Copy config templates from `configs/` and `easy0DTE/configs/`; create your `.env` files with broker credentials and API keys. Do not commit secrets.
2. **Dependencies**: `pip install -r requirements.txt` (root and `easy0DTE/` if needed).
3. **Local run**: `python main.py` (or use `cloud_run_entry` for the web entry used in Cloud Run).
4. **Collector**: See `easyCollector/README.md` for local run and deploy (Polygon, Coinbase, Firestore).

---

## Documentation

| Doc | Purpose |
|-----|---------|
| **[docs/Alerts.md](docs/Alerts.md)** | Alert system (Telegram, types, formatting) |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System architecture and module organization |
| **[docs/Cloud.md](docs/Cloud.md)** | Cloud deployment (GCP, deploy steps, monitoring) |
| **[docs/Data.md](docs/Data.md)** | Data management, watchlist, ORB capture |
| **[docs/OAuth.md](docs/OAuth.md)** | E*TRADE OAuth token management |
| **[docs/Risk.md](docs/Risk.md)** | Risk management and position sizing |
| **[docs/Strategy.md](docs/Strategy.md)** | ORB strategy, timing, validation, performance |
| **easyCollector** | [easyCollector/README.md](easyCollector/README.md), [easyCollector/EXECUTIVE_SUMMARY.md](easyCollector/EXECUTIVE_SUMMARY.md) |
| **Config** | `configs/*.template`, `easyCollector/SECRETS.md` |

---

## Deployment

- **ORB + 0DTE**: Build with root `Dockerfile`; deploy to Cloud Run. Use **`cloud_run_entry.py`** as the container entrypoint. Set `GCP_PROJECT_ID`, region, and inject secrets via your secret manager. See [docs/Cloud.md](docs/Cloud.md).
- **easyCollector**: Deploy from `easyCollector/` (see [easyCollector/README.md](easyCollector/README.md) and [easyCollector/EXECUTIVE_SUMMARY.md](easyCollector/EXECUTIVE_SUMMARY.md)).

---

## License

See [LICENSE](LICENSE) in this repository.

---

*Public repo. Set `GCP_PROJECT_ID` and your own secrets for deployment. Do not commit credentials.*

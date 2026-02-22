# Easy Collector

**Standalone market data collection service** that captures structured snapshots at ORB, SIGNAL, and OUTCOME decision points for **US 0DTE symbols** and **crypto futures**—for research and ML-ready datasets.

---

### New here?

| Goal | Where to go |
|------|-------------|
| **Run locally** | [Quick Start](#quick-start) |
| **Deploy to Cloud Run** | [Deployment](#deployment) |
| **Set up Cloud Scheduler** | [Cloud Scheduler Setup](#cloud-scheduler-setup) |
| **Secrets & API keys** | [SECRETS.md](SECRETS.md) (or `.env.example`) |
| **Production details & verification** | [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) |
| **Troubleshoot data sources** | [docs/DATA_SOURCES_TROUBLESHOOTING.md](docs/DATA_SOURCES_TROUBLESHOOTING.md) |

---

## Overview

Easy Collector runs **standalone**. It fetches market data via **Polygon** (US primary), **Coinbase** (crypto), and fallbacks; computes technical indicators and edge-based outcome labels; and writes idempotent snapshots to **Firestore**. Designed for Google Cloud Run and Cloud Scheduler.

### Purpose

For **US 0DTE**, the service is optimized to learn:
- Directional bias after the market open
- Which setups produce early, sustained moves
- Which conditions lead to chop, decay, or low-expectancy days to avoid

For **crypto futures**, it is optimized to learn:
- Which session opening ranges (London, US, Asia, Reset) predict the largest directional price expansions
- Continuation vs reversal behavior between sessions
- Technical signatures that precede high-distance long or short moves

### Key Features

- **Unified schema**: Single snapshot structure for ORB and indicator-based strategies
- **Idempotent collection**: Deterministic document IDs prevent duplicate snapshots
- **Holiday awareness**: US holidays, low-volume days, and early closes
- **Session-based crypto**: London, US, Reset, and Asia sessions
- **Dry-run mode**: Test collection without writing to Firestore
- **Cloud Run ready**: Serverless deployment with Cloud Scheduler
- **US cache layer**: Prefetch once per day, ~98% API call reduction
- **Two-layer data design**: Indicator slab + snapshot window
- **Edge-based labels**: `edge = MFE - k*MAE - cost_penalty` (v2.0)
- **Session VWAP**: VWAP from session start (not full history)

## Architecture

```
┌─────────────────┐
│ Cloud Scheduler │ (Triggers collection)
└────────┬────────┘
         ▼
┌─────────────────┐     ┌────────┐ ┌──────────┐
│   FastAPI App   │────▶│US Cache│ │ Coinbase │
└────────┬────────┘     └────────┘ └──────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Snapshot Service│────▶│ Indicator Svc  │────▶│ Outcome Labels  │
└────────┬────────┘     └─────────────────┘     └────────┬────────┘
         ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│ Firestore Repo  │◀─────────────────────────────│ (Edge-based)   │
└─────────────────┘                              └─────────────────┘
```

**Data flow**: US data via Polygon (primary) or Alpaca/yfinance; crypto via Coinbase. US cache prefetches once per day. Indicators and outcome labels (edge formula) are computed; snapshots are written to Firestore.

## Quick Start

### 1. Environment and secrets

For **local runs**, put your API key in `secretsprivate/.env`:

```bash
./scripts/setup_local_secrets.sh   # creates secretsprivate/.env from .env.example
```

Edit `secretsprivate/.env` and set `POLYGON_API_KEY`. For production, use your cloud provider’s secret manager (e.g. Google Secret Manager). See **SECRETS.md** for patterns.

Optional env:

```bash
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
USE_FIRESTORE_EMULATOR=false
TIMEFRAME=5m
INDICATOR_LOOKBACK_BARS=120
DRY_RUN=false
```

### 2. Local development

**With Firestore emulator:**

```bash
pip install -r requirements.txt
gcloud emulators firestore start --host-port=localhost:8080
export USE_FIRESTORE_EMULATOR=true
export FIRESTORE_EMULATOR_HOST=localhost:8080
cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

**With real Firestore:**

```bash
gcloud auth application-default login
cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

### 3. Test endpoints

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/collect/us/orb -H "Content-Type: application/json" -d '{}'
curl -X POST http://localhost:8080/collect/crypto/US/orb -H "Content-Type: application/json" -d '{}'
```

## Deployment

### Cloud Run

- Build with the included `Dockerfile` and `cloudbuild.yaml`.
- Set `GCP_PROJECT_ID` and region for your project.
- Provide a symbol list at `data/watchlist/0dte_list.csv` (or configure path in config).
- Store `POLYGON_API_KEY` in Secret Manager and grant the Cloud Run service account access to the secret.
- Deploy (e.g. `gcloud run deploy` or your CI); use `--set-secrets=POLYGON_API_KEY=your-secret:latest` so the key is not in the image.

### Cloud Scheduler (example)

**US (weekdays):**

```bash
# ORB 9:45 ET, SIGNAL 10:30 ET, OUTCOME 15:55 ET
gcloud scheduler jobs create http us-orb-collect \
  --location=us-central1 \
  --schedule="45 9 * * 1-5" \
  --time-zone="America/New_York" \
  --uri="https://YOUR-SERVICE-URL.run.app/collect/us/orb" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{}'
```

**Crypto (example — US session ORB 8:15 ET):**

```bash
gcloud scheduler jobs create http crypto-us-orb \
  --location=us-central1 \
  --schedule="15 8 * * *" \
  --time-zone="America/New_York" \
  --uri="https://YOUR-SERVICE-URL.run.app/collect/crypto/US/orb" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{}'
```

Replace `YOUR-SERVICE-URL` with your Cloud Run service URL.

## Schedule reference

| Market | Snapshot | Time (ET) | Description |
|--------|----------|-----------|-------------|
| US | ORB | 9:45 AM weekdays | Opening range (9:30–9:45) |
| US | SIGNAL | 10:30 AM weekdays | Pre-execution window |
| US | OUTCOME | 3:55 PM weekdays | Pre-close outcome |
| Crypto | ORB/SIGNAL/OUTCOME | Per session | London, US, Reset, Asia |

## Symbols

- **US**: Loaded from `data/watchlist/0dte_list.csv` (tiered). SPX uses SPY proxy with 10× scaling for intraday.
- **Crypto**: BTC-PERP, ETH-PERP, SOL-PERP, XRP-PERP (Coinbase).

## Data model

- **Document ID**: `{market}_{symbol}_{session}_{snapshot_type}_{YYYYMMDD}_{HHMM_ET}`
- **Collections**: `snapshots` (idempotent), `runs` (run logs)

## API reference

- `GET /health` — Health check
- `GET /version` — Service version
- `GET /debug/us/provider_smoke?symbol=SPY&bars=50` — US data smoke test
- `GET /debug/crypto/product_smoke?symbol=BTC-PERP` — Crypto smoke test
- `POST /collect/us/orb`, `/collect/us/signal`, `/collect/us/outcome` — US collection
- `POST /collect/crypto/{session}/orb` (and signal/outcome) — Crypto collection

Request body (optional): `{"timestamp_et": "2025-01-08 09:45:00"}`. Response includes `success`, `summary` (counts, errors, duration).

## Dry-run mode

```bash
export DRY_RUN=true
```

Pipeline runs but snapshots are not written to Firestore; useful for testing.

## Holiday and early close handling

US early closes (e.g. July 3, Black Friday, Christmas Eve) are detected; OUTCOME time is adjusted. Snapshots include `is_market_closed`, `is_early_close`, `early_close_time_et`, etc.

## Troubleshooting

- **Data sources**: Use debug endpoints (`/debug/us/provider_smoke`, `/debug/crypto/product_smoke`). See [docs/DATA_SOURCES_TROUBLESHOOTING.md](docs/DATA_SOURCES_TROUBLESHOOTING.md).
- **Firestore**: `gcloud auth application-default login`; use your project: `gcloud firestore collections list --project YOUR_PROJECT_ID`.
- **Cloud Run logs**: `gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=YOUR_SERVICE_NAME" --project YOUR_PROJECT_ID`.

## Project structure

```
easyCollector/
├── backend/app/          # FastAPI app, config, clients, models, services, storage, utils
├── scripts/              # Validation, smoke tests, deploy helpers
├── docs/                 # Documentation
├── requirements.txt
├── Dockerfile
├── cloudbuild.yaml
└── README.md
```

## License

See [LICENSE](LICENSE) in this repository.

## Contributing

Open an issue or pull request on the repository for bugs, docs, or features.

---

*README last updated: February 2026. For production-grade details and verification, see [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md).*

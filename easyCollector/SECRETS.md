# Easy Collector – Secrets and Deployment

How to manage API keys and other secrets for **local development** and **Cloud Run deployment**.

---

## Local development

- Put sensitive values in **`secretsprivate/.env`** (never commit; it is gitignored).
- `config.py` loads `secretsprivate/.env` when it exists, so `POLYGON_API_KEY` (and optional `ALPACA_KEY`, `ALPACA_SECRET`) are read from there.
- Setup: `cp secretsprivate/.env.example secretsprivate/.env` and fill in the keys. You can reuse `POLYGON_API_KEY` from Ultima Bot’s `backend/.env` if available.

See `secretsprivate/README.md` for details.

---

## Deployment (Cloud Run)

**Do not put secrets in the image or in `cloudbuild.yaml`.** Use **Google Secret Manager** and inject them at deploy time with `--set-secrets`.

### 1. Create or update the secret in Secret Manager

**Option A – sync from `secretsprivate/.env` (recommended):**

```bash
./secretsprivate/ensure_secret_manager_polygon.sh
```

Reads `POLYGON_API_KEY` from `secretsprivate/.env`, creates `polygon-api-key` if missing, or adds a new version.

**Option B – manual:**

```bash
export PROJECT_ID="${GCP_PROJECT_ID:-easy-etrade-strategy}"

# Create
echo -n "YOUR_POLYGON_API_KEY" | gcloud secrets create polygon-api-key --project="$PROJECT_ID" --data-file=-

# Or add version if it exists
echo -n "YOUR_POLYGON_API_KEY" | gcloud secrets versions add polygon-api-key --project="$PROJECT_ID" --data-file=-
```

To check whether `polygon-api-key` exists: `./scripts/check_secret_manager.sh`.

**Full readiness (secret, IAM, deploy wiring):** `./scripts/check_polygon_secret_ready.sh`  
Optional: `./scripts/check_polygon_secret_ready.sh --validate-key` to test the key against Polygon's API (uses Secret Manager, or `secretsprivate/.env` if SM read fails). To test a **new key before** adding to Secret Manager: `./scripts/validate_polygon_key.sh --key=YOUR_KEY` or `./scripts/check_polygon_secret_ready.sh --key=YOUR_KEY`.

### 2. IAM: allow Cloud Run to read the secret

The Cloud Run service account needs `roles/secretmanager.secretAccessor` on the secret (or on the project). For least privilege, grant on the secret only:

```bash
export PROJECT_ID="${GCP_PROJECT_ID:-easy-etrade-strategy}"
export SA_EMAIL="YOUR_CLOUD_RUN_SERVICE_ACCOUNT@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud secrets add-iam-policy-binding polygon-api-key \
  --project="$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"
```

If you use the default Compute Engine service account, it’s `PROJECT_NUMBER-compute@developer.gserviceaccount.com`; otherwise use the SA you set in `--service-account` when deploying.

### 3. Deploy with `--set-secrets`

For end-to-end deploy (build, push, deploy with `--set-secrets`), run **`./deploy-collector.sh`** from the ORB project root (parent of `easyCollector`). It wires `POLYGON_API_KEY=polygon-api-key:latest` into the Cloud Run service.

To deploy manually, pass `--set-secrets` when running `gcloud run deploy`:

```bash
gcloud run deploy easy-collector \
  --image="gcr.io/${PROJECT_ID}/easy-collector:latest" \
  --set-secrets="POLYGON_API_KEY=polygon-api-key:latest" \
  # ... other flags (--region, --platform, --allow-unauthenticated, etc.)
```

This mounts `polygon-api-key:latest` into the `POLYGON_API_KEY` environment variable. No secrets in the image or in build config.

### 4. Optional: Alpaca

If you use Alpaca as primary or fallback:

1. Create secrets, e.g. `alpaca-key` and `alpaca-secret`.
2. Grant the Cloud Run SA `roles/secretmanager.secretAccessor` on them.
3. Add to `--set-secrets`:
   - `ALPACA_KEY=alpaca-key:latest`
   - `ALPACA_SECRET=alpaca-secret:latest`

---

## Polygon for US Market 0DTE: end-to-end chain

| Step | What | Where |
|------|------|-------|
| 1 | Secret `polygon-api-key` in Google Secret Manager | `gcloud secrets`, `ensure_secret_manager_polygon.sh` |
| 2 | Cloud Run SA has `roles/secretmanager.secretAccessor` on `polygon-api-key` | `gcloud secrets add-iam-policy-binding` |
| 3 | Deploy passes `--set-secrets POLYGON_API_KEY=polygon-api-key:latest` | `deploy-collector.sh`, `scripts/deploy.sh` |
| 4 | Container env `POLYGON_API_KEY` | Injected by Cloud Run from the secret |
| 5 | `config.polygon_api_key` | Read from env `POLYGON_API_KEY` (pydantic-settings) |
| 6 | `PolygonClient(api_key=settings.polygon_api_key)` | `clients/polygon_client.py` |
| 7 | US 0DTE symbols, 5m aggs | `GET /v2/aggs/ticker/{ticker}/range/5/minute/...?apiKey=...` |

**Verify:** `./scripts/check_polygon_secret_ready.sh` and, after deploy, `./scripts/check_polygon_coinbase.sh`.

---

## Summary

| Where        | Store                            | Notes                                                                 |
|-------------|-----------------------------------|-----------------------------------------------------------------------|
| Local dev   | `secretsprivate/.env`             | Loaded by `config.py` when the file exists; `.dockerignore` excludes it. |
| Cloud Run   | Secret Manager + `--set-secrets`  | e.g. `POLYGON_API_KEY=polygon-api-key:latest`; no secrets in the image. |

For a full deploy checklist, see `docs/Sessions/Jan 23 Session/DEPLOYMENT_READINESS.md`. For data-source and snapshot troubleshooting (Polygon 403, Coinbase, `volume_delta`), see **`Data.md`** §6.

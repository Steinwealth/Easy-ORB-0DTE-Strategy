# Public repo — what’s included and excluded

This folder is a **public-safe copy** of the Easy ORB 0DTE Strategy application. Production deployment uses the original project; this copy is for sharing on GitHub.

## Included

- **Application**: `main.py`, `cloud_run_entry.py`, `modules/`, `easy0DTE/` (code + config templates), `easyCollector/` (code + safe README/EXECUTIVE_SUMMARY), **priority_optimizer/** (scripts, package, docs; generated data dirs are gitignored).
- **Config**: All non-secret configs: `*.template`, `*.example`, `strategies.env`, `position-sizing.env`, `risk-management.env`, `symbol-scoring.env`, `trading-parameters.env`, `performance.env`, `slip-guard.env`, `performance-targets.env`, `data-providers.env`, `optimized_env_template.env`, `configs/modes/` (standard, advanced, quantum), `configs/environments/` (development, production). No `deployment.env`, `etrade-oauth.env`, `alerts.env`, or other credential-bearing files.
- **Data**: `data/watchlist/core_list.csv`, `data/watchlist/0dte_list.csv`, `data/holidays_custom.json`, `data/holidays_future_proof.json`, and watchlist metadata (sentiment/orb mappings).
- **Docs**: Public [README.md](README.md); [docs/](docs/) with public-safe Alerts.md, ARCHITECTURE.md, Cloud.md, Data.md, OAuth.md, Risk.md, Strategy.md (project IDs and internal URLs replaced with placeholders). Collector docs use safe versions.

## Excluded (not in this copy)

- **Secrets**: `secretsprivate/`, any `.env` with credentials, `configs/*.env` (except templates).
- **Internal docs**: Session notes, CloudSecrets, PrivateSecrets, internal deploy URLs and project IDs.
- **Other**: `ETradeOAuth/`, `priority_optimizer/`, `scripts/` (deploy/ops scripts with project IDs), full `docs/` tree from production.
- **easyCollector**: `easyCollector/docs/Sessions/`, `easyCollector/secretsprivate/`.

## Before you push to GitHub

1. Set `GCP_PROJECT_ID` (and secrets) in your environment; do not rely on hardcoded project IDs in code.
2. Add a `LICENSE` file if you want.
3. Ensure no `.env` or credential files are committed.

See `easyCollector/PUBLIC_REPO_AUDIT.md` for collector-specific sanitization notes.

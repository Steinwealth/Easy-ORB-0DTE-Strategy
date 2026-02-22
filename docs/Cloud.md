# Cloud Deployment Guide
## Easy ORB Strategy – Public Cloud User Guide

**Last Updated**: February 19, 2026  
**Version**: Rev 00259 + Cloud Run entrypoint (cloud_run_entry.py); Feb19: prefetch-validation-715 job, validation candle batch (25/call)  
**Purpose**: Complete, **shareable** guide for deploying, managing, and monitoring the Easy ORB Strategy (ORB ETF + 0DTE Options) and Easy Collector on cloud platforms. Safe to give to another user or use as an AI reconstruction guide. Contains **no private secrets or project-specific identifiers**.

**How to use this guide**
- Replace all placeholders (`YOUR_PROJECT_ID`, `YOUR_SERVICE_URL`, etc.) with your own values.
- For **credentials** (API keys, Telegram tokens, OAuth): use your environment or secret manager; do not commit them.
- For **project-specific** deploy scripts and URLs: maintain your own runbook; this doc is shareable and uses placeholders only.

---

## 📋 **Table of Contents**

1. [Overview](#overview)
2. [Cloud Architecture](#cloud-architecture)
3. [Supported Cloud Platforms](#supported-cloud-platforms)
4. [Google Cloud Platform (GCP) - Primary Platform](#google-cloud-platform-gcp---primary-platform)
5. [Alternative Cloud Platforms](#alternative-cloud-platforms)
6. [Scale-to-Zero Configuration](#scale-to-zero-configuration)
7. [Cloud Scheduler Jobs](#cloud-scheduler-jobs)
8. [Cloud Cleanup Policies](#cloud-cleanup-policies)
9. [Cost Analysis](#cost-analysis)
10. [Deployment Steps](#deployment-steps)
11. [Monitoring & Logging](#monitoring--logging)
12. [Security Configuration](#security-configuration) (includes Google-recommended best practices)
13. [Data Persistence](#data-persistence)
14. [Production Readiness](#production-readiness)

---

## 🎯 **Overview**

The Easy ORB Strategy system consists of three main components:

1. **ORB Strategy**: Trading signals for US market stocks and leveraged ETFs
2. **0DTE Strategy**: Options trading signals for 0DTE options
3. **Easy Collector**: ML data collection service for US equities and crypto assets

All components share:
- Cloud project and infrastructure
- Alert system (Telegram notifications)
- Exit system (trailing stops, take profit)
- OAuth token management
- Trade persistence

**Deployment Status**: ✅ Production-ready with 98/100 deployment readiness score

---

## 🏗️ **Cloud Architecture**

### **Core Architecture Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│                    Cloud Platform (GCP/AWS/etc.)          │
├─────────────────────────────────────────────────────────────┤
│  Container Service (Cloud Run / ECS / Lambda)               │
│  ├── Trading Service (ORB + 0DTE Strategies)              │
│  │   ├── Prime Trading Manager                             │
│  │   ├── Prime Data Manager                                │
│  │   ├── Prime Signal Generator                            │
│  │   ├── Trading Executor (Demo/Live Mode)                │
│  │   ├── Prime Alert Manager                               │
│  │   └── 0DTE Strategy Manager                             │
│  ├── OAuth Service                                         │
│  │   ├── ETradeOAuth Manager                               │
│  │   ├── Token Keepalive Service                          │
│  │   └── Token Refresh Service                             │
│  └── Easy Collector Service                                │
│      ├── US Market Data Collection                          │
│      ├── Crypto Market Data Collection                     │
│      └── ML Dataset Generation                              │
├─────────────────────────────────────────────────────────────┤
│  Object Storage (GCS / S3 / etc.)                          │
│  ├── Trading State Persistence                             │
│  ├── Trade History                                         │
│  ├── Performance Logs                                      │
│  └── ML Datasets                                           │
├─────────────────────────────────────────────────────────────┤
│  Secret Management (Secret Manager / Secrets Manager)       │
│  ├── Broker API Keys                                       │
│  ├── Telegram Bot Tokens                                   │
│  └── OAuth Tokens                                          │
├─────────────────────────────────────────────────────────────┤
│  Scheduler (Cloud Scheduler / EventBridge / etc.)           │
│  ├── Trading Hours Keep-Alive                              │
│  ├── OAuth Token Refresh                                   │
│  ├── End-of-Day Reports                                    │
│  └── Automated Cleanup                                     │
├─────────────────────────────────────────────────────────────┤
│  Logging & Monitoring                                      │
│  ├── Application Logs                                      │
│  ├── Performance Metrics                                   │
│  └── Error Tracking                                        │
└─────────────────────────────────────────────────────────────┘
```

### **Service Components**

#### **1. Trading Service**
- **Purpose**: Executes ORB Strategy and 0DTE Strategy trading signals
- **Resources**: 2 vCPU, 2Gi Memory (recommended)
- **Scaling**: Scale-to-zero enabled (cost optimization)
- **Container entrypoint**: **`cloud_run_entry.py`** — starts a minimal HTTP server on `PORT` immediately so Cloud Run’s startup probe passes, then runs the full app (`main.py --cloud-mode`). This avoids startup timeouts from slow OAuth/config/Secret Manager init.
- **Features**:
  - ORB Strategy (always enabled)
  - 0DTE Strategy (enabled via `ENABLE_0DTE_STRATEGY=true`)
  - Holiday filtering
  - Portfolio health checks
  - Stealth trailing stop system
  - Alert system integration

#### **2. OAuth Service**
- **Purpose**: Manages broker OAuth token lifecycle
- **Resources**: 1 vCPU, 512Mi Memory
- **Scaling**: Scale-to-zero enabled
- **Features**:
  - Token refresh automation
  - Token expiry alerts
  - Web dashboard for manual renewal

#### **3. Easy Collector Service**
- **Purpose**: Collects market data for ML algorithm development
- **Resources**: 1-2 vCPU, 1-2Gi Memory (depending on collection volume)
- **Scaling**: Scale-to-zero enabled
- **Features**:
  - US market snapshot collection (ORB, SIGNAL, OUTCOME)
  - Crypto market snapshot collection (multiple sessions)
  - ML-ready dataset generation

---

## ☁️ **Supported Cloud Platforms**

### **Primary Platform: Google Cloud Platform (GCP)**
- ✅ **Recommended**: Fully tested and optimized
- **Services Used**: Cloud Run, Cloud Storage, Secret Manager, Cloud Scheduler, Cloud Logging, Cloud Monitoring
- **Cost**: ~$17.75-22.25/month (with scale-to-zero)
- **Documentation**: Complete setup guide below

### **Alternative Platforms**

#### **Amazon Web Services (AWS)**
- **Container Service**: AWS ECS (Fargate) or AWS Lambda
- **Storage**: S3 for trade history and state
- **Secrets**: AWS Secrets Manager
- **Scheduler**: Amazon EventBridge (CloudWatch Events)
- **Cost Estimate**: ~$20-30/month (similar configuration)

#### **Microsoft Azure**
- **Container Service**: Azure Container Instances or Azure Functions
- **Storage**: Azure Blob Storage
- **Secrets**: Azure Key Vault
- **Scheduler**: Azure Logic Apps or Timer Triggers
- **Cost Estimate**: ~$20-30/month (similar configuration)

#### **Snowflake**
- **Note**: Snowflake is primarily a data warehouse, not a compute platform
- **Use Case**: Can be used for data storage and analytics
- **Integration**: Connect via API from GCP/AWS compute services
- **Cost**: Pay-per-use data storage and compute

#### **Other Platforms**
- **Vercel / Netlify**: Not suitable (serverless functions have execution time limits)
- **DigitalOcean**: Can use App Platform (similar to Cloud Run)
- **Heroku**: Possible but more expensive (~$25-50/month)

**Recommendation**: Use GCP for best cost optimization and feature set. AWS is a viable alternative with similar costs.

---

## 🚀 **Google Cloud Platform (GCP) - Primary Platform**

### **Required APIs**

Enable the following APIs in your Google Cloud project:

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable artifactregistry.googleapis.com  # For container images
```

### **Service Account Setup**

Create service accounts with required permissions:

```bash
# Create trading service account
gcloud iam service-accounts create etrade-strategy-sa \
    --display-name="ETrade Strategy Service Account"

# Create Easy Collector scheduler account
gcloud iam service-accounts create easy-collector-scheduler \
    --display-name="Easy Collector Scheduler Account"

# Grant permissions to trading service account
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:etrade-strategy-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:etrade-strategy-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:etrade-strategy-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"

# Grant permissions to Easy Collector scheduler account
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker"
```

### **Secret Manager Setup**

**Best practice (zero-code storage):** All credentials live in Secret Manager and are injected at runtime; never in source code. See [Security Configuration](#-security-configuration) for the full checklist.

Store required secrets in Secret Manager:

```bash
# E*TRADE API credentials
echo -n "your-consumer-key" | gcloud secrets create etrade-consumer-key --data-file=-
echo -n "your-consumer-secret" | gcloud secrets create etrade-consumer-secret --data-file=-

# Telegram credentials
echo -n "your-telegram-bot-token" | gcloud secrets create telegram-bot-token --data-file=-
echo -n "your-telegram-chat-id" | gcloud secrets create telegram-chat-id --data-file=-

# E*TRADE OAuth tokens (will be populated by OAuth service)
gcloud secrets create etrade-prod-access-token --data-file=-
gcloud secrets create etrade-prod-refresh-token --data-file=-
```

#### **⚠️ CRITICAL: Secret Manager Cost Optimization**

**Secret Manager charges $0.06 per version per month.** To prevent excessive costs:

1. **Automatic Cleanup**: ✅ **DEPLOYED** (February 9, 2026)
   - **Firebase OAuth App**: `oauth_backend.py` automatically deletes old secret versions when tokens are renewed
   - **Cloud Run Services**: `secret_manager_oauth.py` also includes cleanup (for direct token storage)
   - Both systems keep only the latest version (1 per secret)

2. **Manual Cleanup**: If old versions accumulate, run:
   ```bash
   cd "0. Strategies and Automations"
   bash scripts/cleanup_secrets_optimized.sh
   ```

3. **Best Practice**: Each secret should have only **1 version** (the latest). Monitor version counts monthly:
   ```bash
   for SECRET in etrade-oauth-prod etrade-oauth-sandbox; do
     COUNT=$(gcloud secrets versions list $SECRET --format='value(name)' | wc -l)
     echo "$SECRET: $COUNT versions (should be 1)"
   done
   ```

**Current Cost**: ~$1.20/month (20 billable versions across all projects × $0.06)  
**Expected Cost**: ~$0.78/month (13 secrets in YOUR_PROJECT_ID × 1 version × $0.06)  
**Without Cleanup**: Could exceed $200/month if thousands of versions accumulate

**Status**: ✅ Cleanup code deployed to Firebase Functions (February 9, 2026) - automatic cleanup active

### **Cloud Storage Buckets**

Create buckets for data persistence:

```bash
# Create buckets
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://YOUR_PROJECT_ID-trades/
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://YOUR_PROJECT_ID-state/
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://YOUR_PROJECT_ID-logs/

# Set lifecycle policies (optional - auto-delete after 90 days)
gsutil lifecycle set lifecycle.json gs://YOUR_PROJECT_ID-trades/
```

---

## ⚙️ **Scale-to-Zero Configuration**

### **Overview**

Scale-to-zero allows services to shut down when not in use, dramatically reducing costs. The system automatically wakes up when needed via Cloud Scheduler jobs.

### **Configuration**

**Trading Service**:
```bash
gcloud run deploy YOUR_PROJECT_ID \
    --min-instances 0 \
    --max-instances 1 \
    --no-cpu-throttling \
    # ... other settings
```

**OAuth Service**:
```bash
gcloud run deploy YOUR_PROJECT_ID-oauth \
    --min-instances 0 \
    --max-instances 1 \
    # ... other settings
```

**Easy Collector Service**:
```bash
gcloud run deploy easy-collector \
    --min-instances 0 \
    --max-instances 1 \
    # ... other settings
```

### **Scale-to-Zero Behavior**

**Trading Days (Monday-Friday)**:
- System wakes up at 5:30 AM PT via Good Morning alert
- Stays active through EOD (1:00 PM PT / 4:00 PM ET)
- Cloud Scheduler keep-alive jobs prevent shutdown during trading hours
- Scales to zero after EOD

**Weekends (Saturday-Sunday)**:
- System scales to zero after Friday EOD
- No charges until Monday morning
- No keep-alive jobs run (cost optimization)

**Holidays**:
- System scales to zero on non-trading days
- No keep-alive jobs run (cost optimization)
- Holiday detection prevents unnecessary wake-ups

**Cold Start Impact**:
- Container listens on `PORT` within seconds (minimal server in `cloud_run_entry.py`); full trading loop then initializes in the background
- ~10-30 seconds until trading loop is fully ready when Cloud Scheduler wakes system
- Acceptable for pre-market startup (system wakes before market open)

### **Cost Impact**

**Without Scale-to-Zero**: ~$155/month (24/7 operation)  
**With Scale-to-Zero**: ~$17.75-22.25/month (only pay when running)  
**Savings**: 86-88% cost reduction

**Calculation**:
- Trading days: ~7.5 hours/day × 5 days/week = ~37.5 hours/week
- Without scale-to-zero: 168 hours/week (24/7)
- Reduction: 78% fewer compute hours

---

## ⏰ **Cloud Scheduler Jobs**

### **ORB Strategy & 0DTE Strategy Jobs**

#### **Trading Hours Keep-Alive Jobs**

These jobs keep the trading service alive during trading hours:

```bash
# Pre-market wake-up (5:00-7:00 AM PT, every 3 minutes)
gcloud scheduler jobs create http trading-hours-keepalive-1 \
    --location=us-central1 \
    --schedule="*/3 5-6 * * 1-5" \
    --time-zone="America/Los_Angeles" \
    --uri="https://YOUR_SERVICE_URL/api/health" \
    --http-method=GET \
    --description="Keep trading service alive during pre-market (5-7 AM PT)"

# Trading session start (7:00-10:00 AM PT, every 5 minutes)
gcloud scheduler jobs create http trading-hours-keepalive-2 \
    --location=us-central1 \
    --schedule="*/5 7-9 * * 1-5" \
    --time-zone="America/Los_Angeles" \
    --uri="https://YOUR_SERVICE_URL/api/health" \
    --http-method=GET \
    --description="Keep trading service alive during trading session start (7-10 AM PT)"

# Trading session continuation (10:00 AM-2:00 PM PT, every 5 minutes)
gcloud scheduler jobs create http trading-hours-keepalive-3 \
    --location=us-central1 \
    --schedule="*/5 10-13 * * 1-5" \
    --time-zone="America/Los_Angeles" \
    --uri="https://YOUR_SERVICE_URL/api/health" \
    --http-method=GET \
    --description="Keep trading service alive during trading session (10 AM-2 PM PT)"
```

#### **Good Morning Alert**

Wakes system and sends morning status:

```bash
gcloud scheduler jobs create http oauth-market-open-alert \
    --location=us-central1 \
    --schedule="30 5 * * 1-5" \
    --time-zone="America/Los_Angeles" \
    --uri="https://YOUR_SERVICE_URL/api/alerts/market-open" \
    --http-method=POST \
    --description="Good Morning alert at 5:30 AM PT (8:30 AM ET)"
```

#### **7:00 AM PT – Validation Candle Open (Required for signal collection)**

Captures broker prices at 7:00 AM PT as the **open** for the 7:00–7:15 validation candle (batched 25/call, same as ORB). At 7:15 the service uses current quotes as **close** and computes GREEN/RED for each symbol. Without this job, cold start at 7:15 yields all NEUTRAL and no signals.

```bash
gcloud scheduler jobs create http validation-candle-700 \
    --location=us-central1 \
    --schedule="0 7 * * 1-5" \
    --time-zone="America/Los_Angeles" \
    --uri="https://YOUR_SERVICE_URL/api/alerts/validation-candle-700" \
    --http-method=POST \
    --description="7:00 AM PT: capture open prices for 7:00-7:15 validation candle (broker-only)"
```

#### **7:15 AM PT – Prefetch Validation Candle (Recommended for scale-to-zero)**

Runs the 7:00 open + 7:15 close prefetch so the validation candle (GREEN/RED per symbol) is ready for signal collection even when a different instance runs at 7:15. Same batched quote path as ORB (25/call). Create this job if the trading loop may not run at 7:15 (e.g. scale-to-zero).

```bash
gcloud scheduler jobs create http prefetch-validation-715 \
    --location=us-central1 \
    --schedule="15 7 * * 1-5" \
    --time-zone="America/Los_Angeles" \
    --uri="https://YOUR_SERVICE_URL/api/alerts/prefetch-validation-715" \
    --http-method=POST \
    --description="7:15 AM PT: prefetch 7:00 open + 7:15 close for validation candle"
```

#### **OAuth Token Keep-Alive**

Hourly token refresh to prevent expiry:

```bash
# Production OAuth keep-alive (every hour at :00)
gcloud scheduler jobs create http oauth-keepalive-prod \
    --location=us-central1 \
    --schedule="0 * * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_OAUTH_SERVICE_URL/api/oauth/keepalive/prod" \
    --http-method=POST \
    --description="Production OAuth token keep-alive (hourly)"

# Sandbox OAuth keep-alive (every hour at :30)
gcloud scheduler jobs create http oauth-keepalive-sandbox \
    --location=us-central1 \
    --schedule="30 * * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_OAUTH_SERVICE_URL/api/oauth/keepalive/sandbox" \
    --http-method=POST \
    --description="Sandbox OAuth token keep-alive (hourly)"
```

#### **OAuth Midnight Alert**

Daily token expiry reminder:

```bash
gcloud scheduler jobs create http oauth-midnight-alert \
    --location=us-central1 \
    --schedule="0 0 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_SERVICE_URL/api/alerts/oauth-expiry" \
    --http-method=POST \
    --description="OAuth token expiry alert at midnight ET"
```

#### **End-of-Day Report**

Daily performance summary:

```bash
gcloud scheduler jobs create http end-of-day-report \
    --location=us-central1 \
    --schedule="5 16 * * 1-5" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_SERVICE_URL/api/end-of-day-report" \
    --http-method=POST \
    --description="End-of-day report at 4:05 PM ET (1:05 PM PT)"
```

### **Easy Collector Jobs**

#### **US Market Snapshots (Weekdays Only)**

```bash
# US ORB (9:45 AM ET weekdays)
gcloud scheduler jobs create http easy-collector-us-orb \
    --location=us-central1 \
    --schedule="45 9 * * 1-5" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/us/orb" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --description="Collect US ORB snapshots at 9:45 AM ET on weekdays"

# US SIGNAL (10:30 AM ET weekdays)
gcloud scheduler jobs create http easy-collector-us-signal \
    --location=us-central1 \
    --schedule="30 10 * * 1-5" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/us/signal" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --description="Collect US SIGNAL snapshots at 10:30 AM ET on weekdays"

# US OUTCOME (3:55 PM ET weekdays)
gcloud scheduler jobs create http easy-collector-us-outcome \
    --location=us-central1 \
    --schedule="55 15 * * 1-5" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/us/outcome" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --description="Collect US OUTCOME snapshots at 3:55 PM ET on weekdays"
```

#### **Crypto Market Snapshots (Daily - All Sessions)**

**London Session**:
```bash
# London ORB (3:15 AM ET daily)
gcloud scheduler jobs create http easy-collector-crypto-london-orb \
    --location=us-central1 \
    --schedule="15 3 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/LONDON/orb" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "LONDON"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com

# London SIGNAL (4:00 AM ET daily)
gcloud scheduler jobs create http easy-collector-crypto-london-signal \
    --location=us-central1 \
    --schedule="0 4 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/LONDON/signal" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "LONDON"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com

# London OUTCOME (2:55 AM ET next day)
gcloud scheduler jobs create http easy-collector-crypto-london-outcome \
    --location=us-central1 \
    --schedule="55 2 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/LONDON/outcome" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "LONDON"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

**US Session**:
```bash
# US ORB (8:15 AM ET daily)
gcloud scheduler jobs create http easy-collector-crypto-us-orb \
    --location=us-central1 \
    --schedule="15 8 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/US/orb" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "US"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com

# US SIGNAL (9:00 AM ET daily)
gcloud scheduler jobs create http easy-collector-crypto-us-signal \
    --location=us-central1 \
    --schedule="0 9 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/US/signal" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "US"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com

# US OUTCOME (4:55 PM ET daily)
gcloud scheduler jobs create http easy-collector-crypto-us-outcome \
    --location=us-central1 \
    --schedule="55 16 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/US/outcome" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "US"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

**Reset Session**:
```bash
# Reset ORB (5:15 PM ET daily)
gcloud scheduler jobs create http easy-collector-crypto-reset-orb \
    --location=us-central1 \
    --schedule="15 17 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/RESET/orb" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "RESET"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Reset SIGNAL (6:00 PM ET daily)
gcloud scheduler jobs create http easy-collector-crypto-reset-signal \
    --location=us-central1 \
    --schedule="0 18 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/RESET/signal" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "RESET"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Reset OUTCOME (6:55 PM ET daily)
gcloud scheduler jobs create http easy-collector-crypto-reset-outcome \
    --location=us-central1 \
    --schedule="55 18 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/RESET/outcome" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "RESET"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

**Asia Session**:
```bash
# Asia ORB (7:15 PM ET daily)
gcloud scheduler jobs create http easy-collector-crypto-asia-orb \
    --location=us-central1 \
    --schedule="15 19 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/ASIA/orb" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "ASIA"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Asia SIGNAL (8:00 PM ET daily)
gcloud scheduler jobs create http easy-collector-crypto-asia-signal \
    --location=us-central1 \
    --schedule="0 20 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/ASIA/signal" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "ASIA"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Asia OUTCOME (2:55 AM ET next day)
gcloud scheduler jobs create http easy-collector-crypto-asia-outcome \
    --location=us-central1 \
    --schedule="55 2 * * *" \
    --time-zone="America/New_York" \
    --uri="https://YOUR_COLLECTOR_URL/collect/crypto/ASIA/outcome" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"session": "ASIA"}' \
    --oidc-service-account-email=easy-collector-scheduler@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### **Automated Cleanup Job**

Weekly cleanup of container images and Cloud Run revisions:

```bash
gcloud scheduler jobs create http gcr-image-cleanup-weekly \
    --location=us-central1 \
    --schedule="0 2 * * 0" \
    --time-zone="America/Los_Angeles" \
    --uri="https://YOUR_SERVICE_URL/api/cleanup/images" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"cleanup_images": true, "cleanup_revisions": true}' \
    --description="Weekly cleanup of container images and revisions (Sunday 2:00 AM PT)"
```

### **Complete Job Summary**

| Job Name | Schedule | Purpose | Strategy |
|----------|----------|---------|----------|
| `trading-hours-keepalive-1` | Every 3 min, 5-7 AM PT weekdays | Pre-market wake-up | ORB + 0DTE |
| `trading-hours-keepalive-2` | Every 5 min, 7-10 AM PT weekdays | Trading session start | ORB + 0DTE |
| `trading-hours-keepalive-3` | Every 5 min, 10 AM-2 PM PT weekdays | Trading session continuation | ORB + 0DTE |
| `oauth-market-open-alert` | 5:30 AM PT weekdays | Good Morning alert | ORB + 0DTE |
| `validation-candle-700` | 7:00 AM PT weekdays | Capture open for 7:00-7:15 bar (broker, 25/call) | ORB + 0DTE |
| `prefetch-validation-715` | 7:15 AM PT weekdays | Prefetch 7:00 open + 7:15 close (scale-to-zero) | ORB + 0DTE |
| `oauth-keepalive-prod` | Hourly at :00 | Production OAuth refresh | ORB + 0DTE |
| `oauth-keepalive-sandbox` | Hourly at :30 | Sandbox OAuth refresh | ORB + 0DTE |
| `oauth-midnight-alert` | Midnight ET daily | Token expiry reminder | ORB + 0DTE |
| `end-of-day-report` | 4:05 PM ET weekdays | Daily performance report | ORB + 0DTE |
| `gcr-image-cleanup-weekly` | Sunday 2:00 AM PT | Automated cleanup | ORB + 0DTE + Collector |
| `easy-collector-us-orb` | 9:45 AM ET weekdays | US ORB snapshots | Collector |
| `easy-collector-us-signal` | 10:30 AM ET weekdays | US SIGNAL snapshots | Collector |
| `easy-collector-us-outcome` | 3:55 PM ET weekdays | US OUTCOME snapshots | Collector |
| `easy-collector-crypto-*` | Various (see above) | Crypto snapshots (12 jobs) | Collector |

**Total Jobs**: ~24 Cloud Scheduler jobs

---

## 🧹 **Cloud Cleanup Policies**

### **Container Image Cleanup**

**Automated Cleanup** (via Cloud Scheduler):
- **Schedule**: Weekly (Sunday 2:00 AM PT)
- **Retention Policy**: 
  - Keep last 10 images (regardless of age)
  - Keep images from last 30 days
  - Always keep images tagged with `latest`
- **Expected Savings**: 85% reduction in stored images

**Manual Cleanup Script**:
```bash
# Run cleanup script
./scripts/cleanup_old_images.sh

# Or via API endpoint
curl -X POST https://YOUR_SERVICE_URL/api/cleanup/images \
  -H "Content-Type: application/json" \
  -d '{"cleanup_images": true}'
```

### **Cloud Run Revision Cleanup**

**Automated Cleanup** (via Cloud Scheduler):
- **Schedule**: Weekly (Sunday 2:00 AM PT)
- **Retention Policy**: Keep last 20 revisions per service
- **Protection**: Never delete revisions with active traffic
- **Expected Savings**: 91% reduction in stored revisions

**Manual Cleanup Script**:
```bash
# Run cleanup script
./scripts/cleanup_old_revisions.sh

# Or via API endpoint
curl -X POST https://YOUR_SERVICE_URL/api/cleanup/images \
  -H "Content-Type: application/json" \
  -d '{"cleanup_revisions": true}'
```

### **Cloud Storage Lifecycle Policies**

**Build Artifacts Cleanup**:
- **Policy**: Auto-delete files older than 30 days
- **Applies To**: `*_cloudbuild` buckets, `run-sources-*` buckets
- **Note**: Does NOT affect container images (stored in Artifact Registry)

**Trade History Retention**:
- **Policy**: Keep all trade history (no auto-delete)
- **Manual Cleanup**: Optional - can set lifecycle policy for old trades

**Setup Lifecycle Policy**:
```bash
# Create lifecycle policy JSON
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF

# Apply to bucket
gsutil lifecycle set lifecycle.json gs://YOUR_PROJECT_ID_cloudbuild/
```

### **Cost Impact**

- **Image Storage Savings**: ~$0.15/month
- **Revision Storage Savings**: ~$0.16/month
- **Total Cleanup Savings**: ~$0.31/month
- **Annual Savings**: ~$3.72/year

---

## 💰 **Cost Analysis**

### **Monthly Cost Breakdown (GCP with Scale-to-Zero)**

| Service | Resource | ORB Strategy | 0DTE Strategy | Easy Collector | Total |
|---------|----------|-------------|---------------|----------------|-------|
| **Cloud Run (Trading)** | 2 vCPU, 2Gi, scale-to-zero | $11-15 | Included | - | $11-15 |
| **Cloud Run (OAuth)** | 1 vCPU, 512Mi, scale-to-zero | $2-5 | Included | - | $2-5 |
| **Cloud Run (Collector)** | 1-2 vCPU, 1-2Gi, scale-to-zero | - | - | $3-8 | $3-8 |
| **Cloud Storage** | State & data persistence | $0.50-1.50 | Included | $0.25-0.75 | $0.75-2.25 |
| **Cloud Scheduler** | Keep-alive & collection jobs | $0.10-0.25 | Included | $0.05-0.15 | $0.15-0.40 |
| **Secret Manager** | Credential storage | $0.78-1.20 | Included | $0.02-0.05 | $0.80-1.25 |
| **Cloud Logging** | Application logs | $1-2 | Included | $0.50-1 | $1.50-3 |
| **Cloud Monitoring** | Metrics & dashboards | $0.50-1 | Included | $0.25-0.50 | $0.75-1.50 |
| **Firebase Hosting** | OAuth web app | $0 (free tier) | Included | - | $0 |
| **Total** | | **$15.26-24.85** | **Included** | **$4.07-10.45** | **$19.33-35.30/month** |

### **Cost Optimization**

**Before Optimization**: ~$155/month (24/7 operation) + $200/month (Secret Manager) = ~$355/month  
**After Optimization**: ~$19.33-35.30/month (scale-to-zero) + $1.20/month (Secret Manager) = ~$20.53-36.50/month  
**Savings**: **88-94% cost reduction**

**Key Optimizations**:
1. ✅ Scale-to-zero (only pay when running)
2. ✅ Resource optimization (right-sized containers)
3. ✅ Container image cleanup (automated)
4. ✅ Storage lifecycle policies (auto-delete old files)
5. ✅ Efficient Cloud Scheduler usage
6. ✅ **Secret Manager version cleanup** (automatic, keeps only latest version per secret)

**After Secret Manager fix**: For further reductions (Cloud Run settings, TradingView Agent disabled, Scheduler job pruning, Artifact Registry), see your deploy runbook and cost control docs in `doc_elements/Sessions/2026/Feb12 Session/`.

### **Cost by Strategy**

**ORB Strategy Only**: ~$15.26-24.85/month  
**0DTE Strategy Only**: Included with ORB Strategy (shared infrastructure)  
**Easy Collector Only**: ~$4.07-10.45/month  
**All Strategies Combined**: ~$19.33-35.30/month

**Note**: ORB Strategy and 0DTE Strategy share the same Cloud Run service and infrastructure, so there's no additional cost for enabling 0DTE Strategy.

---

## 🚀 **Deployment Steps**

### **Step 1: Prerequisites**

1. Create Google Cloud project (or use existing)
2. Enable required APIs (see above)
3. Create service accounts with proper permissions
4. Store secrets in Secret Manager
5. Create Cloud Storage buckets

### **Step 2: Build Container Images**

Build from the **ORB Strategy project root** (directory containing `main.py`, `Dockerfile`, `cloud_run_entry.py`). The trading service image uses `cloud_run_entry.py` as the container CMD so the process listens on `PORT` immediately.

```bash
# From ORB Strategy project root (directory containing main.py, Dockerfile, cloud_run_entry.py)
# Build trading service image (includes cloud_run_entry.py entrypoint)
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/YOUR_PROJECT_ID:latest .

# Build OAuth service image (separate project/repo if applicable)
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/YOUR_PROJECT_ID-oauth:latest

# Build Easy Collector image
cd easyCollector
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/easy-collector:latest .
```

### **Step 3: Deploy Services**

**Trading Service**:
```bash
gcloud run deploy YOUR_PROJECT_ID \
    --image gcr.io/YOUR_PROJECT_ID/YOUR_PROJECT_ID:latest \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 1 \
    --min-instances 0 \
    --concurrency 80 \
    --timeout 3600 \
    --no-cpu-throttling \
    --service-account etrade-strategy-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --set-env-vars="ENVIRONMENT=production,STRATEGY_MODE=standard,ETRADE_MODE=demo,SYSTEM_MODE=full_trading,CLOUD_MODE=true,ENABLE_0DTE_STRATEGY=true,LOG_LEVEL=INFO" \
    --allow-unauthenticated
```

**OAuth Service**:
```bash
gcloud run deploy YOUR_PROJECT_ID-oauth \
    --image gcr.io/YOUR_PROJECT_ID/YOUR_PROJECT_ID-oauth:latest \
    --platform managed \
    --region us-central1 \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 1 \
    --min-instances 0 \
    --concurrency 100 \
    --timeout 300 \
    --service-account etrade-strategy-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --set-env-vars="ENVIRONMENT=production" \
    --allow-unauthenticated
```

**Easy Collector Service**:
```bash
gcloud run deploy easy-collector \
    --image gcr.io/YOUR_PROJECT_ID/easy-collector:latest \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 1 \
    --min-instances 0 \
    --concurrency 10 \
    --timeout 900 \
    --service-account etrade-strategy-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --set-env-vars="ENVIRONMENT=production" \
    --allow-unauthenticated
```

### **Step 4: Set Up Cloud Scheduler Jobs**

Use the Cloud Scheduler job commands provided in the [Cloud Scheduler Jobs](#cloud-scheduler-jobs) section above.

**Quick Setup Script**:
```bash
# For Easy Collector (15 jobs)
cd easyCollector
./SETUP_SCHEDULER.sh

# For ORB/0DTE Strategy (manual setup - see commands above)
# Or use setup scripts if available
```

### **Step 5: Verify Deployment**

```bash
# Check services are running
gcloud run services list --region=us-central1

# Check logs
gcloud run services logs read YOUR_PROJECT_ID --region=us-central1 --limit 50

# Test health endpoint
curl https://YOUR_SERVICE_URL/api/health
```

---

## 📊 **Monitoring & Logging**

### **Cloud Logging**

View logs in real-time:
```bash
# View recent logs
gcloud run services logs read YOUR_PROJECT_ID \
    --region=us-central1 \
    --limit 50

# Follow logs
gcloud run services logs tail YOUR_PROJECT_ID \
    --region=us-central1

# Filter logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=YOUR_PROJECT_ID" \
    --limit 50 \
    --format json
```

### **Cloud Monitoring**

Set up monitoring dashboards:
- Trading performance metrics
- API usage and rate limits
- System health and errors
- Cost tracking
- Request latency
- Error rates

### **Alerting**

Configure alerting policies for:
- Service downtime
- High error rates
- Cost threshold exceeded
- OAuth token expiry warnings

---

## 🔒 **Security Configuration**

### **Google-Recommended Security Best Practices**

Follow this unified security framework to reduce risk from long-lived credentials and unauthorized access. These align with Google Cloud’s advisory for credential and key management.

#### **1. Zero-code storage (credentials never in source)**

- **Never** commit API keys, OAuth tokens, or secrets to source code or version control.
- Store all credentials in **Secret Manager** and inject at runtime (e.g. Cloud Run env or startup).
- Keep a local your local credentials file (do not commit) (or similar) out of the repo (add to `.gitignore`); use it only for your own reference and for populating Secret Manager.
- This project uses Secret Manager for E*TRADE keys, Telegram tokens, and OAuth tokens; no secrets live in the codebase.

#### **2. Disable dormant keys**

- **Audit** service account keys and API keys regularly.
- **Decommission** any key with no activity in the last 30 days.
- Prefer **Workload Identity** or **no user-managed keys** where possible (e.g. Cloud Run using a service account without downloaded keys).

#### **3. Enforce API and key restrictions**

- **Never** leave an API key unrestricted.
- **Limit keys** to specific APIs (e.g. only the APIs your app needs).
- **Apply restrictions** by environment: IP addresses, HTTP referrers, or bundle IDs as appropriate for the key type.
- In Google Cloud Console: APIs & Services → Credentials → select key → set “API restrictions” and “Application restrictions”.

#### **4. Least privilege (service accounts)**

- **Never** grant broad roles (e.g. Owner, Editor) to service accounts.
- Grant only the **minimum roles** required (e.g. `roles/secretmanager.secretAccessor`, `roles/storage.objectAdmin`, `roles/logging.logWriter`, `roles/run.invoker` as needed).
- Use **IAM Recommender** in the Cloud Console (IAM → Recommender) to find and remove unused permissions for service accounts.
- Review recommendations periodically and prune access.

#### **5. Mandatory rotation and key lifecycle**

- **Service account keys (if you must use them):**  
  Enforce a maximum lifetime with the organization policy `iam.serviceAccountKeyExpiryHours` (e.g. 90 days) so keys expire and must be rotated.
- **Prefer no user-managed keys:**  
  Where possible, use **Workload Identity** or attach a service account to Cloud Run/Compute so no key file exists. To block creation of new keys, set the organization policy `iam.disableServiceAccountKeyCreation` (or the constraint `iam.managed.disableServiceAccountKeyCreation`) so new service account keys cannot be created.
- **Broker/OAuth tokens:**  
  Rotate via your OAuth flow and Secret Manager; keep only the latest version per secret to limit exposure and cost.

#### **6. Operational safeguards**

- **Essential Contacts:**  
  In Google Cloud Console → IAM & Admin → **Essential Contacts**, set contacts for security and billing so critical notifications (e.g. security incidents, abuse) reach the right people.
- **Billing anomaly and budget alerts:**  
  In Billing → Budgets & alerts, create a **budget** and enable **billing anomaly** and **budget threshold** alerts. A sudden spike in consumption is often the first sign of compromised credentials; act on these alerts promptly.

### **IAM Roles (project setup)**

Apply least privilege when creating service accounts:

- Service account with **minimal required permissions** (see § Google-Recommended above).
- Secret Manager: `roles/secretmanager.secretAccessor` for credentials only.
- Storage: `roles/storage.objectAdmin` (or narrower) for persistence buckets only.
- Logging: `roles/logging.logWriter`.
- **No** roles such as Owner, Editor, or Security Admin unless strictly required.

### **Network Security**

- Cloud Run services are publicly accessible (required for Cloud Scheduler HTTP invocations).
- OAuth and management endpoints protected by access codes or authentication.
- API endpoints can be protected by IAM or identity-aware access (optional).
- Use a VPC connector for private resources if needed.

### **Data Encryption**

- All secrets encrypted at rest in Secret Manager.
- All data encrypted in transit (HTTPS).
- Cloud Storage encrypted by default; CMEK optional for additional control.

---

## 💾 **Data Persistence**

### **GCS Persistence**

**Trade History**:
- Trades persist immediately to GCS when closed
- Trade history survives Cloud Run redeployments
- Format: JSON files in `gs://YOUR_PROJECT_ID-trades/`

**System State**:
- Trading state persisted to GCS
- Account balances persisted (Demo mode)
- Format: JSON files in `gs://YOUR_PROJECT_ID-state/`

**Performance Logs**:
- Performance metrics logged to GCS
- Format: JSON files in `gs://YOUR_PROJECT_ID-logs/`

### **Backup Strategy**

- **Automatic**: GCS provides automatic redundancy
- **Manual**: Export trade history periodically
- **Retention**: Keep all trade history (no auto-delete)

---

## ✅ **Production Readiness**

### **Pre-Deployment Checklist**

- [ ] Google Cloud project created and configured
- [ ] Required APIs enabled
- [ ] Service accounts created with **least-privilege** permissions (see Security Configuration)
- [ ] **No credentials in code**; all secrets in Secret Manager, injected at runtime
- [ ] Secrets stored in Secret Manager
- [ ] Container images built and tested
- [ ] Cloud Run services configured
- [ ] Cloud Scheduler jobs created
- [ ] Monitoring and alerting set up
- [ ] **Essential Contacts** set (IAM & Admin → Essential Contacts) for security/billing notifications
- [ ] **Billing budget and anomaly alerts** configured; notifications acted on
- [ ] Network security configured
- [ ] Backup and recovery procedures tested
- [ ] GCS persistence configured
- [ ] Cost monitoring active

### **Post-Deployment Checklist**

- [ ] All services running and healthy
- [ ] Monitoring dashboards populated
- [ ] Alerting policies active
- [ ] OAuth tokens refreshed successfully
- [ ] Trading signals generating correctly
- [ ] Telegram alerts working
- [ ] Performance metrics within expected ranges
- [ ] Cost monitoring active
- [ ] Trade persistence working
- [ ] 0DTE strategy enabled (if configured)
- [ ] Easy Collector collecting data (if configured)
- [ ] **IAM Recommender** reviewed; unused service account permissions pruned
- [ ] **Service account / API keys** audited; dormant keys (30+ days inactive) decommissioned

### **Best Practices**

#### **Security (Google-recommended)**
- **Zero-code storage:** Never commit keys; use Secret Manager and inject at runtime
- **Least privilege:** Use IAM Recommender to prune unused permissions; no Owner/Editor for service accounts
- **Key restrictions:** Restrict API keys to specific APIs and environments (IP/referrer/bundle)
- **Rotation / no keys:** Prefer no user-managed SA keys (e.g. Cloud Run attached SA); if keys exist, enforce expiry and rotate
- **Operational:** Set Essential Contacts and billing anomaly/budget alerts; act on spikes (possible compromise)
- Encrypt all sensitive data; rotate credentials regularly; audit dormant keys quarterly

#### **Performance**
- Monitor resource utilization
- Optimize based on actual usage patterns
- Implement proper caching strategies
- Use async processing where possible

#### **Reliability**
- Implement comprehensive error handling
- Set up proper monitoring and alerting
- Regular backup procedures
- Disaster recovery planning
- GCS persistence for trade history

#### **Cost Management**
- Right-size resources based on actual usage
- Monitor costs continuously
- Implement budget alerts
- Regular cost optimization reviews
- Use scale-to-zero for cost optimization

---

## 🔄 **Alternative Cloud Platforms**

### **Amazon Web Services (AWS)**

**Container Service**: AWS ECS (Fargate) or AWS Lambda  
**Storage**: S3  
**Secrets**: AWS Secrets Manager  
**Scheduler**: Amazon EventBridge (CloudWatch Events)  
**Cost**: ~$20-30/month (similar configuration)

**Deployment Steps**:
1. Create ECS cluster or Lambda functions
2. Store secrets in AWS Secrets Manager
3. Set up EventBridge rules for scheduling
4. Configure S3 buckets for data persistence
5. Set up CloudWatch for monitoring

### **Microsoft Azure**

**Container Service**: Azure Container Instances or Azure Functions  
**Storage**: Azure Blob Storage  
**Secrets**: Azure Key Vault  
**Scheduler**: Azure Logic Apps or Timer Triggers  
**Cost**: ~$20-30/month (similar configuration)

**Deployment Steps**:
1. Create Container Instances or Functions
2. Store secrets in Azure Key Vault
3. Set up Logic Apps for scheduling
4. Configure Blob Storage for data persistence
5. Set up Azure Monitor for monitoring

### **Migration Guide**

To migrate from GCP to AWS/Azure:
1. Export trade history from GCS
2. Recreate secrets in new platform's secret manager
3. Update configuration files with new service URLs
4. Redeploy container images to new platform
5. Set up equivalent scheduler jobs
6. Test all functionality
7. Import trade history to new storage

---

## 📞 **Support & Troubleshooting**

### **Common Issues**

**Service Not Waking Up**:
- Check Cloud Scheduler jobs are enabled
- Verify service URLs are correct
- Check service account permissions
- Review logs for errors

**OAuth Token Expiry**:
- Check OAuth keep-alive jobs are running
- Verify token refresh endpoint is accessible
- Review OAuth service logs
- Manually renew tokens via web dashboard

**High Costs**:
- Verify scale-to-zero is enabled (`min-instances=0`)
- Check Cloud Scheduler jobs aren't running too frequently
- Review Cloud Storage usage
- Enable cleanup policies
- **Check Secret Manager versions** (should be 1 per secret):
  ```bash
  for SECRET in $(gcloud secrets list --format='value(name)'); do
    COUNT=$(gcloud secrets versions list $SECRET --format='value(name)' | wc -l)
    if [ $COUNT -gt 1 ]; then
      echo "⚠️  $SECRET has $COUNT versions (should be 1)"
    fi
  done
  ```
  If versions > 1, run: `bash scripts/cleanup_secrets_optimized.sh`

### **Useful Commands**

```bash
# List all Cloud Scheduler jobs
gcloud scheduler jobs list --location=us-central1

# Check service status
gcloud run services describe SERVICE_NAME --region=us-central1

# View service logs
gcloud run services logs read SERVICE_NAME --region=us-central1 --limit 100

# Check costs
gcloud billing accounts list
gcloud billing projects list --billing-account=BILLING_ACCOUNT_ID
```

---

## 📝 **Revision History**

### **Latest Updates (February 9, 2026)**

**Cloud Run entrypoint (Feb 2026)**:
- ✅ **`cloud_run_entry.py`** added as container entrypoint so the process listens on `PORT` immediately
- ✅ Avoids Cloud Run startup timeout while OAuth/config/Secret Manager init runs in the background
- ✅ Full app runs via `main.py --cloud-mode` after minimal HTTP server is up

**Rev 00259 (Jan 22 - Cloud Cleanup Automation)**:
- ✅ Automated weekly cleanup of container images and Cloud Run revisions
- ✅ Cleanup endpoint added to main.py
- ✅ Cloud Scheduler job created for automated cleanup
- ✅ Retention policies: Keep last 10 images + 30 days, keep last 20 revisions
- ✅ Expected savings: 85% reduction in images, 91% reduction in revisions

**Secret Manager Cleanup (February 9, 2026)**:
- ✅ Firebase OAuth app deployed with automatic secret cleanup
- ✅ Cleanup code active in `oauth_backend.py` (Firebase Functions)
- ✅ Cleanup code active in `secret_manager_oauth.py` (Cloud Run services)
- ✅ Current cost: ~$1.20/month (20 billable versions across all projects)
- ✅ Expected cost: ~$0.78/month (13 secrets × 1 version × $0.06)
- ✅ Savings: ~$200/month (from $198/month to $1.20/month)

### **Previous Updates**

**Rev 00247 (Jan 20 - Critical Bug Fixes)**:
- ✅ ETrade API Batch Limit Fix (25 symbol limit)
- ✅ 0DTE Import Path Fix
- ✅ ORB Capture Alert Backfill Fix
- ✅ Deployment Configuration fixes

**Rev 00246 (Jan 19 - 0DTE Strategy Improvements)**:
- ✅ Priority Score Formula v1.1
- ✅ Direction-Aware Red Day Filtering
- ✅ Expanded Delta Selection (0.15-0.35)

---

**Cloud Deployment Guide - Complete and Ready for Production!** 🚀

*Last Updated: February 9, 2026*  
*Version: Rev 00259 + Cloud Run entrypoint (cloud_run_entry.py)*  
*Maintainer: Easy ORB Strategy Development Team*

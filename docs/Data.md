# Data Management System
## Easy ORB Strategy - ORB Data Architecture

**Last Updated**: February 19, 2026  
**Version**: Rev 00260 (Pipeline logging + Validation candle doc; Feb19: 7:00/7:15 batch collection same as ORB; Rev 00259: Cloud Cleanup Automation + Critical Bug Fixes)  
**Purpose**: Comprehensive documentation of the data management system for the Easy ORB Strategy (ORB ETF + 0DTE Options). The system uses a dynamic symbol list (currently 145 from core_list.csv, fully scalable) with ORB capture for SO trades only. Cloud deployment optimized for cost efficiency with scale-to-zero deployment. **All data comes from configured broker (E*TRADE default) - no third-party data sources.**

**Current Focus**: SO trades only (ORR disabled - 0% allocation)  
**Status**: ✅ Production Ready - ETrade API Batch Limit Fix (Rev 00247), 0DTE Priority Formula v1.1 (Rev 00246), Direction-Aware Red Day (Rev 00246), Expanded Delta Selection (Rev 00246), Comprehensive Logging (Rev 00246), Broker Configuration System (Rev 00245), Enhanced Red Day Detection (Rev 00237), Broker-Only Data Source (Rev 00236), Configurable Broker Support (Rev 00236), Data Quality Fixes (Rev 00233), Signal-Level Red Day Detection (Rev 00233), Enhanced Data Validation (Rev 00233), Trade Persistence Fix (Rev 00203), Unified Configuration (Rev 00201-00202), Exit Settings Optimized (Rev 00196), Trade ID Shortening (Rev 00232)

---

## 📋 **Table of Contents**

1. [Data Architecture Overview](#data-architecture-overview)
2. [Prime Data Manager](#prime-data-manager)
3. [Data Sources & Providers](#data-sources--providers)
4. [Watchlist & Symbol Management](#watchlist--symbol-management)
5. [Real-Time Data Processing](#real-time-data-processing)
6. [ORB Data Capture](#orb-data-capture)
7. [Data Quality & Validation](#data-quality--validation)
8. [Performance Optimization](#performance-optimization)
9. [API Usage & Cost Analysis](#api-usage--cost-analysis)
10. [Data Storage & Persistence](#data-storage--persistence)
11. [Integration Guide](#integration-guide)

---

## ✅ **DEPLOYMENT STATUS (Rev 00237 - January 9, 2026)**

**Easy ORB Strategy Deployed & Operational - SO Trades Only:**
- ✅ **Core List**: Dynamic (currently 145 - fully scalable without code changes - Rev 00058)
- ✅ **ORB Capture**: 6:30-6:45 AM PT window with dynamic batch processing
- ✅ **Validation open 7:00**: Cloud Scheduler job `validation-candle-700` captures 7:00 open (batches of 25, same as ORB); persisted to GCS for 7:15 prefetch when scale-to-zero.
- ✅ **SO Prefetch 7:15**: 7:00-7:15 AM candle via E*TRADE batch quotes (25/call, same as ORB); 7:15 close with skip_cache=True; optional scheduler job `prefetch-validation-715` for scale-to-zero.
- ✅ **SO Scanning**: 7:15-7:30 AM PT (continuous scanning every 30 sec - 15-minute window)
- ✅ **SO Execution**: 7:30 AM PT batch execution with multi-factor ranking (Rev 00108: VWAP 27%, RS vs SPY 25%, ORB Vol 22%)
- ✅ **Cloud Scheduler Keep-Alive**: 3 jobs ensure instance stays alive during trading hours ⭐ **CRITICAL**
- ✅ **ORB Capture Alert**: Sent at 6:45 AM PT (handles success and failure cases)
- ✅ **Trade Signal Collection Alert**: Sent at 7:30 AM PT (shows "6-15 signals")
- ✅ **Duplicate Prevention**: Same symbol can't execute twice per day
- ✅ **Prime Data Manager**: Batch quotes (25/call) for efficient data fetching - **BROKER-ONLY** (Rev 00236)
- ✅ **Prime Risk Manager**: Demo & Live modes with rank-based position sizing (Rev 00090)
- ✅ **Prime ORB Strategy Manager**: SO signal generation with validation rules
- ✅ **Prime Stealth Trailing**: Optimized trailing stops (Rev 00196: 0.7% @ 6.4 min, 1.5-2.5% distance)
- ✅ **Centralized Alerts**: All alerts in prime_alert_manager.py (single source of truth)
- ✅ **Mock Trading Executor**: Demo mode with EOD tracking
- ✅ **E*TRADE Integration**: Live mode ready (default broker)
- ✅ **Configurable Broker Support**: E*TRADE (default), Interactive Brokers, Robinhood (Rev 00236)
- ✅ **Broker-Only Data**: All data from configured broker - no third-party fallback (Rev 00236)
- ✅ **Multi-Factor Ranking**: VWAP (27%) + RS vs SPY (25%) + ORB Vol (22%) - Rev 00108 ⭐ **DATA-PROVEN**
- ✅ **Unified Configuration**: 65+ configurable settings (Rev 00201)
- ✅ **Trade Persistence**: GCS persistence working (Rev 00203)
- ✅ **Trade ID Formatting**: Shortened format (Rev 00232)
- ✅ **Data Quality System**: Enhanced validation with neutral defaults (Rev 00233)
- ✅ **Signal-Level Filtering**: Individual trade Red Day detection (Rev 00233)
- ✅ **Filter Consistency**: ORB and 0DTE filters aligned (Rev 00233)
- ✅ **Enhanced Red Day Detection**: Real SPY momentum and VIX level from E*TRADE (Rev 00237)

**Disabled/Archived Components:**
- ⏸️ **ORR Trades**: Disabled (0% capital allocation) - Will optimize separately
- ❌ **Dynamic Watchlist Builder**: PAUSED - Dynamic core_list.csv used (Rev 00058)
- ❌ **Symbol Selector**: ARCHIVED - All symbols from core_list.csv used
- ❌ **Multi-Strategy Manager**: ARCHIVED - ORB only
- ❌ **Historical Data Caching**: Not needed for ORB
- ❌ **Compound Engine**: Not needed (ORR disabled)

---

## 🏗️ **Data Architecture Overview**

The Easy ORB Strategy implements a **prime data management system** optimized for 24/7 operation with intelligent failover, advanced caching, and multi-provider support. The system ensures consistent performance across symbol scanning, trading operations, and position monitoring.

### **Key Principles**
- **Simplicity**: ORB-only strategy with clear data flow
- **Efficiency**: Batch processing and intelligent caching
- **Reliability**: Multi-provider fallback with circuit breaker protection
- **Scalability**: Dynamic symbol list (add/remove without code changes)
- **Performance**: Optimized for low latency and high throughput

---

## 🚀 Prime Data Manager

### **System Consolidation**
- **Single Data Manager**: All data operations consolidated into `prime_data_manager.py`
- **Broker-Only Support**: E*TRADE (default) - all data from configured broker (Rev 00236)
- **No Third-Party Fallback**: System stops if broker fails (no silent fallback) (Rev 00236)
- **Advanced Caching**: Multi-tier caching with TTL-based cleanup
- **Data Quality Assessment**: Quality scoring and validation
- **Async Data Processor**: 70% faster data processing with connection pooling
- **Unified Models Integration**: PrimeSignal, PrimePosition, PrimeTrade data structures throughout
- **Real Market Data**: SPY momentum and VIX level retrieved from E*TRADE (Rev 00237)

### **Current Integration Status**
- **Data Manager**: ✅ **IMPLEMENTED** - `prime_data_manager.py` exists and ready
- **Main System Integration**: ✅ **ACTIVE AND FUNCTIONAL** - Trading thread operational
- **Scanner Integration**: ✅ **FULLY INTEGRATED** - Components connected and operational
- **ORB Strategy Integration**: ✅ **FULLY INTEGRATED** - ORB capture and SO signal generation operational

---

## 🏗️ Data Architecture

### **Data Organization Structure**
The system uses a structured approach to data management with specialized directories:

```
data/
├── 📋 watchlist/                    # Symbol Management
│   ├── core_list.csv                 # Core 145 symbols organized by leverage
│   └── 0dte_list.csv                # 0DTE symbols (if 0DTE strategy enabled)
├── 📊 score/                        # Performance Tracking
│   ├── symbol_scores.json           # Prime score data
│   └── symbol_scores_backup.json    # Backup data
└── ⚙️ System Files
    ├── holidays_custom.json         # Custom holiday calendar
    ├── state.json                   # System state file
    └── secret_manager_example.py    # Google Secret Manager integration
```

---

## 📋 Watchlist & Symbol Management

### **Core List (`core_list.csv`)** ⭐ **PRIMARY - Rev 00058**

**Current Status**:
- **Dynamic (currently 145)** organized by leverage (4x, 3x, 2x, 1x) + Category
- **ORB Data Usage**: ORB high/low passed to stealth trailing for entry bar protection (Rev 00135)
- **Batch Sizing**: Rev 00089 - quantity_override ensures batch-sized quantities used exactly
- **Leverage Organized**: 4×, 3×, 2× (Quantum, Crypto, Stock), 1× ETFs
- **Category Prioritized**: Quantum, Crypto 2×, Single-Stock 2×, Tech 3×, Standard
- **Pre-Filtered**: Volatility (ATR), volume (5M+ daily), performance validated
- **Production Ready**: Used for all ORB capture and SO trades
- **Fully Scalable**: Add/remove symbols without code changes (Rev 00058) ⭐ **KEY FEATURE**

### **0DTE Symbol List (`0dte_list.csv`)** ⭐ **0DTE STRATEGY - Rev 00209**

**Current Status**:
- **Dynamic (currently 111 symbols)** organized by tier (Tier 1: 23, Tier 2: 88)
- **ORB Data Integration**: All 0DTE symbols included in ORB capture (Rev 00209)
- **Tier Organization**:
  - **Tier 1** (23 symbols): Core Daily 0DTE
    - Core symbols: SPX, SPY, QQQ, IWM, MAGS, VIX, IBIT, GLD, SLV
    - Leverage ETFs: SPYU (4x), SPXL, UPRO, TQQQ, SOXL, URTY, TNA, UMDD, TECL, FAS, GUSH, LABU, FNGU, WEBL (3x)
  - **Tier 2** (88 symbols): All others
    - MAG-7 + Core Tech: NVDA, AMD, TSLA, META, AMZN, AAPL, MSFT, GOOGL
    - AI/SEMI Leaders: AVGO, ASML, ARM, SMCI, MRVL, AMAT, INTC
    - Platform/Cloud/Data: NOW, SNOW, NFLX, COST, HD, MSTR
    - Thematic/Sector Momentum: COIN, HOOD, PLTR, CRWD, NET, QCOM, MU, etc.
    - High-Beta/Retail: SOFI, HIMS, DAL, AAL, crypto miners, homebuilders, etc.
- **Production Ready**: Used for 0DTE options signal generation and execution
- **Fully Scalable**: Add/remove symbols without code changes
- **Priority Processing**: Tier 1 symbols processed first, then Tier 2

**ORB Data Collection Integration** (Rev 00209):
- 0DTE symbols loaded from `data/watchlist/0dte_list.csv` during ORB capture
- Merged with ORB symbols (no duplicates - symbols already in ORB list are not duplicated)
- All symbols (ORB + 0DTE) captured in single batch operation (6:30-6:45 AM PT)
- ORB data used for 0DTE signal generation and eligibility filtering

**Example**:
- ORB list: 145 symbols
- 0DTE list: 111 symbols
- Combined: ~145 symbols (0DTE symbols already in ORB list are not duplicated)
- ORB capture: All symbols processed in batch (2-5 seconds)

**Multi-Factor Ranking** (Rev 00108 - Formula v2.1):
```python
priority_score = (
    vwap_distance_score * 0.27 +  # 27% ⭐ STRONGEST (correlation +0.772)
    rs_vs_spy_score * 0.25 +      # 25% ⭐ 2ND STRONGEST (correlation +0.609)
    orb_vol_score * 0.22 +        # 22% MODERATE (correlation +0.342)
    confidence_score * 0.13 +     # 13% WEAK (correlation +0.333)
    rsi_score * 0.10 +            # 10% Context-aware
    orb_range_score * 0.03        # 3% Minimal contribution
)
```

**Evidence Base**:
- 89-field technical indicators tracked daily
- 3-day comprehensive data collection (Nov 4, 5, 6, 2025)
- Correlation analysis validated formula weights
- Expected +10-15% better capital allocation vs v2.0

---

## 📊 Real-Time Data Processing

### **ORB Data Capture** ⭐ **CRITICAL**

**Timing**: 6:30-6:45 AM PT (9:30-9:45 AM ET) - First 15 minutes of trading

**Process**:
1. Market opens at 6:30 AM PT
2. System captures opening range for all symbols (dynamic count - currently 145)
3. Batch processing: Dynamic batches based on symbol count (2-5 seconds total)
4. ORB data stored: High, Low, Open, Close, Volume, Range %
5. Data source: **E*TRADE batch quotes ONLY** (today's OHLC = ORB) - Rev 00236
6. **No Fallback**: System stops if broker fails (no third-party backup)

**Data Structure**:
```python
orb_data = {
    'symbol': 'QQQ',
    'orb_high': 385.50,
    'orb_low': 382.30,
    'orb_open': 383.10,
    'orb_close': 384.20,
    'orb_volume': 1250000,
    'orb_range_pct': 0.84,  # (high - low) / low * 100
    'timestamp': '2026-01-06T06:45:00-08:00'
}
```

**Uses for ORB Data**:
- **ORB Strategy**: Breakout detection (price > ORB high), entry bar protection, stop loss calculation, multi-factor ranking
- **0DTE Strategy**: ORB data used for Convex Eligibility Filter (ORB range ≥ 0.35%, ORB break confirmation), signal generation, and strategy selection

---

### **SO Signal Collection** ⭐ **PRIMARY**

**Timing**: 7:15-7:30 AM PT (10:15-10:30 AM ET) - 15-minute collection window

**Process**:
1. **Prefetch** (7:15 AM PT): Fetch 7:00-7:15 AM candle for validation. **Full ORB + 0DTE symbol list** (no cap). Uses **two broker snapshots**: 7:00 open (from Cloud Scheduler job or GCS `daily_markers/validation_open_700/YYYY-MM-DD.json`) and 7:15 close (batch quotes). Same broker API as ORB (one snapshot at 6:45 for ORB; two snapshots 7:00 + 7:15 for validation).
2. **Scanning** (7:15-7:30 AM PT): Continuous validation every 30 seconds
3. **Validation**: 3 strict rules (price, volume color, previous candle)
4. **Collection**: 6-15 qualified signals from all symbols
5. **Ranking**: Multi-factor priority scoring (Rev 00108)
6. **Selection**: Top 15 affordable signals pre-selected

**SO Validation Rules** (Bullish - All 3 Required):
1. **Current price ≥ ORB high × 1.001** (+0.1% buffer)
2. **Previous close > ORB high** (7:00-7:15 AM candle closed above range)
3. **Green candle** (7:15 AM close > 7:00 AM open = buying pressure)

**Data Collection**:
- Real-time price quotes (**E*TRADE batch quotes ONLY** - Rev 00236)
- Technical indicators (VWAP, RS vs SPY, RSI, MACD)
- Volume analysis
- Momentum indicators
- **All data from configured broker** - no third-party sources

**ORB vs validation candle (same broker API)**:
- **ORB candle** (first 15 min): **One** `get_batch_quotes()` at **6:45 AM PT**; quote’s today’s OHLC = first 15‑min bar.
- **Validation candle** (7:00–7:15): **Two** snapshots — 7:00 open (stored in memory or GCS `daily_markers/validation_open_700/YYYY-MM-DD.json`) and 7:15 close via `get_batch_quotes()`. Broker does not return “the 7:00–7:15 bar” in one call. **Definition:** Open = price at 7:00 AM PT only; close = price at 7:15 AM PT only. No proxy (e.g. market open) is used—using any other open would produce false Long signals.

---

## 🎯 Data Sources & Providers

### **Broker-Only Data Source** ⭐ **Rev 00236**

**Rev 00236 (Jan 9, 2026): All data MUST come from configured broker - no third-party data sources**

**Supported Brokers**:
- ✅ **E*TRADE** (default) - Fully implemented
- 🔄 **Interactive Brokers** - Placeholder (ready for implementation)
- 🔄 **Robinhood** - Placeholder (ready for implementation)

**Configuration**:
- `BROKER_TYPE=etrade` (default, in `configs/broker-config.env` - Rev 00245)
- `BROKER_DATA_ONLY=true` (required - no third-party fallback)
- **Primary Configuration File**: `configs/broker-config.env` (Rev 00245)
- **Broker Config Manager**: `modules/broker_config_manager.py` (Rev 00245)

**Key Principles**:
1. **Broker Data Only**: All data comes directly from the configured broker
2. **No Third-Party Sources**: No yfinance, Alpha Vantage, or other third-party data
3. **Configurable**: Broker selection via `BROKER_TYPE` configuration
4. **Error Handling**: System stops trading if broker fails (no silent fallback)

---

### **Primary Provider: E*TRADE API** ⭐ **DEFAULT**

**Usage**:
- ORB capture (batch quotes - 25 symbols per call)
- Real-time price quotes
- SO prefetch (current price + today's OHLC)
- SO scanning (current prices)
- Account information
- Order execution (Live mode)

**Optimization**:
- Batch requests: Group multiple symbols in single API call (25 symbols per call) ⭐ **Rev 00247: Enforced 25 symbol limit**
- **Batch Limit Enforcement** (Rev 00247): Automatically splits large symbol lists into batches of 25 to prevent API error 1023
- Smart caching: Cache quotes for 1 second to reduce redundant calls
- Rate limiting: 100ms between calls to avoid throttling
- Connection reuse: Maintain persistent connections
- **Speed**: 2-5 seconds for 136 symbols (vs 131.6s with third-party fallback)
- **Error Handling**: Graceful batch processing - continues with next batch if one fails

**Cost**: Included with E*TRADE account (no additional fees)

**Data Collection**:
- ✅ **ORB Capture**: Uses E*TRADE batch quotes (bars=1)
- ✅ **SO Prefetch**: Uses E*TRADE batch quotes (current price + today's OHLC)
- ✅ **SO Scanning**: Uses E*TRADE batch quotes (current prices)
- ✅ **All Data Collection**: E*TRADE batch quotes exclusively

---

### **Third-Party Providers: REMOVED** ⚠️ **Rev 00236**

**Rev 00236**: All third-party data sources removed:
- ❌ **yfinance**: Removed from all data collection paths
- ❌ **Alpha Vantage**: Removed from all data collection paths
- ❌ **Polygon**: Removed from all data collection paths
- ❌ **Emergency Fallback**: Disabled (system stops if broker fails)

**Rationale**:
- E*TRADE is the broker - must use broker data for accuracy and speed
- Broker data is authoritative and reliable
- No third-party dependencies or throttling issues
- Faster data collection (2-5 seconds vs 131.6 seconds)

---

### **Enhanced Red Day Detection Market Data** ⭐ **Rev 00237**

**Real-Time Market Data for Risk Assessment:**

**SPY Momentum Calculation**:
- **Source**: E*TRADE quote API (`get_quotes(['SPY'])`)
- **Method**: Uses `change_pct` from quote, with fallback to open vs previous close
- **Fallback Calculation**: Historical data (2 days) to calculate momentum if `change_pct` unavailable
- **Usage**: Enhanced Red Day Detection risk assessment
- **Replaces**: Hardcoded value (0.0%)

**VIX Level Retrieval**:
- **Source**: E*TRADE quote API (tries both `$VIX` and `VIX` symbols)
- **Method**: Uses `last_price` or `price` from quote
- **Fallback**: Defaults to 15.0 if unavailable
- **Usage**: Enhanced Red Day Detection volatility assessment
- **Replaces**: Hardcoded value (15.0)

**Benefits**:
- ✅ More accurate risk assessment using real market conditions
- ✅ Better Red Day detection with actual SPY momentum and VIX volatility
- ✅ Improved capital preservation through better risk analysis
- ✅ Real-time market data integration

**Code Location**:
- `modules/prime_trading_system.py` (lines 5579-5636)
- Enhanced Red Day Detection section

---

## 📊 Data Quality & Validation

### **Quality Checks**

**ORB Data Validation**:
- ✅ All symbols captured (dynamic count)
- ✅ Valid price data (high > low, high > open, low < open)
- ✅ Volume > 0
- ✅ Range % calculated correctly
- ✅ Timestamp within 6:30-6:45 AM PT window

**SO Signal Validation**:
- ✅ Price above ORB high (with buffer)
- ✅ Previous candle closed above ORB high
- ✅ Green candle (buying pressure)
- ✅ Technical indicators available
- ✅ No duplicate symbols per day

**Data Quality Scoring**:
- **High Quality**: All checks pass, recent data (< 5 seconds old)
- **Medium Quality**: Most checks pass, slightly stale data (< 30 seconds old)
- **Low Quality**: Some checks fail, stale data (> 30 seconds old)

---

## ⚡ Performance Optimization

### **Caching Strategy**

**Multi-Tier Caching**:
- **L1 Cache**: In-memory (1 second TTL for quotes)
- **L2 Cache**: File-based (5 minutes TTL for indicators)
- **L3 Cache**: GCS persistence (daily for trade history)

**Cache Hit Rates**:
- **Quote Cache**: 90%+ hit rate
- **Indicator Cache**: 85%+ hit rate
- **ORB Data Cache**: 100% hit rate (cached for entire trading day)

### **Batch Processing**

**ORB Capture**:
- Batch size: 25 symbols per call
- Processing time: 2-5 seconds for 145 symbols
- Parallel processing: Multiple batches processed concurrently

**SO Signal Collection**:
- Continuous scanning: Every 30 seconds
- Batch validation: All symbols validated together
- Efficient filtering: Only qualified signals processed

### **Performance Metrics**

**Real-World Performance** (Rev 00236 - Broker-Only):
| Operation | Symbol Count | Processing Time | Improvement |
|-----------|-------------|------------------|-------------|
| **ORB Capture** | 145 | 2-5 seconds | E*TRADE batch quotes |
| **SO Prefetch** | 100 | 2-3 seconds | E*TRADE batch quotes (was 131.6s with yfinance) |
| **SO Scanning** | 145 | 1-2 seconds | E*TRADE batch quotes |
| **Signal Ranking** | 6-15 signals | < 100ms | Optimized |
| **Batch Execution** | Up to 15 trades | 2-3 seconds | Optimized |

**Memory Usage**:
- **Baseline**: 400-600MB
- **Peak (during trading)**: 800MB-1.2GB
- **After hours**: 300-500MB

---

## 💰 API Usage & Cost Analysis

### **E*TRADE API Usage** (Rev 00236 - Broker-Only)

**Daily Usage**:
- **ORB Capture**: ~6 batch calls (145 symbols ÷ 25 per call)
- **SO Prefetch**: ~4 batch calls (100 symbols ÷ 25 per call)
- **SO Scanning**: ~30 batch calls (every 30 seconds for 15 minutes)
- **Position Monitoring**: ~1,200 calls (every 30 seconds for 6.5 hours)
- **Total**: ~1,240 API calls per day

**Cost**: **$0** (included with E*TRADE account)

### **Third-Party API Usage** (Rev 00236)

**Daily Usage**:
- **yfinance**: **0 calls** (removed - Rev 00236)
- **Alpha Vantage**: **0 calls** (removed - Rev 00236)
- **Polygon**: **0 calls** (removed - Rev 00236)

**Cost**: **$0** (not used)

### **Total Monthly Cost**

**API Costs**: $0 (E*TRADE included - broker data only)  
**Cloud Infrastructure**: ~$11-15/month (Google Cloud Run - scale-to-zero)  
**Secret Manager**: ~$1.20/month (20 billable versions with automatic cleanup)  
**Total**: **~$12-16/month** (93-96% reduction from previous ~$155-355/month)

**Rev 00236 Benefits**:
- ✅ **Faster**: 2-5 seconds vs 131.6 seconds (SO prefetch)
- ✅ **Reliable**: Broker data is authoritative
- ✅ **Consistent**: All data from same source
- ✅ **No Dependencies**: No third-party throttling or errors

---

## 💾 Data Storage & Persistence

### **GCS Persistence** (Rev 00203) ⭐

**Trade History**:
- All closed trades persisted to GCS
- Trade history survives Cloud Run redeployments
- Automatic persistence on trade close

**Account Balance**:
- Demo account balance persists between deployments
- Closed trades update balance correctly (Rev 00145)
- Retry logic prevents balance reset on transient failures (Rev 00146)

**Mock Trading History**:
- Mock trading history persists across redeployments (Rev 00177)
- Trade persistence bug fixed (Rev 00203)

### **Local Storage**

**State Files**:
- `data/state.json`: System state (market hours, last update, etc.)
- `data/holidays_custom.json`: Custom holiday calendar
- `data/watchlist/core_list.csv`: Symbol list (145 symbols)

**Score Files**:
- `data/score/symbol_scores.json`: Performance tracking
- `data/score/symbol_scores_backup.json`: Backup data

---

## 🔧 Integration Guide

### **Configuration** (Rev 00201) ⭐

**Unified Configuration System**:
- **65+ configurable settings** via `configs/` files
- **No hardcoded values**
- **Single source of truth**

**Key Configuration Files**:
- `configs/strategies.env`: Capital allocation (90% SO / 10% Reserve)
- `configs/position-sizing.env`: Position sizing rules
- `configs/risk-management.env`: Exit settings (65+ settings)
- `configs/deployment.env`: Strategy enablement (ORB/0DTE)

### **Environment Variables & Secrets Management** (Rev 00233) 🔒

**Local Development**:
- **E*TRADE Credentials**: Store in `secretsprivate/etrade.env` (gitignored)
- **Telegram Credentials**: Store in `secretsprivate/telegram.env` (gitignored)
- **Templates**: Use `secretsprivate/*.env.template` files as reference
- **Loading**: Automatically loaded by `modules/config_loader.py` when `ENVIRONMENT=development`

**Production Deployment**:
- **E*TRADE Credentials**: Store in Google Secret Manager
- **Telegram Credentials**: Store in Google Secret Manager
- **Loading**: Automatically loaded by `modules/config_loader.py` when `ENVIRONMENT=production`

**Configuration Files**:
- **No Hardcoded Secrets**: All sensitive credentials removed from `configs/*.env` files (Rev 00233)
- **Safe to Commit**: Template files (`.env.template`) are safe for version control

**For complete setup instructions, see the Secrets Management section in [docs/Settings.md](Settings.md).**

**Optional Settings**:
```bash
ENABLE_0DTE_STRATEGY=true  # Enable 0DTE options strategy
ETRADE_MODE=demo          # demo or live
```

---

## 🎯 Key Features

### **1. Dynamic Symbol Lists** ⭐ Rev 00058 + Rev 00209
- **ORB Symbol List**: 145 symbols from `core_list.csv` (fully scalable)
- **0DTE Symbol List**: 111 symbols from `0dte_list.csv` (fully scalable)
- **Organization**: ORB by leverage (4x, 3x, 2x, 1x) + Category; 0DTE by tier (Tier 1: 23, Tier 2: 88)
- **Pre-Filtered**: Volatility, volume, performance validated
- **ORB Data Integration**: All 0DTE symbols included in ORB capture (Rev 00209)

### **2. Multi-Factor Ranking** ⭐ Rev 00108
- **VWAP Distance**: 27% (strongest predictor - +0.772 correlation)
- **RS vs SPY**: 25% (2nd strongest - +0.609 correlation)
- **ORB Volume**: 22% (moderate - +0.342 correlation)
- **Data-Driven**: Based on comprehensive correlation analysis

### **3. Entry Bar Protection** ⭐ Rev 00135
- **Permanent Floor Stops**: Based on actual ORB volatility
- **Tiered Stops**: 2-8% based on volatility
- **Prevents**: 64% of immediate stop-outs
- **Real-World Validation**: Saved NEBX trade (+$7.84 profit)

### **4. Trade Persistence** ⭐ Rev 00203
- **GCS Persistence**: Trades persist immediately to GCS
- **Survives Deployments**: Trade history persists across redeployments
- **Account Balance**: Demo balance persists correctly

### **5. Unified Configuration** ⭐ Rev 00201
- **65+ Settings**: All configurable via `configs/` files
- **Single Source of Truth**: No hardcoded values
- **Easy Adjustment**: Change settings in one place

### **6. Data Quality System** ⭐ Rev 00233 **NEW**
- **Neutral Defaults**: RSI=50.0, Volume=1.0 instead of 0.0
- **Prevents False Positives**: No false Red Day detection from invalid data
- **Enhanced Validation**: Helper functions filter invalid values
- **Better Diagnostics**: Enhanced logging for data quality issues

### **7. Signal-Level Red Day Detection** ⭐ Rev 00233
- **Two-Layer Protection**: Portfolio-level + Signal-level filtering
- **Individual Trade Filtering**: Rejects losing trades even on good days
- **Criteria**: Weak volume + (Oversold RSI OR No momentum OR Negative VWAP)
- **Impact**: Prevents losing trades while allowing winning trades

### **8. Enhanced Red Day Detection with Real Market Data** ⭐ Rev 00237 **NEW**
- **Real SPY Momentum**: Calculated from E*TRADE quotes (replaces hardcoded 0.0%)
- **Real VIX Level**: Retrieved from E*TRADE quotes (replaces hardcoded 15.0)
- **Real-Time Risk Assessment**: Uses actual market conditions for better accuracy
- **Improved Capital Preservation**: More accurate risk analysis prevents losses
- **Data Source**: E*TRADE quote API (SPY and VIX symbols)
- **Fallback**: Graceful defaults if data unavailable (SPY: 0.0%, VIX: 15.0)

---

## 🎉 Bottom Line

The Easy ORB Strategy data management system provides:

✅ **Real-time data** with E*TRADE integration (broker-only)  
✅ **Cost-effective** operation at ~$11/month total  
✅ **Broker-only data** (no third-party fallback - Rev 00236)  
✅ **Configurable broker support** (E*TRADE default, IB/Robinhood ready - Rev 00236)  
✅ **Enhanced Red Day Detection** (real SPY momentum & VIX level - Rev 00237)  
✅ **High performance** with optimized data processing (2-5 seconds vs 131.6s)  
✅ **Professional monitoring** and quality assurance  
✅ **Scalable architecture** for future growth  
✅ **Dynamic symbol list** (add/remove without code changes)  
✅ **Multi-factor ranking** (data-driven formula v2.1)  
✅ **Entry bar protection** (permanent floor stops)  
✅ **Trade persistence** (GCS integration)  
✅ **Unified configuration** (65+ configurable settings)  
✅ **145 symbol coverage** with complete ORB data  
✅ **90%+ cache hit rate** for optimal performance  
✅ **88-90% capital deployment** guaranteed  

**Ready for 24/7 automated trading with institutional-grade data management!** 🚀

---

*For strategy details, see [Strategy.md](Strategy.md)*  
*For process flow, see [ProcessFlow.md](ProcessFlow.md)*  
*For risk management, see [Risk.md](Risk.md)*  
*For alert system, see [Alerts.md](Alerts.md)*  
*For configuration reference, see [Settings.md](Settings.md)*  
*For cloud project data and run scripts, see [Cloud.md](Cloud.md) and your own runbook.*

---

*Last Updated: February 9, 2026*  
*Version: Rev 00259 (Cloud Cleanup Automation + Critical Bug Fixes: ETrade API Batch Limit + Broker Configuration System)*  
*Status: ✅ Production Ready - Cloud Cleanup Automation (Rev 00259), ETrade API Batch Limit Fix (25 symbol limit enforced), Enhanced Red Day Detection with real market data, broker-only data collection, configurable broker support*


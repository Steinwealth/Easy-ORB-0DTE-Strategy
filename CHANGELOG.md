# Changelog

Revision history for the Easy ORB 0DTE Strategy. Dates and revision IDs preserved from production release notes.

---

## Latest Updates (Rev 00259)

### Cloud Cleanup Automation (Rev 00259 - January 22, 2026)

1. **Automated Image & Revision Cleanup** (Rev 00259)
   - **Cleanup Endpoint**: `POST /api/cleanup/images` added to main.py
   - **Cloud Scheduler Job**: Weekly cleanup every Sunday at 2:00 AM PT
   - **Retention Policy**: Keep last 10 images + 30 days, keep last 20 revisions per service
   - **Expected Savings**: 85% reduction in images (127 → ~20), 91% reduction in revisions (224 → ~20)
   - **Cost Impact**: ~$0.31/month storage savings
   - **Automation**: Fully automated, no manual intervention required

2. **Cleanup Scripts Created** (Rev 00259)
   - `scripts/cleanup_old_images.sh`: GCR image cleanup
   - `scripts/cleanup_old_revisions.sh`: Cloud Run revision cleanup
   - `scripts/setup_cleanup_scheduler.sh`: Automated scheduler setup
   - **Cleanup docs:** docs/Cloud.md § Cloud Cleanup

### 0DTE Strategy Improvements (Rev 00246 - January 19, 2026)

1. **Priority Score Formula v1.1** (Rev 00246)
   - **Breakout**: 35% (↑ from 30%)
   - **Range**: 30% (↑ from 25%)
   - **Volume**: 20% (same)
   - **Eligibility**: 15% (same)
   - **RS vs SPY**: REMOVED (not relevant for 0DTE options)
   - **Momentum**: REMOVED (redundant with breakout score)
   - Focused on options-relevant factors for better ranking

2. **Red Day Check - Direction-Aware Filtering** (Rev 00246)
   - **LONG (CALL) trades**: Rejected on Red Days
   - **SHORT (PUT) trades**: Allowed and encouraged on Red Days
   - SHORT signals get bonus on Red Days (perfect for PUT trades)
   - Better utilization of declining market conditions

3. **Delta Selection Expanded** (Rev 00246)
   - **Old Range**: 0.15-0.25 (too restrictive)
   - **New Range**: 0.15-0.35
   - SPX/QQQ/SPY: Delta up to 0.30 (↑ from 0.25) for high volatility
   - Other symbols: Delta up to 0.35 (NEW) for high volatility opportunities
   - More trade opportunities with expanded range

4. **Comprehensive Logging** (Rev 00246)
   - Added logging throughout entire 0DTE flow (ORB Capture → Execution)
   - Per-signal filtering with detailed breakdown
   - Hard Gate validation logging (OI, spread, volume)
   - Priority score breakdown logging
   - Easy troubleshooting when 0 signals execute

### 0DTE Strategy Optimizations (Rev 00238 - January 9, 2026)

1. **Long Call Optimization** (Rev 00238)
   - Premium minimum: Lowered from $0.20 to $0.15 (allows $0.19 entries like successful trades)
   - Target delta: Adjusted from 0.40 to 0.15 (OTM for maximum gamma explosion)
   - **Validation**: Strategy aligns with high-return trades (QQQ +300%, IWM +460%)
   - **Example**: QQQ 628c @ $0.19 → $0.97 (+410% if QQQ moves +0.86%)

2. **Real-Time Options Price Tracking** (Rev 00238)
   - Options prices updated every 30 seconds from E*TRADE API
   - Position values updated with real options prices (not underlying price movement)
   - Exit decisions based on actual options P&L (captures 300-400%+ moves)
   - **Implementation**: `ETradeOptionsAPI.get_option_quote()` + `OptionsTradingExecutor.update_positions_with_real_prices()`
   - **Impact**: Accurate profit target triggers and exit decisions

3. **Symbol List Expansion** (Rev 00209)
   - Expanded from 3 symbols (SPX, QQQ, SPY) to 111 symbols
   - Tier organization (Tier 1: 23, Tier 2: 88)
   - All 0DTE symbols included in ORB capture (6:30-6:45 AM PT)
   - ORB data used for 0DTE signal generation and eligibility filtering

4. **Comprehensive Documentation** (Rev 00238)
   - easy0DTE/docs/Strategy.md: Daily trading workflow and entry rules
   - easy0DTE/docs/Data.md: Broker data connections and symbol list management
   - easy0DTE/docs/Alerts.md: Complete 0DTE alert system documentation
   - Main ORB docs updated with 0DTE integration details

### Enhanced Red Day Detection with Real Market Data (Rev 00237 - January 9, 2026)

1. **Real SPY Momentum Calculation** (Rev 00237)
   - Fetches SPY quote from E*TRADE for real-time momentum
   - Calculates from `change_pct` or open vs previous close
   - Replaces hardcoded value (0.0%) with actual market data
   - More accurate risk assessment

2. **Real VIX Level Retrieval** (Rev 00237)
   - Fetches VIX quote from E*TRADE (tries both `$VIX` and `VIX`)
   - Replaces hardcoded value (15.0) with actual volatility level
   - Better Red Day detection with real market volatility
   - Graceful fallback if unavailable

### Broker-Only Data Source & Configurable Broker Support (Rev 00236)

3. **Broker-Only Data Collection** (Rev 00236)
   - All data comes from configured broker (E*TRADE default)
   - No third-party data sources (yfinance, Alpha Vantage removed)
   - Faster data collection (2-5 seconds vs 131.6 seconds)
   - More reliable (broker data is authoritative)

4. **Configurable Broker Support** (Rev 00236)
   - E*TRADE (default) - Fully implemented
   - Interactive Brokers - Placeholder (ready for implementation)
   - Robinhood - Placeholder (ready for implementation)
   - Configuration via `BROKER_TYPE` setting

5. **Configuration Organization** (Rev 00236)
   - Removed deprecated `DATA_PRIORITY` settings
   - Fixed emergency fallback conflicts
   - Organized GCP settings (deployment-specific)
   - Clear file responsibilities

### Previous Updates (Rev 00233)

4. **Data Quality Fixes** (Rev 00233)
   - Fixed RSI and Volume defaulting to 0.0
   - Use neutral defaults (RSI=50.0, Volume=1.0)
   - Prevents false Red Day detection

5. **Fail-Safe Mode Consistency** (Rev 00233)
   - Fixed signals marked Red Day but ORB bypassed filter
   - Clear `is_red_day` flag when fail-safe activates
   - ORB and 0DTE filters now consistent

6. **Enhanced Data Validation** (Rev 00233)
   - Added helper functions with neutral defaults
   - More accurate Red Day detection

7. **Signal-Level Red Day Detection** (Rev 00233)
   - Individual signal filtering for Red Day characteristics
   - Two-layer protection (portfolio + signal level)

8. **Enhanced Convex Filter Logging** (Rev 00233)
   - Detailed rejection reasons for top 5 signals
   - Better diagnostics and troubleshooting

9. **Trade ID Shortening** (Rev 00232)
   - ORB: `MOCK_SYMBOL_YYMMDD_microseconds`
   - 0DTE: `DEMO_SYMBOL_YYMMDD_STRIKE_TYPE_microseconds`
   - Better alert readability

---

*Revision history moved from production README; dates and revision IDs preserved.*

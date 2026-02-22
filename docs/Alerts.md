# Alert System User Guide
## Easy ORB Strategy - Complete Alert System Documentation

**Last Updated**: February 19, 2026  
**Version**: Rev 00260 + Feb19 (0-signals diagnostic, validation candle logging; EOD single source)  
**Purpose**: Complete user guide for the Easy ORB Strategy alert system, covering ORB Strategy, 0DTE Strategy, and Easy Collector alerts. Includes Telegram setup instructions and all alert types.

**Note**: Store Telegram bot tokens and chat IDs in your environment or secret manager (never in docs). Use your own project URLs and cloud configuration for deployment.

---

## 📋 **Table of Contents**

1. [Alert System Overview](#alert-system-overview)
2. [Telegram Setup Guide](#telegram-setup-guide)
3. [ORB Strategy Alerts](#orb-strategy-alerts)
4. [0DTE Strategy Alerts](#0dte-strategy-alerts)
5. [OAuth System Alerts](#oauth-system-alerts)
6. [Alert Configuration](#alert-configuration)
7. [Daily Alert Flow](#daily-alert-flow)
8. [Alert Formatting](#alert-formatting)
9. [Troubleshooting](#troubleshooting)

---

## 🚨 **Alert System Overview**

The Easy ORB Strategy implements a comprehensive alert system that provides real-time notifications for all critical system events. The system operates across three main components:

1. **ORB Strategy**: Trading signals for US market stocks and leveraged ETFs
2. **0DTE Strategy**: Options trading signals for 0DTE options
3. **OAuth System**: Token management and renewal notifications

### **Core Features**

- **Multi-Channel Delivery**: Telegram notifications with HTML formatting
- **Rich Formatting**: Emoji-enhanced messages with structured data
- **Dual Timezone Support**: All alerts display both PT and ET times with AM/PM format
- **Intelligent Throttling**: Prevents alert spam while maintaining critical notifications
- **Source Tracking**: Clear identification of alert source
- **Trade ID Formatting**: Shortened trade IDs for cleaner format
- **Enhanced Execution Alerts**: Bold formatting for key metrics

### **Alert Delivery**

All alerts are delivered via **Telegram** using HTML formatting. The system supports:
- Real-time trade notifications
- System status updates
- Performance summaries
- Error and warning alerts
- OAuth token management alerts

---

## 📱 **Telegram Setup Guide**

### **Step 1: Create a Telegram Bot**

1. Open Telegram and search for **@BotFather**
2. Start a conversation with BotFather
3. Send the command: `/newbot`
4. Follow the prompts to:
   - Choose a name for your bot (e.g., "Easy ORB Strategy Alerts")
   - Choose a username for your bot (must end in `bot`, e.g., `easy_orb_alerts_bot`)
5. BotFather will provide you with a **Bot Token** (e.g., `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)
6. **Save this token** - you'll need it for configuration

### **Step 2: Get Your Chat ID**

1. Open Telegram and search for **@userinfobot**
2. Start a conversation with @userinfobot
3. The bot will reply with your **Chat ID** (e.g., `123456789`)
4. **Save this Chat ID** - you'll need it for configuration

**Alternative Method** (if @userinfobot doesn't work):
1. Create a group chat or use an existing one
2. Add your bot to the group
3. Send a message in the group
4. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
5. Look for the `"chat":{"id"` field in the response
6. Use the **negative number** (e.g., `-123456789`) for group chats

### **Step 3: Configure Telegram Credentials**

#### **For Local Development**

1. Navigate to `secretsprivate/` directory
2. Copy `telegram.env.template` to `telegram.env`
3. Edit `telegram.env` and add your credentials:
   ```bash
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   TELEGRAM_ENABLED=true
   ```
4. **Important**: `telegram.env` is gitignored - never commit it to version control

#### **For Cloud Deployment (GCP)**

Store credentials in Google Cloud Secret Manager:

```bash
# Store bot token
echo -n "your_bot_token_here" | gcloud secrets create telegram-bot-token --data-file=-

# Store chat ID
echo -n "your_chat_id_here" | gcloud secrets create telegram-chat-id --data-file=-
```

The system will automatically load these credentials in production.

### **Step 4: Test Your Setup**

1. Start the trading system
2. You should receive a test alert or the Good Morning alert
3. If you don't receive alerts, check:
   - Bot token is correct
   - Chat ID is correct
   - Bot is not blocked
   - System logs for errors

### **Step 5: Configure Alert Preferences**

Edit `configs/alerts.env` (or use `configs/alerts.env.template` as a base):

```bash
# Enable Telegram alerts
TELEGRAM_ALERTS_ENABLED=true

# Rate limiting (prevent spam)
TELEGRAM_MAX_MESSAGES_PER_MINUTE=20
TELEGRAM_RATE_LIMIT_ENABLED=true
TELEGRAM_ALERT_COOLDOWN_SECONDS=60

# Alert types to receive
TELEGRAM_ALERT_TYPES=entry,exit,error,performance,daily_summary,system_status
```

### **Troubleshooting Telegram Setup**

**Issue**: No alerts received
- ✅ Verify bot token is correct
- ✅ Verify chat ID is correct
- ✅ Check that bot is not blocked
- ✅ Review system logs for errors
- ✅ Test bot manually: `https://api.telegram.org/bot<TOKEN>/getMe`

**Issue**: Alerts received but formatting is broken
- ✅ Check that HTML formatting is enabled (default)
- ✅ Verify message doesn't contain invalid HTML
- ✅ System will auto-fallback to plain text if HTML fails

**Issue**: Too many alerts (spam)
- ✅ Adjust `TELEGRAM_MAX_MESSAGES_PER_MINUTE` in config
- ✅ Enable `TELEGRAM_RATE_LIMIT_ENABLED`
- ✅ Increase `TELEGRAM_ALERT_COOLDOWN_SECONDS`

---

## 📊 **ORB Strategy Alerts**

### **1. Good Morning Alert (5:30 AM PT / 8:30 AM ET)**

**Trigger**: Cloud Scheduler at 5:30 AM PT daily  
**Purpose**: System status check and token validation  
**Content**:
- Token status (valid/expired)
- Configuration mode (Demo/Live)
- System health check
- Trading readiness status

**Features**:
- Time validation: Only sends between 5:30-5:35 AM PT
- Deduplication: One alert per day maximum (GCS-based)
- Protection: Rejects calls outside valid window

**Example**:
```
====================================================================

🌅 <b>Good Morning</b> | 🎮 DEMO Mode
          Time: 05:30 AM PT (08:30 AM ET)

✅ <b>System Status:</b>
          • OAuth Tokens: VALID ✅
          • Trading Mode: DEMO
          • System: READY

📊 <b>Next:</b> ORB Capture at 6:45 AM PT
          (Opening Range Breakout data collection)

====================================================================
```

### **2. ORB Capture Complete (6:45 AM PT / 9:45 AM ET)**

**Trigger**: After ORB capture completes (6:45 AM PT)  
**Purpose**: Confirmation of opening range capture  
**Content**:
- Number of symbols captured (dynamic count, typically 145)
- Capture method (E*TRADE batch quotes only — broker-only, no third-party fallback)
- Processing time
- Any errors (no fallback; see Data.md for broker-only data source)
- Symbol count breakdown

**Example**:
```
====================================================================

✅ <b>ORB Capture Complete</b>
          Time: 06:45 AM PT (09:45 AM ET)

📊 <b>Capture Summary:</b>
          • Symbols Captured: 145
          • Method: E*TRADE Batch Quotes
          • Processing Time: 2.3 seconds

📈 <b>Next:</b> Signal Collection at 7:30 AM PT
          (Trade signal generation)

====================================================================
```

### **3. Trade Signal Collection (7:30 AM PT / 10:30 AM ET)**

**Trigger**: After signal collection and rules confirmation completes (7:30 AM PT)  
**Purpose**: Single alert showing final confirmed trade lists (after all rules and risk management)  
**Content**:
- **SO Signal Collection**: Final confirmed SO trades ready for execution
  - Number of confirmed SO trades (typically 6-15, or 0)
  - All rules and risk management applied
  - Final execution-ready list (SO list is long-only)
- **0DTE Signal Collection**: Final confirmed 0DTE options trades ready for execution (if enabled)
  - Shows **CALL (Long)** and **PUT (Short)** breakdown
  - Number of confirmed 0DTE options trades (qualified)
  - All rules and risk management applied
  - Final execution-ready list (0DTE includes both Long and Short)
- When there are **0 signals**, the alert may include a **diagnostic reason** (`zero_signals_reason`) to help troubleshoot (e.g. validation candle data not available, all NEUTRAL, volume color, ORB data, rule rejection). See [CLOUD_LOGS_0_SIGNALS_FEB19.md](doc_elements/Sessions/2026/Feb19%20Session/CLOUD_LOGS_0_SIGNALS_FEB19.md) for log diagnosis.

**Note**: SO list is long-only. 0DTE list includes both Long (CALL) and Short (PUT). Both lists represent **final execution-ready trades** confirmed to open positions (after all rules and risk management). There is no separate 0DTE Signal Collection alert—0DTE is included in this single Trade Signal Collection alert.

**Example**:
```
====================================================================

📊 <b>Trade Signal Collection</b>
          Time: 07:30 AM PT (10:30 AM ET)

✅ <b>SO Signals (ORB Strategy):</b>
          • Confirmed Trades: 8
          • All rules applied: ✅
          • Ready for execution: ✅

✅ <b>0DTE Signals (Options Strategy):</b>
          • Confirmed Trades: 3
          • All rules applied: ✅
          • Ready for execution: ✅

📈 <b>Next:</b> Execution at 7:30 AM PT

====================================================================
```

### **4. Standard Order Execution (7:30 AM PT / 10:30 AM ET)**

**Trigger**: After batch execution completes (7:30 AM PT)  
**Purpose**: Detailed execution summary with enhanced formatting  
**Content**:
- Number of trades executed
- Total capital deployed
- Capital efficiency percentage
- Position details for each trade:
  - **Symbol** (e.g., QQQ, SPY)
  - **Quantity** (shares)
  - **Entry Price**
  - **<b>Rank #X</b>** (bold priority rank)
  - **<b>Priority Score: 0.856</b>** (bold priority score)
  - **<b>Confidence: 85%</b>** (bold confidence)
  - **<b>Momentum: 75/100</b>** (bold momentum)
  - **Trade ID**: Shortened format (e.g., `DEMO_QQQ_260105_485_488_c_704400`)

**Example**:
```
====================================================================

✅ <b>Standard Order Execution</b>
          Time: 07:30 AM PT (10:30 AM ET)

📊 <b>Execution Summary:</b>
          Trades Executed: 6
          Capital Deployed: $792.50 (88.1%)
          Capital Efficiency: 88.1%

📈 <b>Positions:</b>
          • QQQ - 12 shares @ $42.50
            <b>Rank #1</b> | <b>Priority Score: 0.856</b>
            <b>Confidence: 85%</b> | <b>Momentum: 75/100</b>
            Trade ID: DEMO_QQQ_260106_485_488_c_704400

          • SPY - 8 shares @ $485.00
            <b>Rank #2</b> | <b>Priority Score: 0.823</b>
            <b>Confidence: 82%</b> | <b>Momentum: 68/100</b>
            Trade ID: DEMO_SPY_260106_485_488_c_704401

====================================================================
```

### **5. Portfolio Health Check (Every 15 Minutes)**

**Trigger**: Every 15 minutes (7:45 AM - 12:45 PM PT)  
**Purpose**: Monitor portfolio health and trigger emergency exits  
**Content**:
- **EMERGENCY** (3+ red flags): Close ALL positions immediately
- **WARNING** (2 red flags): Close weak positions (P&L < -0.5%)
- **OK** (0-1 red flags): Continue normal trading (no alert, log only)

**Red Flags Monitored**:
- Win rate <35%
- Avg P&L <-0.5%
- Low momentum <40%
- Weak peaks <0.8%
- All positions losing (100% losers)

**Example (EMERGENCY)**:
```
====================================================================

🚨 <b>Portfolio Health: EMERGENCY</b>
          Time: 08:00 AM PT (11:00 AM ET)

⚠️ <b>Red Flags Detected:</b>
          • Win Rate: 20% (< 35%)
          • Avg P&L: -1.2% (< -0.5%)
          • Low Momentum: 30% (< 40%)

🔄 <b>Action:</b> Closing ALL positions immediately
          (Emergency exit to preserve capital)

====================================================================
```

### **6. Position Exit Alerts**

#### **Individual Exits**

**Trigger**: When individual position closes  
**Purpose**: Detailed exit information  
**Content**:
- Exit reason (trailing stop, breakeven, rapid exit, etc.)
- Entry and exit prices
- P&L (absolute and percentage)
- Hold time
- Peak price reached
- Trade ID (shortened format)

**Example**:
```
====================================================================

🔄 <b>Position Closed</b>
          Time: 09:15 AM PT (12:15 PM ET)

📉 <b>QQQ - 12 shares</b>
          Entry: $42.50 → Exit: $43.15
          P&L: +$7.80 (+1.53%)
          Hold Time: 1h 45m
          Peak: $43.25 (+1.76%)
          Exit Reason: Trailing Stop
          Trade ID: DEMO_QQQ_260106_485_488_c_704400

====================================================================
```

#### **Aggregated Exits (Batch Closes)**

**Trigger**: Batch closes (EOD, emergency, weak day)  
**Purpose**: ONE alert for all positions closed  
**Content**:
- Summary of exit reasons
- Total P&L
- Number of positions closed
- Individual position details (if space permits)
- Prevents duplicate notifications

**Example**:
```
====================================================================

🔄 <b>End of Day Close</b>
          Time: 12:55 PM PT (03:55 PM ET)

📊 <b>Summary:</b>
          Positions Closed: 6
          Total P&L: +$45.23 (+5.7%)

📈 <b>Positions:</b>
          • QQQ: +$12.50 (+2.1%) - Trailing Stop
          • SPY: +$8.75 (+1.8%) - Breakeven
          • TQQQ: +$15.20 (+3.2%) - EOD Close
          • SOXL: +$4.50 (+0.9%) - EOD Close
          • UPRO: +$2.28 (+0.5%) - EOD Close
          • NEBX: +$2.00 (+0.3%) - EOD Close

====================================================================
```

### **7. Rapid Exit Alerts**

**Trigger**: When trades rapidly exit (no momentum or reversal)  
**Purpose**: Notification of early exits to prevent losses  
**Content**:
- Exit reason (NO_MOMENTUM or IMMEDIATE_REVERSAL)
- Time held
- Entry and exit prices
- P&L
- Peak price reached
- Trade ID

**Example**:
```
====================================================================

🚨 <b>RAPID EXIT - No Momentum</b>

⏰ <b>Time Held:</b> 18 minutes

📉 <b>12 QQQ @ $42.50</b> • <b>$510.00</b>
  • <b>Current P&L:</b> -0.15% (-$0.77)
  • <b>Peak:</b> $42.63 (+0.30%)
  • Trade ID: DEMO_QQQ_260106_485_488_c_704400

🚨 <b>Exit Reason:</b>
  Peak movement <+0.3% after 15 minutes
  Trade shows no momentum - exiting to limit loss

💡 <b>Action:</b> Position closed at -0.15%
   Early exit to prevent further loss

====================================================================
```

### **8. Red Day Alert**

**Trigger**: When Red Day is detected during signal collection  
**Purpose**: Notification that trading is blocked due to market conditions  
**Content**:
- Red Day reason (pattern description)
- Market conditions (RSI, volume, MACD, etc.)
- Action taken (ORB trades blocked, 0DTE PUTs allowed)
- Signal collection summary
- Strategy impact

**Example**:
```
====================================================================

🚨 <b>RED DAY DETECTED</b> | 🎮 DEMO Mode
          Time: 07:30 AM PT (10:30 AM ET)

🔍 <b>Red Day Reason:</b>
          Oversold market conditions detected (RSI < 40 in 60%+ symbols)

📊 <b>Market Conditions:</b>
          • Oversold (RSI under 40): 65%
          • Overbought (RSI above 80): 5%
          • Weak Volume (below 1.0x): 45%
          • Avg RSI: 38.5
          • Avg Volume: 0.85x

💰 <b>Action Taken:</b>
          • ORB trades: <b>BLOCKED</b> (capital preserved)
          • 0DTE options: <b>CONTINUING</b> (will prioritize PUTs)

📊 <b>Signal Collection:</b>
          • Symbols Scanned: 145
          • ORB Signals: 8 (blocked from execution)
          • 0DTE Signals: 3 (will process PUTs)

====================================================================
```

### **9. Holiday Alert**

**Trigger**: When market is closed or low-volume day detected (5:30 AM PT, instead of Good Morning)  
**Purpose**: Notification that trading is skipped  
**Content**:
- Holiday name and date
- Skip reason: **MARKET_CLOSED** (bank holiday, market closed; emoji 🏖️) or **LOW_VOLUME** (market open but low volume, e.g. Halloween; emoji 🎃)
- Next trading day
- System status

**Example**:
```
====================================================================

🏖️ <b>Market Holiday</b>
          Time: 05:30 AM PT (08:30 AM ET)

📅 <b>Holiday:</b> Christmas Day
          Reason: MARKET_CLOSED

⏭️ <b>Next Trading Day:</b> Thursday, December 26, 2025

✅ <b>System Status:</b>
          • Trading: SKIPPED
          • System: IDLE
          • Next Alert: Tomorrow at 5:30 AM PT

====================================================================
```

### **10. End-of-Day Report (4:05 PM ET / 1:05 PM PT)**

**Trigger**: Cloud Scheduler endpoint at 4:05 PM ET (1:05 PM PT) daily  
**Source**: Single source - Cloud Scheduler endpoint only (Rev 00260)  
**Deduplication**: GCS-based (prevents duplicate reports)  
**Purpose**: Daily and weekly performance summary  
**Content**:
- Account balance (Demo/Live)
- Total P&L for the day
- Win rate
- Number of trades
- Average P&L per trade
- Best and worst trades
- All-time statistics (if available)
- Weekly summary (if applicable)

**Note**: EOD reports are triggered ONLY by Cloud Scheduler endpoint (`/api/end-of-day-report`). Internal trading loop EOD triggers were removed (Rev 00260) to prevent duplicate reports. GCS-based deduplication ensures only one report per day even if Cloud Scheduler retries.

**Example**:
```
====================================================================

📊 <b>End-of-Day Report</b>
          Date: January 9, 2026
          Time: 01:05 PM PT (04:05 PM ET)

💰 <b>P&L (TODAY):</b>
          <b>+5.7%</b> +$45.23
          Win Rate: 66.7% • Total Trades: 6
          Profit Factor: 2.1
          Average Win: $12.50
          Average Loss: -$5.20
          Best Trade: +$15.20
          Worst Trade: -$2.00

🎖️ <b>P&L (WEEK M-F):</b>
          <b>+12.3%</b> +$98.50
          Win Rate: 62.5% • Total Trades: 24
          Profit Factor: 1.8

💎 <b>Account Balances (All Time):</b>
          <b>+45.2%</b> +$362.00
          <b>$1,162.00</b>
          Win Rate: 58.3% • Total Trades: 120
          Profit Factor: 1.6
          Wins: 70 • Losses: 50

====================================================================
```

### **11. ORB Capture Failed Alert**

**Trigger**: When ORB capture fails for all symbols  
**Purpose**: Notification of data collection failure  
**Content**:
- Number of symbols attempted
- Failure reason
- Recovery actions taken
- Next steps

---

## 🎯 **0DTE Strategy Alerts**

When `ENABLE_0DTE_STRATEGY=true` is set in `configs/deployment.env`, the 0DTE Strategy has its own comprehensive alert system integrated with the ORB Strategy alerts.

### **1. 0DTE ORB Capture (Integrated with ORB Capture Complete)**

**Trigger**: After ORB capture completes (6:45 AM PT)  
**Purpose**: Same as ORB Capture Complete; when 0DTE is enabled, the **single** ORB Capture Complete alert includes 0DTE symbol **counts**.  
**Content** (in the one ORB Capture Complete alert):
- **Opening Range Capture**: ORB Strategy symbols captured and active counts
- **0DTE ORB Capture** (when 0DTE enabled): 0DTE symbols captured count and active count, same capture duration
- Detailed 0DTE symbol ORB data (e.g. SPX/QQQ/SPY high/low) is **not** shown in this alert (simplified per Rev 00186)

**Note**: There is no separate "0DTE ORB Capture" alert. The main **ORB Capture Complete** alert includes both ORB Strategy and 0DTE Strategy **counts** when 0DTE is enabled. Watchlist path for 0DTE symbols: `data/watchlist/0dte_list.csv`.

### **2. 0DTE Options Signal Collection (Integrated with Trade Signal Collection)**

**Trigger**: At 7:30 AM PT, same time as Trade Signal Collection  
**Purpose**: Summary of qualified 0DTE options signals; **no separate alert**—0DTE is included in the **Trade Signal Collection** alert.  
**Content** (in the single Trade Signal Collection alert when 0DTE enabled):
- SO (ORB) confirmed trades count and list (long-only)
- 0DTE confirmed trades: **CALL (Long)** and **PUT (Short)** breakdown, counts, and symbol list
- All rules and risk management applied; final execution-ready lists for both

**Integration**: There is only one Signal Collection alert. It shows both ORB Strategy (SO) and 0DTE Strategy results. A deprecated separate `send_options_signal_collection_alert` exists in code but is not used; the unified `send_so_signal_collection()` sends the combined alert.

### **3. 0DTE Options Execution Alert**

**Trigger**: After 0DTE options execution completes (7:30 AM PT, after ORB execution)  
**Purpose**: Detailed execution summary with enhanced formatting  
**Content**:
- Number of options trades executed
- Total capital deployed
- Capital efficiency percentage
- Strategy types breakdown (debit spreads, long calls/puts, lotto sleeves)
- Position details for each trade:
  - **Symbol** (SPX, QQQ, SPY, IWM, etc.)
  - **Strategy Type** (debit spread, long call, long put, lotto, momentum scalper, ITM probability)
  - **Strikes** (for spreads: long leg / short leg; for single-leg: strike)
  - **Delta** (target delta achieved)
  - **Premium** (entry premium per contract)
  - **Quantity** (number of contracts)
  - **Capital Allocated** (position size)
  - **Trade ID**: Shortened format (e.g., `DEMO_SPX_260106_485_488_c_704400`)

**Example**:
```
====================================================================

✅ <b>0DTE Options Execution</b>
          Time: 07:30 AM PT (10:30 AM ET)

📊 <b>Execution Summary:</b>
          Trades Executed: 3
          Capital Deployed: $1,250.00 (25.0%)
          Capital Efficiency: 25.0%

📈 <b>Positions:</b>
          • QQQ - Long Call 628c @ $0.19 (100 contracts)
            Strategy: <b>Long Call</b> | <b>Delta: 0.15</b>
            Trade ID: DEMO_QQQ_260109_628_c_704400

          • SPY - Debit Spread 485/486c @ $0.07 (50 spreads)
            Strategy: <b>Debit Spread</b> | <b>Delta: 0.20</b>
            Trade ID: DEMO_SPY_260109_485_486_d_704401

          • IWM - Long Call 257c @ $0.35 (25 contracts)
            Strategy: <b>Long Call</b> | <b>Delta: 0.15</b>
            Trade ID: DEMO_IWM_260109_257_c_704402

====================================================================
```

### **4. 0DTE Options Position Exit Alerts**

#### **Individual Exits**

**Trigger**: When individual options position closes  
**Purpose**: Detailed exit information with real-time options P&L  
**Content**:
- Exit reason (profit target +60%, profit target +120%, hard stop, time stop, invalidation, EOD, etc.)
- Entry and exit prices (real options prices, not underlying)
- P&L (absolute and percentage) - based on actual options moves
- Hold time
- Peak value reached
- Strategy type (debit spread, long call, etc.)
- Trade ID (shortened format)

**Real-Time Price Tracking**:
- Exit decisions based on actual options prices (fetched from E*TRADE API every 30 seconds)
- P&L calculated from real options moves (e.g., $0.19 → $0.97 = +410%)
- Accurate profit target triggers and exit decisions

#### **Aggregated Exits**

**Trigger**: Batch closes (EOD at 12:55 PM PT, emergency exits)  
**Purpose**: ONE alert for all options positions closed  
**Content**:
- Summary of exit reasons
- Total P&L (based on actual options prices)
- Number of positions closed
- Individual position details (if space permits)
- Strategy breakdown (debit spreads, long calls, etc.)

**Example**:
```
====================================================================

🔄 <b>0DTE Options End of Day Close</b>
          Time: 12:55 PM PT (03:55 PM ET)

📊 <b>Summary:</b>
          Positions Closed: 3
          Total P&L: +$1,250.00 (+250.0%)

📈 <b>Positions:</b>
          • QQQ 628c: +$780.00 (+410.5%) - Profit Target +60%
          • SPY 485/486c: +$350.00 (+100.0%) - EOD Close
          • IWM 257c: +$120.00 (+137.1%) - EOD Close

====================================================================
```

### **5. 0DTE Options Partial Profit Alert**

**Trigger**: When partial profit is taken (automated profit targets enabled)  
**Purpose**: Notification of partial profit realization  
**Content**:
- Partial profit amount (based on actual options prices)
- Remaining position size
- Current P&L (real-time options prices)
- Profit target reached (+60% or +120%)
- Strategy type
- Trade ID

**Automated Profit Targets**:
- **First Target**: +60% → Sell 50% of position
- **Second Target**: +120% → Sell 25% of remaining position
- **Runner**: Trails remaining position until exit conditions

### **6. 0DTE Options Runner Exit Alert**

**Trigger**: When runner position exits (after partial profits taken)  
**Purpose**: Notification of runner exit  
**Content**:
- Runner exit P&L (based on actual options prices)
- Total position P&L (partial profits + runner)
- Exit reason (VWAP reclaim, ORB midpoint reclaim, time cutoff, etc.)
- Strategy type
- Trade ID

### **7. 0DTE Options Health Check Alert (Optional)**

**Trigger**: Health check for options positions (if enabled)  
**Purpose**: Monitor options portfolio health  
**Content**:
- Health status (OK, WARNING, EMERGENCY)
- Open positions summary
- Risk metrics (total capital at risk, max position size, etc.)
- Real-time options P&L

### **8. 0DTE Options End-of-Day Report**

**Trigger**: Cloud Scheduler endpoint at 4:05 PM ET (1:05 PM PT) daily  
**Source**: Single source - Cloud Scheduler endpoint only (Rev 00260)  
**Deduplication**: GCS-based (prevents duplicate reports - Rev 00260)  
**Purpose**: Daily options performance summary

**Note**: 0DTE EOD reports are triggered ONLY by Cloud Scheduler endpoint (`/api/end-of-day-report`). Internal trading loop EOD triggers were removed (Rev 00260) to prevent duplicate reports. GCS-based deduplication ensures only one report per day even if Cloud Scheduler retries.  
**Content**:
- Total options P&L (based on actual options prices)
- Number of options trades executed
- Win rate (winning trades / total trades)
- Best and worst trades (with strategy types)
- Strategy breakdown:
  - Debit spreads: count, P&L, win rate
  - Long calls/puts: count, P&L, win rate
  - Momentum scalpers: count, P&L, win rate
  - ITM probability spreads: count, P&L, win rate
  - Lotto sleeves: count, P&L, win rate
- Average P&L per trade
- Capital efficiency (capital deployed / available capital)

**Example**:
```
====================================================================

📊 <b>0DTE Options End-of-Day Report</b>
          Date: January 9, 2026
          Time: 01:05 PM PT (04:05 PM ET)

💰 <b>Performance Summary:</b>
          Total P&L: +$1,250.00 (+250.0%)
          Trades Executed: 3
          Win Rate: 100% (3/3)
          Average P&L: +$416.67 per trade

📈 <b>Strategy Breakdown:</b>
          • Long Calls: 2 trades, +$900.00 (+450.0% avg)
          • Debit Spreads: 1 trade, +$350.00 (+100.0%)

🎯 <b>Best Trade:</b> QQQ 628c @ $0.19 → $0.97 (+410.5%)
📉 <b>Worst Trade:</b> SPY 485/486c @ $0.07 → $0.14 (+100.0%)

====================================================================
```

---

## 🔐 **OAuth System Alerts**

### **1. OAuth Token Expiry Alert (Midnight)**

**Trigger**: Cloud Scheduler at 9:00 PM PT (12:00 AM ET) daily  
**Purpose**: Alert when production token expires at midnight ET (sandbox deprecated — only production is used for data)  
**Delivery**: Direct Telegram API (works 24/7, independent of main trading system)  
**Independence**: Sends even when trading system is not actively running

**Example**:
```
====================================================================

⚠️ <b>OAuth Tokens Expired</b>
          Time: 09:00 PM PT (12:00 AM ET)

🚨 <b>Token Status:</b>
          E*TRADE tokens are <b>EXPIRED</b> ❌

🌐 <b>Public Dashboard:</b>
          https://YOUR_OAUTH_WEB_APP_URL

⚠️ Renew Production Token (used for all data and trading — Demo and Live)

👉 <b>Action Required:</b>
1. Visit the public dashboard
2. Click "Renew Production" (production token only; sandbox not used)
3. Enter access code (store in your environment or secret manager)
4. Complete OAuth authorization
5. Token will be renewed and stored

====================================================================
```

### **2. OAuth Production Token Renewed**

**Trigger**: Successful production token renewal (via any management portal URL)  
**Purpose**: Confirmation when production tokens are renewed  
**Delivery**: Direct Telegram API call (works 24/7, independent of trading system)

### **3. OAuth Sandbox Token Renewed** *(deprecated)*

**Trigger**: Sandbox token renewal (deprecated — only production tokens are used for data and trading)  
**Purpose**: Informational only. Sandbox tokens are deprecated; the app uses **production tokens only** for data and API. You only need to renew the production token.
**Delivery**: Direct Telegram API call (works 24/7, independent of trading system)

---

## ⚙️ **Alert Configuration**

### **Configuration Files**

**Local Development**: `secretsprivate/telegram.env`
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ENABLED=true
```

**Production**: Google Cloud Secret Manager
- `telegram-bot-token`
- `telegram-chat-id`

**Alert Settings**: `configs/alerts.env`
```bash
# Enable Telegram alerts
TELEGRAM_ALERTS_ENABLED=true

# Rate limiting
TELEGRAM_MAX_MESSAGES_PER_MINUTE=20
TELEGRAM_RATE_LIMIT_ENABLED=true
TELEGRAM_ALERT_COOLDOWN_SECONDS=60

# Alert types
TELEGRAM_ALERT_TYPES=entry,exit,error,performance,daily_summary,system_status
```

### **Alert Manager**

The alert manager (`modules/prime_alert_manager.py`) handles:
- Alert formatting
- Telegram delivery
- Error handling
- Rate limiting
- Alert deduplication
- Trade ID generation

---

## 📅 **Daily Alert Flow**

**Typical Trading Day** (Monday-Friday):

1. **9:00 PM PT (Midnight ET)**: OAuth Tokens Expired alert 🔴 (if tokens expired)
2. **5:30 AM PT (8:30 AM ET)**: Good Morning alert 🌅
   - Time validation: Only sends 5:30-5:35 AM PT
   - Deduplication: One alert per day maximum
3. **6:45 AM PT**: ORB Capture Complete (all symbols captured)
4. **7:30 AM PT - Step 1**: Trade Signal Collection (SO + 0DTE in one alert; shows 6-15 signals or 0 signals; when 0 signals, may include diagnostic reason)
   - Deduplication: GCS-based
5. **7:30 AM PT - Step 2**: SO Execution (shows executed trades with bold formatting)
6. **7:30 AM PT - Step 3**: 0DTE Options Execution (if enabled, shows options trades)
7. **7:45 AM PT & Every 15 Min**: Portfolio Health Check 🛡️
   - EMERGENCY alert (3+ red flags, aggregated close all)
   - WARNING alert (2 red flags, aggregated close weak)
   - OK (no alert, log only)
8. **Throughout Day**: Smart Loss Prevention & Normal Exits 🛡️
   - **Individual exits**: Trailing stop, breakeven, rapid exit (1 position)
   - **Health check exits**: Emergency/weak day (aggregated)
   - **0DTE exits**: Individual and aggregated options exits
9. **12:55 PM PT**: EOD Close (1 aggregated alert for all positions)
10. **1:05 PM PT (4:05 PM ET)**: End-of-Day Report (TODAY + WEEKLY summary) - Single source: Cloud Scheduler endpoint only (Rev 00260)

**Holidays/Weekends**:
- **Holiday Alert**: Sent at 5:30 AM PT instead of Good Morning alert
- **No Trading Alerts**: System skips trading-related alerts on holidays

---

## 🎨 **Alert Formatting**

### **Execution Alerts**

**Enhanced Formatting**:
- **Bold Priority Rank**: `<b>Rank #1</b>`
- **Bold Priority Score**: `<b>Priority Score: 0.856</b>`
- **Bold Confidence**: `<b>Confidence: 85%</b>`
- **Bold Momentum**: `<b>Momentum: 75/100</b>`
- **Bold Delta**: `<b>Delta: 0.25</b>`

**Trade ID Format**:
- **Shortened Format**: `DEMO_QQQ_260105_485_488_c_704400`
- **Components**:
  - Mode: `DEMO` or `LIVE`
  - Symbol: `QQQ`, `SPY`, `SPX`, `IWM`, etc.
  - Date: `260105` (YYMMDD format)
  - Strike/Price info: `485_488` (spread) or `628` (single-leg)
  - Strategy type: `c` (call), `p` (put), `d` (debit), `cr` (credit), `l` (lotto)
  - Unique ID: `704400`
- **Applied To**: All strategies (ORB and 0DTE), both Demo and Live modes

### **Exit Alerts**

- Clear exit reason
- Entry/exit prices
- P&L highlighted
- Hold time displayed
- Peak price reached
- Trade ID (shortened format)

### **Error Alerts**

- Error type and message
- Affected symbols or positions
- Recovery actions taken
- Next steps

---

## 🛠️ **Troubleshooting**

### **Common Issues**

**1. Alerts Not Received**
- ✅ Check Telegram bot token and chat ID configuration
- ✅ Verify Cloud Scheduler jobs are running
- ✅ Check Cloud Run service logs for errors
- ✅ Test bot manually: `https://api.telegram.org/bot<TOKEN>/getMe`
- ✅ Verify bot is not blocked

**2. Duplicate Alerts**
- ✅ Should be fixed with deduplication (GCS-based)
- ✅ Check alert deduplication logic
- ✅ Verify Cloud Scheduler jobs aren't running multiple times

**3. Trade IDs Too Long**
- ✅ Should be fixed with shortened format (Rev 00231)
- ✅ Verify shortened format is being used
- ✅ Format: `DEMO_QQQ_260105_485_488_c_704400`

**4. Missing Bold Formatting**
- ✅ Should be fixed with enhanced formatting (Rev 00231)
- ✅ Check alert manager formatting code
- ✅ Verify HTML formatting is enabled

**5. Too Many Alerts (Spam)**
- ✅ Adjust `TELEGRAM_MAX_MESSAGES_PER_MINUTE` in config
- ✅ Enable `TELEGRAM_RATE_LIMIT_ENABLED`
- ✅ Increase `TELEGRAM_ALERT_COOLDOWN_SECONDS`

**6. Alerts Received But Formatting Broken**
- ✅ Check that HTML formatting is enabled (default)
- ✅ Verify message doesn't contain invalid HTML
- ✅ System will auto-fallback to plain text if HTML fails

### **Useful Commands**

```bash
# Test Telegram bot
curl "https://api.telegram.org/bot<TOKEN>/getMe"

# Test sending message
curl -X POST "https://api.telegram.org/bot<TOKEN>/sendMessage" \
  -d "chat_id=<CHAT_ID>" \
  -d "text=Test message"

# View recent logs
gcloud run services logs read YOUR_SERVICE_NAME --region=us-central1 --limit 50
```

---

## 📝 **Revision History**

### **Latest Updates (February 9, 2026 - Rev 00259)**

**Rev 00259 (Feb 9 - Alerts doc alignment)**:
- ✅ Trade Signal Collection: clarified single alert for SO + 0DTE; 0-signals diagnostic reason documented
- ✅ 0DTE ORB Capture: clarified integrated into ORB Capture Complete (counts only; no separate alert)
- ✅ 0DTE Signal Collection: clarified no separate alert; included in Trade Signal Collection
- ✅ Revision history and footer date aligned

**Rev 00260 (EOD Single Source Consolidation)**:
- ✅ EOD reports consolidated to single source (Cloud Scheduler endpoint only)
- ✅ GCS-based deduplication active for both ORB and 0DTE EOD reports
- ✅ Timing: 1:05 PM PT (4:05 PM ET)

### **Previous Updates**

**Rev 00246 (Jan 19 - 0DTE Strategy Improvements)**:
- ✅ 0DTE Priority Score Formula v1.1
- ✅ Direction-Aware Red Day Filtering
- ✅ Expanded Delta Selection (0.15-0.35)

**Rev 00233 (Jan 8 - Performance Improvements & Data Quality Fixes)**:
- ✅ Secrets Management: All sensitive credentials moved to `secretsprivate/`
- ✅ Good Morning Alert Time Validation: Only sends 5:30-5:35 AM PT
- ✅ Good Morning Alert Deduplication: GCS-based (one alert per day maximum)

**Rev 00231 (Jan 6 - Trade ID Shortening & Alert Formatting)**:
- ✅ Trade ID Shortening: Shortened trade IDs for cleaner format
- ✅ Alert Formatting Enhancements: Bold formatting for key metrics

---

**Alert System User Guide - Complete and Ready for Use!** 🚀

*Last Updated: February 19, 2026*  
*Version: Rev 00260 + Feb19 (0-signals diagnostic, validation candle); EOD single source*  
*Maintainer: Easy ORB Strategy Development Team*

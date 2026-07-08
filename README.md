# Nifty 50 Options Strategy Backtester

A complete 10-year backtesting system for Nifty 50 weekly options strategies, built with Python and interactive HTML dashboards. Tests 15 strategies across 3 capital tiers using Black-Scholes pricing on 286,000+ real intraday candles (2016–2026).

---

## Project Files

| File | Description |
|---|---|
| `strategy_backtest_v3_dashboard.html` | **Main interactive dashboard** — charts, heatmaps, metrics for all 15 strategies |
| `strategy_guide.html` | **Strategy explanation guide** — how each strategy works, examples, pros/cons |
| `backtest_strategies_v3.py` | **Main backtester script (v3)** — run this to regenerate results |
| `backtest_strategies_v2.py` | Previous version (v2, 12 strategies) |
| `backtest_strategies.py` | Original version (v1, 6 strategies) |
| `NIFTY50_INTRADAY_5m_10YEARS.csv` | 5-minute intraday data — 2016 to 2023 |
| `NIFTY50_5min_angelone.csv` | 5-minute intraday data — 2023 to 2026 (Angel One) |
| `NIFTY50_MASTER_DAILY.csv` | Daily OHLC + India VIX data — 2016 to 2026 |
| `NIFTY50_15min_angelone.csv` | 15-minute intraday data (Angel One) |
| `NIFTY50_1min_angelone.csv` | 1-minute intraday data (Angel One) |



## How to Run

### Requirements
```
pip install pandas numpy scipy
```

### Run Backtester
```bash
python -X utf8 backtest_strategies_v3.py
```

This will:
1. Load all intraday + daily data
2. Run the Entry Time Optimizer across 15 time slots
3. Backtest all 15 strategies for 3 capital tiers
4. Print full results to console
5. Save `strategy_backtest_v3_dashboard.html`

### Open Dashboard
Double-click `strategy_backtest_v3_dashboard.html` in Chrome or Edge.

---

## Capital Tiers

| Tier | Capital | Lots | Units |
|---|---|---|---|
| Small | < ₹50,000 | 3 lots | 195 units |
| Medium | < ₹1,50,000 | 10 lots | 650 units |
| Large | > ₹1,50,000 | 20 lots | 1,300 units |

> **Lot size: 65 units** (Nifty 50 weekly options)

---

## Strategy Overview

### ✅ SELL Strategies (Premium Collection) — All Profitable

| ID | Name | When | Entry | Exit | Stop | Win Rate | 10Y P&L (Medium) |
|---|---|---|---|---|---|---|---|
| S3 | Expiry Straddle Sell | Thursday only | 9:20 AM | 2:30 PM | None | 89.1% | ₹1.42 Cr |
| S13 | Daily Straddle Sell+Stop | Every day | 9:30 AM | 2:30 PM | 0.6% move @ 12PM | 84.6% | ₹4.29 Cr |
| S14 | Iron Condor Sell | Every day | 10:00 AM | 2:30 PM | Hedge covers it | 84.8% | ₹1.82 Cr |
| S15 | Mid-Week Strangle Sell | Tue/Wed/Thu | 9:30 AM | 2:30 PM | 1.2% move @ 1PM | 96.1% | ₹1.36 Cr |

---

### ❌ BUY Strategies (Directional) — All Unprofitable Over 10 Years

| ID | Name | When | Entry | Exit | Win Rate | 10Y P&L (Medium) |
|---|---|---|---|---|---|---|
| S1 | ORB-15 Breakout | Break of 15-min range | 9:30 AM | 1:00 PM | 8.8% | -₹2.43 Cr |
| S2 | VIX + Gap Signal | Gap > 0.3% | 9:20 AM | 12:30 PM | 26.1% | -₹94.6 L |
| S4 | Monday Gap Reversal | Monday + Gap > 0.5% | 9:20 AM | 11:00 AM | 36.4% | -₹2.26 L |
| S5 | 3-Day Momentum Buy | 3 consecutive same-direction days | 9:20 AM | 12:00 PM | 25.4% | -₹56 L |
| S6 | High-VIX Straddle Buy | VIX > 20 | 9:20 AM | 1:00 PM | 8.7% | -₹1 Cr |
| S7 | First Candle Direction | Every day | 9:20 AM | 11:00 AM | 34.1% | -₹1.07 Cr |
| S8 | 10AM Confirmed Trend | Move > 0.2% | 10:00 AM | 12:30 PM | 29.7% | -₹87.1 L |
| S9 | 11:30AM Reversal Buy | Move > 0.5% | 11:30 AM | 1:30 PM | 25.8% | -₹59.7 L |
| S10 | Power Hour Follow | Every day | 2:15 PM | 3:20 PM | 27.7% | -₹1.66 Cr |
| S11 | 1.5hr Range Breakout | Every day | 10:45 AM | 1:15 PM | 0.0% | -₹38.9 L |
| S12 | Morning Surge Scalp | Open range > 0.4% | 9:45 AM | 11:30 AM | 17.9% | -₹2.82 Cr |

---

## Key Findings

### 1. Selling options beats buying options — consistently over 10 years
All 4 sell strategies were profitable. All 11 buy strategies lost money. This is consistent with the well-known fact that ~80% of options expire worthless, making sellers structurally advantaged.

### 2. Best Entry Time (Optimizer Result)
The Entry Time Optimizer tested a simple trend-follow strategy at every 15-minute slot from 9:15 AM to 2:30 PM. Results:
- **Best time: 10:00 AM** with only **29.7% win rate** and negative average P&L
- No entry time produces a profitable buy strategy
- Confirms: the problem is buying options, not the entry time

### 3. Best Overall Strategy: S13 — Daily Straddle Sell with Stop
- Trades every single day
- Sell ATM straddle at 9:30 AM
- Check at 12:00 PM: if Nifty moved > 0.6% from ATM strike, exit early
- Otherwise hold to 2:30 PM
- **₹4.29 Crore profit over 10 years on 10 lots**

### 4. Safest Strategy: S14 — Iron Condor Sell
- Defined maximum loss (hedge limits downside)
- No stop-loss needed
- 4 legs: sell 0.5% OTM call + put, buy 1.5% OTM call + put as hedge
- Lower profit than naked straddle but far lower risk

---

## How the Backtester Works

### Option Pricing
Options are priced using the **Black-Scholes model** with:
- Underlying: Nifty 50 spot price from 5-minute intraday data
- Implied Volatility: India VIX / 100 (from daily data)
- Risk-Free Rate: 6.5% per year
- Strike: ATM = nearest 50-point multiple to spot price
- Expiry: This week's Thursday (fixed in v3)

### Trade Simulation
```
For a BUY trade:
  P&L = Exit Price × (1 - 0.5% slippage) - Entry Price × (1 + 0.5% slippage)
  Net P&L = P&L per unit × lots × 65 units - ₹20 transaction cost

For a SELL trade:
  P&L = Entry Price × (1 - 0.5% slippage) - Exit Price × (1 + 0.5% slippage)
  Net P&L = P&L per unit × lots × 65 units - ₹20 transaction cost
```

### Assumptions & Limitations
- Black-Scholes pricing is a simplification — real option prices include skew, liquidity premium, and bid-ask spread
- Slippage is fixed at 0.5% — real slippage may be higher for large orders
- No margin interest cost is modeled for sell strategies
- Results are theoretical — past performance does not guarantee future results

---

## Data Sources

| Dataset | Source | Period | Frequency |
|---|---|---|---|
| NIFTY50_INTRADAY_5m_10YEARS.csv | NSE historical data | 2016–2023 | 5-minute |
| NIFTY50_5min_angelone.csv | Angel One API | 2023–2026 | 5-minute |
| NIFTY50_MASTER_DAILY.csv | NSE + VIX data | 2016–2026 | Daily |

---

## Dashboard Features

Open `strategy_backtest_v3_dashboard.html` in Chrome or Edge:

- **3 Capital Tier Tabs** — Small / Medium / Large
- **Best Strategy Banner** — auto-selected champion per tier
- **KPI Pills** — Capital, Lots, CAGR, Sharpe, Win Rate, Max Drawdown
- **Equity Curves** — cumulative P&L line chart for all 15 strategies (10 years)
- **Metrics Table** — sortable comparison of all strategies
- **Monthly Heatmap** — green/red calendar view, click any strategy to switch
- **Entry Time Optimizer Tab** — bar charts showing win rate + avg P&L by entry time

---

## Risk Warning

> **This tool is for research and education only.**
>
> Selling options carries **unlimited risk** if the market makes an extreme move. A single event day (budget, RBI rate decision, global crash) can erase weeks of accumulated premium income.
>
> **Before trading real money:**
> - Paper trade for at least 3 months
> - Start with 1 lot maximum
> - Set hard stop-losses on your broker platform
> - Understand margin requirements (₹1–3 Lakhs per lot for selling)
> - Consult a SEBI-registered financial advisor

---

*Built with Python + Black-Scholes | 286,701 intraday candles | 2,601 trading days | 2016–2026*

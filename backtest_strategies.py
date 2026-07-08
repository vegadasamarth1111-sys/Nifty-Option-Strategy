"""
Nifty 50 Options Strategy Backtester
=====================================
6 Research-Based Strategies | Black-Scholes Synthetic Pricing | 10-Year Backtest
Capital Tiers: Small (<50k) | Medium (<1.5L) | Large (>1.5L)
Lot Size: 65 units
Output: strategy_backtest_dashboard.html
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import time, timedelta
import warnings
import os
import json
import sys

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOT_SIZE   = 65          # Nifty 50 lot size (units per lot)
RISK_FREE  = 0.065       # RBI repo rate approximation
SLIPPAGE   = 0.005       # 0.5% slippage on premium
TXN_COST   = 20          # ₹20 per trade brokerage

CAPITAL_TIERS = {
    "Small  (< ₹50k)":   {"capital": 50_000,   "lots": 3},
    "Medium (< ₹1.5L)": {"capital": 1_50_000,  "lots": 10},
    "Large  (> ₹1.5L)": {"capital": 2_00_000,  "lots": 20},
}

DATA_DIR = r"c:\samarth\projects\nifty 50 option"
OUTPUT_FILE = os.path.join(DATA_DIR, "strategy_backtest_dashboard.html")

# ─────────────────────────────────────────────
# 1. BLACK-SCHOLES PRICER
# ─────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option pricing. Returns 0 if expired."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return max(intrinsic, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0.01)

def get_atm_strike(spot, step=50):
    """Round spot to nearest 50 (Nifty strike step)."""
    return round(spot / step) * step

def next_thursday(dt):
    """Get the next expiry Thursday from a given date."""
    days_ahead = 3 - dt.weekday()  # Thursday = 3
    if days_ahead <= 0:
        days_ahead += 7
    return dt + timedelta(days=days_ahead)

# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    print("? Loading data...")

    # Load 5-min intraday (10 years)
    intraday_path = os.path.join(DATA_DIR, "NIFTY50_INTRADAY_5m_10YEARS.csv")
    angel_path    = os.path.join(DATA_DIR, "NIFTY50_5min_angelone.csv")
    daily_path    = os.path.join(DATA_DIR, "NIFTY50_MASTER_DAILY.csv")

    # Load intraday
    df5 = pd.read_csv(intraday_path)
    df5.columns = df5.columns.str.strip()
    df5.rename(columns={df5.columns[0]: "timestamp"}, inplace=True)
    df5["timestamp"] = pd.to_datetime(df5["timestamp"])
    df5 = df5.dropna(subset=["timestamp", "open", "high", "low", "close"])

    # Load angel one data for more recent coverage
    dfa = pd.read_csv(angel_path)
    dfa.columns = dfa.columns.str.strip()
    dfa.rename(columns={dfa.columns[0]: "timestamp"}, inplace=True)
    dfa["timestamp"] = pd.to_datetime(dfa["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    dfa = dfa.dropna(subset=["timestamp", "open", "high", "low", "close"])

    # Combine, remove duplicates
    df5min = pd.concat([df5, dfa], ignore_index=True)
    df5min = df5min.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df5min["date"] = df5min["timestamp"].dt.date
    df5min["time"] = df5min["timestamp"].dt.time

    # Load daily (for VIX)
    daily = pd.read_csv(daily_path)
    daily.columns = daily.columns.str.strip()
    daily.rename(columns={daily.columns[0]: "date"}, inplace=True)
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily = daily[["date", "open", "close", "vix"]].rename(
        columns={"open": "daily_open", "close": "daily_close", "vix": "vix"}
    )
    daily = daily.dropna(subset=["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # Fill missing VIX with rolling mean
    daily["vix"] = daily["vix"].fillna(method="ffill").fillna(15.0)
    daily["vix_iv"] = daily["vix"] / 100.0  # convert % to decimal

    # Previous day close for gap calculation
    daily["prev_close"] = daily["daily_close"].shift(1)
    daily["gap_pct"]    = (daily["daily_open"] - daily["prev_close"]) / daily["prev_close"]

    # Day-of-week features
    daily["date_dt"]    = pd.to_datetime(daily["date"])
    daily["day_of_week"]= daily["date_dt"].dt.dayofweek  # 0=Mon, 3=Thu
    daily["is_thursday"]= (daily["day_of_week"] == 3).astype(int)
    daily["is_monday"]  = (daily["day_of_week"] == 0).astype(int)

    # 3-day trend
    daily["ret"]          = daily["daily_close"].pct_change()
    daily["trend_3day"]   = (daily["ret"].rolling(3).apply(
        lambda x: 1 if all(v > 0 for v in x) else (-1 if all(v < 0 for v in x) else 0)
    )).fillna(0).astype(int)

    print(f"  ? Intraday: {len(df5min):,} rows | {df5min['date'].min()} ? {df5min['date'].max()}")
    print(f"  ? Daily:    {len(daily):,} rows | VIX available")
    return df5min, daily

# ─────────────────────────────────────────────
# 3. STRATEGY ENGINE
# ─────────────────────────────────────────────
def get_price_at(df5min_day, target_time):
    """Get close price at a specific time from the day's 5-min bars."""
    row = df5min_day[df5min_day["time"] == target_time]
    if len(row) == 0:
        # Try nearest time within 10 mins
        times = pd.to_datetime([str(t) for t in df5min_day["time"].values])
        target = pd.to_datetime(str(target_time))
        diffs = abs(times - target)
        if diffs.min().seconds > 600:
            return None
        return float(df5min_day.iloc[diffs.argmin()]["close"])
    return float(row.iloc[-1]["close"])

def simulate_trade(spot_entry, spot_exit, vix_iv, expiry_date, trade_date,
                   entry_time_frac, exit_time_frac, option_side, action,
                   lots, lot_size=LOT_SIZE, r=RISK_FREE, slippage=SLIPPAGE, txn=TXN_COST):
    """
    Simulate a single option trade using Black-Scholes.
    action: 'buy' or 'sell'
    option_side: 'call' or 'put'
    Returns: P&L in ₹
    """
    K = get_atm_strike(spot_entry)
    sigma = max(vix_iv, 0.05)

    T_entry = max((expiry_date - trade_date).days / 365.0 - entry_time_frac / 365.0, 1 / 365.0)
    T_exit  = max((expiry_date - trade_date).days / 365.0 - exit_time_frac / 365.0, 0.0)

    price_entry = bs_price(spot_entry, K, T_entry, r, sigma, option_side)
    price_exit  = bs_price(spot_exit,  K, T_exit,  r, sigma, option_side)

    if action == "buy":
        price_entry_adj = price_entry * (1 + slippage)
        price_exit_adj  = price_exit  * (1 - slippage)
        pnl_per_unit = price_exit_adj - price_entry_adj
    else:  # sell
        price_entry_adj = price_entry * (1 - slippage)
        price_exit_adj  = price_exit  * (1 + slippage)
        pnl_per_unit = price_entry_adj - price_exit_adj

    total_pnl = pnl_per_unit * lots * lot_size - txn
    return total_pnl, price_entry, price_exit

# ─────────────────────────────────────────────
# 4. RUN ALL STRATEGIES
# ─────────────────────────────────────────────
def run_strategies(df5min, daily, lots):
    print(f"\n? Running strategies ({lots} lots)...")

    results = {f"S{i}": [] for i in range(1, 7)}

    # Group intraday by date for fast lookup
    grouped = {str(d): g for d, g in df5min.groupby("date")}

    for idx, row in daily.iterrows():
        date      = row["date"]
        date_key  = str(date)
        vix_iv    = row["vix_iv"]
        gap_pct   = row["gap_pct"] if pd.notna(row["gap_pct"]) else 0
        is_thu    = row["is_thursday"]
        is_mon    = row["is_monday"]
        trend3    = row["trend_3day"]
        daily_open= row["daily_open"] if pd.notna(row["daily_open"]) else None

        if date_key not in grouped:
            continue

        day_df = grouped[date_key]
        if len(day_df) < 5:
            continue

        # Determine expiry date
        expiry = next_thursday(row["date_dt"]).date()
        T_base = (expiry - date).days / 365.0
        if T_base <= 0:
            T_base = 1 / 365.0

        # Time fractions for intraday (fraction of year each candle advances)
        FRAC_9_15  = 0
        FRAC_9_20  = 5  / (252 * 6.25 * 60)
        FRAC_9_30  = 15 / (252 * 6.25 * 60)
        FRAC_9_45  = 30 / (252 * 6.25 * 60)
        FRAC_10_15 = 60 / (252 * 6.25 * 60)
        FRAC_11_00 = 105/ (252 * 6.25 * 60)
        FRAC_12_00 = 165/ (252 * 6.25 * 60)
        FRAC_12_30 = 195/ (252 * 6.25 * 60)
        FRAC_13_00 = 225/ (252 * 6.25 * 60)
        FRAC_14_30 = 315/ (252 * 6.25 * 60)
        FRAC_15_15 = 360/ (252 * 6.25 * 60)

        # Helper
        def p(t): return get_price_at(day_df, t)

        # ── S1: Opening Range Breakout (ORB-15) ──────────────────────
        try:
            p_915 = p(time(9, 15))
            p_920 = p(time(9, 20))
            p_925 = p(time(9, 25))
            p_930 = p(time(9, 30))
            p_1300= p(time(13, 0))
            if all(v is not None for v in [p_915, p_920, p_925, p_930, p_1300]):
                range_high = max(p_915, p_920, p_925)
                range_low  = min(p_915, p_920, p_925)
                if p_930 > range_high:
                    side = "call"
                elif p_930 < range_low:
                    side = "put"
                else:
                    side = None
                if side:
                    pnl, pe, px = simulate_trade(p_930, p_1300, vix_iv, expiry, date,
                                                 FRAC_9_30, FRAC_13_00, side, "buy", lots)
                    results["S1"].append({"date": date, "pnl": pnl, "side": side,
                                         "entry": pe, "exit": px})
        except Exception:
            pass

        # ── S2: VIX Regime + Gap Signal ──────────────────────────────
        try:
            p_920  = p(time(9, 20))
            p_1230 = p(time(12, 30))
            if p_920 is not None and p_1230 is not None and pd.notna(gap_pct):
                if vix_iv * 100 > 18:
                    side = "put"  if gap_pct > 0.003 else "call" if gap_pct < -0.003 else None
                else:
                    side = "call" if gap_pct > 0.003 else "put"  if gap_pct < -0.003 else None
                if side:
                    pnl, pe, px = simulate_trade(p_920, p_1230, vix_iv, expiry, date,
                                                 FRAC_9_20, FRAC_12_30, side, "buy", lots)
                    results["S2"].append({"date": date, "pnl": pnl, "side": side,
                                         "entry": pe, "exit": px})
        except Exception:
            pass

        # ── S3: Thursday Expiry Theta Crusher (Straddle Sell) ────────
        try:
            if is_thu:
                p_920  = p(time(9, 20))
                p_1430 = p(time(14, 30))
                if p_920 is not None and p_1430 is not None:
                    # Sell straddle = sell call + sell put
                    pnl_c, pe_c, px_c = simulate_trade(p_920, p_1430, vix_iv, expiry, date,
                                                       FRAC_9_20, FRAC_14_30, "call", "sell", lots)
                    pnl_p, pe_p, px_p = simulate_trade(p_920, p_1430, vix_iv, expiry, date,
                                                       FRAC_9_20, FRAC_14_30, "put",  "sell", lots)
                    total_pnl = pnl_c + pnl_p
                    results["S3"].append({"date": date, "pnl": total_pnl, "side": "straddle_sell",
                                         "entry": pe_c + pe_p, "exit": px_c + px_p})
        except Exception:
            pass

        # ── S4: Monday Gap Reversal ───────────────────────────────────
        try:
            if is_mon and pd.notna(gap_pct) and abs(gap_pct) > 0.005:
                p_920  = p(time(9, 20))
                p_1100 = p(time(11, 0))
                if p_920 is not None and p_1100 is not None:
                    side = "put" if gap_pct > 0 else "call"  # fade the gap
                    pnl, pe, px = simulate_trade(p_920, p_1100, vix_iv, expiry, date,
                                                 FRAC_9_20, FRAC_11_00, side, "buy", lots)
                    results["S4"].append({"date": date, "pnl": pnl, "side": side,
                                         "entry": pe, "exit": px})
        except Exception:
            pass

        # ── S5: 3-Day Momentum Continuation ──────────────────────────
        try:
            if trend3 != 0:
                p_920  = p(time(9, 20))
                p_1200 = p(time(12, 0))
                if p_920 is not None and p_1200 is not None:
                    side = "call" if trend3 == 1 else "put"
                    pnl, pe, px = simulate_trade(p_920, p_1200, vix_iv, expiry, date,
                                                 FRAC_9_20, FRAC_12_00, side, "buy", lots)
                    results["S5"].append({"date": date, "pnl": pnl, "side": side,
                                         "entry": pe, "exit": px})
        except Exception:
            pass

        # ── S6: High-VIX Straddle Buy ─────────────────────────────────
        try:
            if vix_iv * 100 > 20:
                p_920  = p(time(9, 20))
                p_1300 = p(time(13, 0))
                if p_920 is not None and p_1300 is not None:
                    pnl_c, pe_c, px_c = simulate_trade(p_920, p_1300, vix_iv, expiry, date,
                                                       FRAC_9_20, FRAC_13_00, "call", "buy", lots)
                    pnl_p, pe_p, px_p = simulate_trade(p_920, p_1300, vix_iv, expiry, date,
                                                       FRAC_9_20, FRAC_13_00, "put",  "buy", lots)
                    total_pnl = pnl_c + pnl_p
                    results["S6"].append({"date": date, "pnl": total_pnl, "side": "straddle_buy",
                                         "entry": pe_c + pe_p, "exit": px_c + px_p})
        except Exception:
            pass

    # Convert to dataframes
    dfs = {}
    for k, v in results.items():
        if v:
            df = pd.DataFrame(v)
            df["date"] = pd.to_datetime(df["date"])
            df["month"] = df["date"].dt.to_period("M")
            df["year"]  = df["date"].dt.year
            df["cum_pnl"] = df["pnl"].cumsum()
            dfs[k] = df
            print(f"  {k}: {len(df)} trades | Net P&L: ?{df['pnl'].sum():,.0f}")
        else:
            print(f"  {k}: 0 trades")
    return dfs

# ─────────────────────────────────────────────
# 5. PERFORMANCE METRICS
# ─────────────────────────────────────────────
def compute_metrics(df, capital):
    if df is None or len(df) == 0:
        return {}
    pnl = df["pnl"].values
    cum = np.cumsum(pnl)
    total_return = cum[-1]
    n_years = (df["date"].max() - df["date"].min()).days / 365.25
    cagr = ((capital + total_return) / capital) ** (1 / max(n_years, 0.1)) - 1 if n_years > 0 else 0
    daily_pnl = df.groupby(df["date"].dt.date)["pnl"].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0
    # Max drawdown
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    max_dd = drawdown.min()
    wins  = pnl[pnl > 0]
    losses= pnl[pnl < 0]
    win_rate = len(wins) / len(pnl) * 100
    avg_win  = wins.mean()  if len(wins) > 0  else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")
    return {
        "Total P&L": f"₹{total_return:,.0f}",
        "CAGR": f"{cagr*100:.1f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Max Drawdown": f"₹{max_dd:,.0f}",
        "Win Rate": f"{win_rate:.1f}%",
        "Avg Win": f"₹{avg_win:,.0f}",
        "Avg Loss": f"₹{avg_loss:,.0f}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Total Trades": len(pnl),
        "raw_cagr": cagr,
        "raw_sharpe": sharpe,
        "raw_pnl": total_return,
        "raw_maxdd": max_dd,
    }

# ─────────────────────────────────────────────
# 6. GENERATE MONTHLY HEATMAP DATA
# ─────────────────────────────────────────────
def monthly_heatmap(df):
    if df is None or len(df) == 0:
        return {}
    monthly = df.groupby(["year", df["date"].dt.month])["pnl"].sum().reset_index()
    monthly.columns = ["year", "month_num", "pnl"]
    result = {}
    for _, row in monthly.iterrows():
        key = f"{int(row['year'])}-{int(row['month_num']):02d}"
        result[key] = round(row["pnl"], 0)
    return result

# ─────────────────────────────────────────────
# 7. BUILD HTML DASHBOARD
# ─────────────────────────────────────────────
STRATEGY_NAMES = {
    "S1": "ORB-15 Breakout",
    "S2": "VIX + Gap Signal",
    "S3": "Expiry Theta Crusher",
    "S4": "Monday Gap Reversal",
    "S5": "3-Day Momentum",
    "S6": "High-VIX Straddle Buy",
}

STRATEGY_DESC = {
    "S1": "Buy ATM Call/Put when 9:30 AM price breaks the first 15-min high/low range. Exit at 1:00 PM.",
    "S2": "Buy Call/Put based on VIX level + overnight gap direction. Exit 12:30 PM.",
    "S3": "Sell ATM Straddle on expiry Thursdays at 9:20 AM. Collect theta decay. Exit 2:30 PM.",
    "S4": "Fade Monday gaps > 0.5% — buy opposite side. Exit 11:00 AM.",
    "S5": "Follow 3-day price trend — buy Call if 3 green days, Put if 3 red days. Exit 12:00 PM.",
    "S6": "When VIX > 20, buy ATM Straddle expecting a big move in either direction. Exit 1:00 PM.",
}

COLORS = ["#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8", "#ff922b"]

def build_dashboard(all_tier_results, all_tier_metrics, all_tier_heatmaps, capital_tiers):
    print("\n? Building interactive dashboard...")

    # Serialize data for JS
    def serialize(dfs):
        out = {}
        for k, df in dfs.items():
            if df is not None and len(df) > 0:
                out[k] = {
                    "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
                    "cum_pnl": df["cum_pnl"].round(0).tolist(),
                    "pnl": df["pnl"].round(0).tolist(),
                }
        return out

    tier_keys = list(capital_tiers.keys())
    tdata  = {t: serialize(all_tier_results[t]) for t in tier_keys}
    tmets  = {t: all_tier_metrics[t]  for t in tier_keys}
    theats = {t: all_tier_heatmaps[t] for t in tier_keys}

    # Find best strategy per tier
    best_per_tier = {}
    for t in tier_keys:
        best_s, best_score = None, -999999
        for s, m in all_tier_metrics[t].items():
            if m and "raw_cagr" in m:
                score = m["raw_cagr"] * 0.5 + m["raw_sharpe"] * 0.3 - abs(m["raw_maxdd"]) / 1e6 * 0.2
                if score > best_score:
                    best_score = score
                    best_s = s
        best_per_tier[t] = best_s

    months_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # Precompute metric dicts OUTSIDE the f-string to avoid Python f-string
    # misinterpreting {{k:v for ...}} as a set containing a dict comprehension
    tmets_public = {
        t: {s: {k: v for k, v in m.items() if not k.startswith("raw_")}
            for s, m in d.items() if m}
        for t, d in tmets.items()
    }
    tmets_raw = {
        t: {s: {k: v for k, v in m.items() if k.startswith("raw_")}
            for s, m in d.items() if m}
        for t, d in tmets.items()
    }
    capital_tiers_js = {
        t: {"capital": v["capital"], "lots": v["lots"]}
        for t, v in capital_tiers.items()
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nifty 50 Options — Strategy Backtester</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:       #0a0e1a;
    --bg2:      #111827;
    --bg3:      #1a2234;
    --card:     #1e2a3d;
    --border:   #2a3850;
    --accent:   #00d4ff;
    --accent2:  #7c3aed;
    --green:    #10b981;
    --red:      #ef4444;
    --yellow:   #f59e0b;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --gold:     #fbbf24;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }}

  /* ── Header ── */
  .header {{
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1040 50%, #0a0e1a 100%);
    border-bottom: 1px solid var(--border);
    padding: 32px 48px;
    display: flex; align-items: center; justify-content: space-between;
    position: relative; overflow: hidden;
  }}
  .header::before {{
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse 80% 60% at 50% -20%, rgba(0,212,255,0.12) 0%, transparent 70%);
  }}
  .header-left h1 {{
    font-size: 2rem; font-weight:900;
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    letter-spacing:-0.02em;
  }}
  .header-left p {{ color:var(--muted); margin-top:6px; font-size:0.9rem; }}
  .header-badge {{
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(124,58,237,0.15));
    border: 1px solid rgba(0,212,255,0.3);
    border-radius:12px; padding:12px 20px; text-align:center;
  }}
  .header-badge .num {{ font-size:1.8rem; font-weight:900; color:var(--accent); }}
  .header-badge .lbl {{ font-size:0.75rem; color:var(--muted); margin-top:2px; }}

  /* ── Tabs ── */
  .tabs-wrap {{ padding: 24px 48px 0; background: var(--bg2); border-bottom:1px solid var(--border); display:flex; gap:8px; }}
  .tab-btn {{
    padding:10px 24px; border:none; border-radius:8px 8px 0 0; cursor:pointer;
    font-family:inherit; font-size:0.88rem; font-weight:600;
    background:transparent; color:var(--muted); transition:all 0.2s;
  }}
  .tab-btn.active {{ background:var(--card); color:var(--accent); border-bottom:2px solid var(--accent); }}
  .tab-btn:hover:not(.active) {{ color:var(--text); background:var(--bg3); }}

  /* ── Main ── */
  .main {{ padding:32px 48px; max-width:1600px; margin:0 auto; }}

  /* ── Section labels ── */
  .section-label {{ font-size:0.75rem; font-weight:700; letter-spacing:0.1em; color:var(--accent); text-transform:uppercase; margin-bottom:16px; }}

  /* ── Best-strategy banner ── */
  .best-banner {{
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(0,212,255,0.08));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius:16px; padding:24px 32px; margin-bottom:32px;
    display:flex; align-items:center; gap:24px;
  }}
  .best-trophy {{ font-size:3rem; }}
  .best-title {{ font-size:0.8rem; color:var(--muted); font-weight:600; letter-spacing:0.05em; text-transform:uppercase; }}
  .best-name  {{ font-size:1.6rem; font-weight:900; color:var(--green); margin:4px 0; }}
  .best-desc  {{ font-size:0.85rem; color:var(--muted); }}
  .best-stats {{ margin-left:auto; display:flex; gap:24px; }}
  .best-stat  {{ text-align:center; }}
  .best-stat .val {{ font-size:1.3rem; font-weight:800; color:var(--accent); }}
  .best-stat .key {{ font-size:0.72rem; color:var(--muted); margin-top:2px; }}

  /* ── Grid ── */
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:32px; }}
  .grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin-bottom:32px; }}
  @media(max-width:900px) {{ .grid-2,.grid-3 {{ grid-template-columns:1fr; }} }}

  /* ── Card ── */
  .card {{
    background:var(--card); border:1px solid var(--border); border-radius:16px;
    padding:24px; position:relative; overflow:hidden;
  }}
  .card::after {{
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }}
  .card-title {{ font-size:0.88rem; font-weight:700; color:var(--muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:0.05em; }}
  .card-subtitle {{ font-size:0.78rem; color:var(--muted); opacity:0.7; margin-bottom:20px; }}
  .chart-wrap {{ position:relative; height:280px; }}

  /* ── Metrics table ── */
  .metrics-table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
  .metrics-table th {{ color:var(--muted); font-weight:600; padding:8px 12px; text-align:left; border-bottom:1px solid var(--border); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; }}
  .metrics-table td {{ padding:10px 12px; border-bottom:1px solid rgba(42,56,80,0.4); }}
  .metrics-table tr:last-child td {{ border-bottom:none; }}
  .metrics-table tr:hover td {{ background:rgba(0,212,255,0.04); }}
  .strat-name {{ font-weight:700; color:var(--text); }}
  .pos {{ color:var(--green); font-weight:600; }}
  .neg {{ color:var(--red);   font-weight:600; }}
  .neu {{ color:var(--yellow); font-weight:600; }}
  .best-row td {{ background:rgba(16,185,129,0.07) !important; }}
  .best-row .strat-name {{ color:var(--green); }}
  .crown {{ display:inline-block; margin-right:4px; }}

  /* ── Heatmap ── */
  .heatmap-grid {{ display:grid; grid-template-columns: 56px repeat(12,1fr); gap:3px; font-size:0.72rem; }}
  .hm-header {{ color:var(--muted); text-align:center; padding:4px 0; font-weight:600; }}
  .hm-year   {{ color:var(--muted); font-weight:700; display:flex; align-items:center; padding-right:6px; }}
  .hm-cell   {{ border-radius:5px; text-align:center; padding:6px 2px; font-weight:600; cursor:default; transition:transform 0.15s; }}
  .hm-cell:hover {{ transform:scale(1.1); z-index:10; position:relative; }}
  .hm-empty  {{ background:var(--bg3); }}

  /* ── KPI pills ── */
  .kpi-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
  .kpi {{
    background:var(--bg3); border:1px solid var(--border); border-radius:12px;
    padding:14px 20px; flex:1; min-width:130px;
  }}
  .kpi .label {{ font-size:0.72rem; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }}
  .kpi .value {{ font-size:1.3rem; font-weight:800; margin-top:4px; }}

  /* ── Strategy card pills ── */
  .strat-cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:32px; }}
  .strat-card {{ background:var(--bg3); border:1px solid var(--border); border-radius:12px; padding:16px 18px; cursor:pointer; transition:all 0.2s; }}
  .strat-card:hover {{ border-color:var(--accent); background:var(--card); }}
  .strat-card.active {{ border-color:var(--accent); background:var(--card); box-shadow:0 0 0 1px rgba(0,212,255,0.2); }}
  .strat-card .sc-num {{ font-size:0.72rem; color:var(--muted); font-weight:700; letter-spacing:0.05em; }}
  .strat-card .sc-name {{ font-weight:700; font-size:0.9rem; margin:4px 0; }}
  .strat-card .sc-pnl {{ font-size:1rem; font-weight:800; }}

  /* ── Tabs content ── */
  .tab-content {{ display:none; }}
  .tab-content.active {{ display:block; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width:6px; height:6px; }}
  ::-webkit-scrollbar-track {{ background:var(--bg2); }}
  ::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:3px; }}

  /* ── Responsive ── */
  @media(max-width:1100px) {{
    .header {{ padding:20px 24px; flex-direction:column; gap:16px; align-items:flex-start; }}
    .tabs-wrap {{ padding:16px 24px 0; }}
    .main {{ padding:20px 24px; }}
    .strat-cards {{ grid-template-columns:repeat(2,1fr); }}
    .best-stats {{ display:none; }}
  }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-left">
    <h1>⚡ Nifty 50 Options Backtester</h1>
    <p>6 Research-Based Strategies · Black-Scholes Pricing · 10-Year Analysis (2016–2026) · Lot Size 65 units</p>
  </div>
  <div style="display:flex;gap:16px;">
    <div class="header-badge"><div class="num">6</div><div class="lbl">Strategies</div></div>
    <div class="header-badge"><div class="num">3</div><div class="lbl">Capital Tiers</div></div>
    <div class="header-badge"><div class="num">10Y</div><div class="lbl">Backtest</div></div>
  </div>
</div>

<!-- TIER TABS -->
<div class="tabs-wrap" id="tierTabs">
  {" ".join(f'<button class="tab-btn {"active" if i==0 else ""}" onclick="switchTier({i})" id="tierBtn{i}">{t}</button>' for i,t in enumerate(tier_keys))}
</div>

<!-- STRATEGY TABS inside main -->
<div class="main">
  {chr(10).join(f'<div class="tab-content {"active" if i==0 else ""}" id="tier{i}">{{}}</div>' for i,t in enumerate(tier_keys))}
</div>

<script>
// ── DATA ──────────────────────────────────────
const TIER_KEYS  = {json.dumps(tier_keys)};
const STRAT_NAMES= {json.dumps(STRATEGY_NAMES)};
const STRAT_DESC = {json.dumps(STRATEGY_DESC)};
const COLORS     = {json.dumps(COLORS)};
const TDATA      = {json.dumps(tdata)};
const TMETS      = {json.dumps(tmets_public)};
const TMETS_RAW  = {json.dumps(tmets_raw)};
const THEATS     = {json.dumps(theats)};
const BEST_TIER  = {json.dumps(best_per_tier)};
const LOT_SIZE   = {LOT_SIZE};
const CAPITAL_TIERS = {json.dumps(capital_tiers_js)};
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

// ── Utility ───────────────────────────────────
function pnlColor(v) {{ return v >= 0 ? '#10b981' : '#ef4444'; }}
function pnlClass(v) {{ return v >= 0 ? 'pos' : 'neg'; }}
function fmt(v) {{
  if (typeof v !== 'number') return v;
  return '₹' + v.toLocaleString('en-IN', {{maximumFractionDigits:0}});
}}

// ── Charts registry (to destroy before redraw) ─
const chartRegistry = {{}};
function destroyChart(id) {{
  if (chartRegistry[id]) {{ chartRegistry[id].destroy(); delete chartRegistry[id]; }}
}}

// ── Equity Chart ─────────────────────────────
function drawEquity(tierKey, canvasId) {{
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const data = TDATA[tierKey];
  const datasets = Object.entries(data).map(([s, sd], i) => ({{
    label: STRAT_NAMES[s] || s,
    data: sd.dates.map((d,j) => ({{x: d, y: sd.cum_pnl[j]}})),
    borderColor: COLORS[i % COLORS.length],
    backgroundColor: 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.3,
  }}));
  chartRegistry[canvasId] = new Chart(ctx, {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      interaction: {{ mode:'index', intersect:false }},
      plugins: {{
        legend: {{ labels: {{ color:'#94a3b8', font:{{ size:11 }}, boxWidth:12 }} }},
        tooltip: {{
          backgroundColor:'#1e2a3d', borderColor:'#2a3850', borderWidth:1,
          callbacks: {{ label: ctx => `  ${{ctx.dataset.label}}: ₹${{ctx.parsed.y.toLocaleString('en-IN')}}` }}
        }}
      }},
      scales: {{
        x: {{ type:'time', time:{{unit:'month'}}, grid:{{color:'#1a2234'}}, ticks:{{color:'#64748b', maxTicksLimit:20}} }},
        y: {{
          grid:{{color:'#1a2234'}}, ticks:{{color:'#64748b', callback: v => '₹'+v.toLocaleString('en-IN')}},
          title:{{display:true, text:'Cumulative P&L (₹)', color:'#64748b', font:{{size:11}}}}
        }}
      }}
    }}
  }});
}}

// ── Heatmap ───────────────────────────────────
function drawHeatmap(tierKey, stratKey, containerId) {{
  const container = document.getElementById(containerId);
  if (!container) return;
  const data = THEATS[tierKey][stratKey] || {{}};
  const keys = Object.keys(data);
  if (keys.length === 0) {{ container.innerHTML = '<p style="color:#64748b;text-align:center;padding:32px;">No trades for this strategy.</p>'; return; }}
  const years = [...new Set(keys.map(k => parseInt(k.split('-')[0])))].sort();
  const maxAbs = Math.max(...Object.values(data).map(Math.abs), 1);

  let html = '<div class="heatmap-grid">';
  html += '<div class="hm-header"></div>';
  MONTHS.forEach(m => html += `<div class="hm-header">${{m}}</div>`);
  years.forEach(yr => {{
    html += `<div class="hm-year">${{yr}}</div>`;
    for (let mo = 1; mo <= 12; mo++) {{
      const key = `${{yr}}-${{String(mo).padStart(2,'0')}}`;
      if (data[key] !== undefined) {{
        const v = data[key];
        const ratio = v / maxAbs;
        let bg, txt;
        if (v >= 0) {{
          const g = Math.round(100 + ratio * 105);
          bg = `rgba(16,${{g}},129,${{0.2 + ratio * 0.6}})`;
          txt = '#10b981';
        }} else {{
          const r = Math.round(180 + Math.abs(ratio) * 75);
          bg = `rgba(${{r}},60,60,${{0.2 + Math.abs(ratio) * 0.6}})`;
          txt = '#ef4444';
        }}
        html += `<div class="hm-cell" style="background:${{bg}};color:${{txt}};" title="${{key}}: ₹${{v.toLocaleString('en-IN')}}">${{v >= 0 ? '+' : ''}}${{Math.round(v/1000)}}k</div>`;
      }} else {{
        html += '<div class="hm-cell hm-empty">—</div>';
      }}
    }}
  }});
  html += '</div>';
  container.innerHTML = html;
}}

// ── Metrics Table ─────────────────────────────
function buildMetricsTable(tierKey, bestS) {{
  const strats = Object.keys(STRAT_NAMES);
  const mets   = TMETS[tierKey]  || {{}};
  const metsR  = TMETS_RAW[tierKey] || {{}};
  let html = `
  <table class="metrics-table">
    <thead><tr>
      <th>Strategy</th><th>Total P&L</th><th>CAGR</th>
      <th>Sharpe</th><th>Max DD</th><th>Win %</th><th>Profit Factor</th><th>Trades</th>
    </tr></thead><tbody>`;
  strats.forEach(s => {{
    const m = mets[s];
    if (!m) return;
    const isBest = s === bestS;
    const pnlNum = parseFloat((m['Total P&L'] || '0').replace(/[₹,]/g,''));
    html += `<tr class="${{isBest ? 'best-row' : ''}}">
      <td class="strat-name">${{isBest ? '<span class="crown">👑</span>' : ''}}<span style="color:${{COLORS[parseInt(s[1])-1]}}">${{s}}</span> ${{STRAT_NAMES[s]}}</td>
      <td class="${{pnlClass(pnlNum)}}">${{m['Total P&L'] || '—'}}</td>
      <td>${{m['CAGR'] || '—'}}</td>
      <td>${{m['Sharpe'] || '—'}}</td>
      <td class="neg">${{m['Max Drawdown'] || '—'}}</td>
      <td>${{m['Win Rate'] || '—'}}</td>
      <td class="${{parseFloat(m['Profit Factor']) >= 1 ? 'pos' : 'neg'}}">${{m['Profit Factor'] || '—'}}</td>
      <td style="color:#64748b">${{m['Total Trades'] || 0}}</td>
    </tr>`;
  }});
  html += '</tbody></table>';
  return html;
}}

// ── KPI Pills ─────────────────────────────────
function buildKPIs(tierKey, bestS) {{
  const m  = TMETS[tierKey][bestS]  || {{}};
  const ct = CAPITAL_TIERS[tierKey] || {{}};
  const pnlNum = parseFloat((m['Total P&L'] || '0').replace(/[₹,]/g,''));
  return `
  <div class="kpi-row">
    <div class="kpi"><div class="label">Capital</div><div class="value" style="color:#00d4ff">₹${{ct.capital?.toLocaleString('en-IN')}}</div></div>
    <div class="kpi"><div class="label">Lots</div><div class="value" style="color:#7c3aed">${{ct.lots}} lots</div></div>
    <div class="kpi"><div class="label">Total P&L</div><div class="value" style="color:${{pnlColor(pnlNum)}}">${{m['Total P&L']||'—'}}</div></div>
    <div class="kpi"><div class="label">CAGR</div><div class="value" style="color:#10b981">${{m['CAGR']||'—'}}</div></div>
    <div class="kpi"><div class="label">Sharpe</div><div class="value" style="color:#f59e0b">${{m['Sharpe']||'—'}}</div></div>
    <div class="kpi"><div class="label">Max Drawdown</div><div class="value" style="color:#ef4444">${{m['Max Drawdown']||'—'}}</div></div>
  </div>`;
}}

// ── Strategy mini-cards ────────────────────────
function buildStratCards(tierKey, bestS, activeSIdx) {{
  const strats = Object.keys(STRAT_NAMES);
  let html = '<div class="strat-cards">';
  strats.forEach((s, i) => {{
    const mets = TMETS[tierKey][s];
    if (!mets) return;
    const pnlNum = parseFloat((mets['Total P&L']||'0').replace(/[₹,]/g,''));
    html += `<div class="strat-card ${{i === activeSIdx ? 'active' : ''}}" onclick="switchStrat(currentTier, ${{i}})" id="sc-${{currentTier}}-${{i}}">
      <div class="sc-num">${{s}} ${{s === bestS ? '👑' : ''}}</div>
      <div class="sc-name" style="color:${{COLORS[i]}}">${{STRAT_NAMES[s]}}</div>
      <div class="sc-pnl" style="color:${{pnlColor(pnlNum)}}">${{mets['Total P&L']||'—'}}</div>
    </div>`;
  }});
  html += '</div>';
  return html;
}}

// ── Best banner ───────────────────────────────
function buildBestBanner(tierKey) {{
  const bestS = BEST_TIER[tierKey];
  if (!bestS) return '';
  const m = TMETS[tierKey][bestS] || {{}};
  const pnlNum = parseFloat((m['Total P&L']||'0').replace(/[₹,]/g,''));
  return `
  <div class="best-banner">
    <div class="best-trophy">🏆</div>
    <div>
      <div class="best-title">Best Strategy for this Tier</div>
      <div class="best-name">${{STRAT_NAMES[bestS]}}</div>
      <div class="best-desc">${{STRAT_DESC[bestS]}}</div>
    </div>
    <div class="best-stats">
      <div class="best-stat"><div class="val">${{m['CAGR']||'—'}}</div><div class="key">CAGR</div></div>
      <div class="best-stat"><div class="val">${{m['Sharpe']||'—'}}</div><div class="key">Sharpe</div></div>
      <div class="best-stat"><div class="val">${{m['Win Rate']||'—'}}</div><div class="key">Win Rate</div></div>
      <div class="best-stat"><div class="val" style="color:${{pnlColor(pnlNum)}}">${{m['Total P&L']||'—'}}</div><div class="key">Total P&L</div></div>
    </div>
  </div>`;
}}

// ── Render a tier ─────────────────────────────
let currentTier = 0;
let currentStrat = {{}};  // tierIndex -> stratIndex

function renderTier(tierIdx) {{
  const tierKey = TIER_KEYS[tierIdx];
  const bestS   = BEST_TIER[tierKey];
  const strats  = Object.keys(STRAT_NAMES).filter(s => TMETS[tierKey] && TMETS[tierKey][s]);
  const sIdx    = currentStrat[tierIdx] || 0;
  const activeS = strats[sIdx] || strats[0];

  const container = document.getElementById(`tier${{tierIdx}}`);
  if (!container) return;

  container.innerHTML = `
    ${{buildBestBanner(tierKey)}}
    ${{buildKPIs(tierKey, bestS)}}

    <p class="section-label">📈 Cumulative Equity Curves — All Strategies</p>
    <div class="card" style="margin-bottom:32px">
      <div class="card-title">All Strategies Comparison</div>
      <div class="card-subtitle">Cumulative P&L over 10 years for ${{tierKey}}</div>
      <div class="chart-wrap"><canvas id="eq-${{tierIdx}}"></canvas></div>
    </div>

    <p class="section-label">📊 Strategy Performance Table</p>
    <div class="card" style="margin-bottom:32px; overflow-x:auto">
      ${{buildMetricsTable(tierKey, bestS)}}
    </div>

    <p class="section-label">🗓️ Monthly P&L Heatmap — Select Strategy</p>
    ${{buildStratCards(tierKey, bestS, sIdx)}}
    <div class="card" style="margin-bottom:32px">
      <div class="card-title" id="hm-title-${{tierIdx}}">${{STRAT_NAMES[activeS]}} — Monthly P&L</div>
      <div class="card-subtitle" id="hm-sub-${{tierIdx}}">${{STRAT_DESC[activeS]}}</div>
      <div id="hm-${{tierIdx}}" style="margin-top:16px;overflow-x:auto"></div>
    </div>
  `;

  drawEquity(tierKey, `eq-${{tierIdx}}`);
  drawHeatmap(tierKey, activeS, `hm-${{tierIdx}}`);
}}

function switchTier(idx) {{
  document.querySelectorAll('.tab-btn').forEach((b,i) => b.classList.toggle('active', i===idx));
  document.querySelectorAll('.tab-content').forEach((c,i) => c.classList.toggle('active', i===idx));
  currentTier = idx;
  renderTier(idx);
}}

function switchStrat(tierIdx, sIdx) {{
  currentStrat[tierIdx] = sIdx;
  const tierKey = TIER_KEYS[tierIdx];
  const strats  = Object.keys(STRAT_NAMES).filter(s => TMETS[tierKey] && TMETS[tierKey][s]);
  const activeS = strats[sIdx] || strats[0];
  // Update card active state
  document.querySelectorAll(`[id^="sc-${{tierIdx}}-"]`).forEach((el,i) => el.classList.toggle('active', i===sIdx));
  // Redraw heatmap
  drawHeatmap(tierKey, activeS, `hm-${{tierIdx}}`);
  const titleEl = document.getElementById(`hm-title-${{tierIdx}}`);
  if (titleEl) titleEl.textContent = `${{STRAT_NAMES[activeS]}} \u2014 Monthly P&L`;
  const subEl = document.getElementById(`hm-sub-${{tierIdx}}`);
  if (subEl) subEl.textContent = STRAT_DESC[activeS] || '';
}}

// ── Init ──────────────────────────────────────
window.addEventListener('load', () => {{
  // Init all tiers (render only active first for speed)
  renderTier(0);
  // Pre-render others lazily
  TIER_KEYS.forEach((_, i) => {{ if (i > 0) {{ setTimeout(() => renderTier(i), 200 * i); }} }});
}});
</script>
</body>
</html>"""
    return html

# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Nifty 50 Options Strategy Backtester")
    print("=" * 60)

    df5min, daily = load_data()

    all_tier_results  = {}
    all_tier_metrics  = {}
    all_tier_heatmaps = {}

    for tier_name, tier_cfg in CAPITAL_TIERS.items():
        lots = tier_cfg["lots"]
        capital = tier_cfg["capital"]
        print(f"\n{'?'*50}")
        print(f"  Tier: {tier_name} | Capital: ?{capital:,} | Lots: {lots}")
        print(f"{'?'*50}")

        dfs = run_strategies(df5min, daily, lots)
        metrics = {}
        heatmaps = {}

        for s, df in dfs.items():
            metrics[s]  = compute_metrics(df, capital)
            heatmaps[s] = monthly_heatmap(df)

        all_tier_results[tier_name]  = dfs
        all_tier_metrics[tier_name]  = metrics
        all_tier_heatmaps[tier_name] = heatmaps

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for tier, mets in all_tier_metrics.items():
        print(f"\n  ? {tier}")
        for s, m in mets.items():
            if m:
                print(f"    {s} [{STRATEGY_NAMES[s]}]: P&L={m.get('Total P&L','?')} | CAGR={m.get('CAGR','?')} | Sharpe={m.get('Sharpe','?')} | WinRate={m.get('Win Rate','?')}")

    # Build & save HTML
    html = build_dashboard(all_tier_results, all_tier_metrics, all_tier_heatmaps, CAPITAL_TIERS)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n? Dashboard saved: {OUTPUT_FILE}")
    print("   ? Open in Chrome/Edge for best experience")

if __name__ == "__main__":
    main()

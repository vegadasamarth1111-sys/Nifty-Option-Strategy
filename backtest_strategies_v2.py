"""
Nifty 50 Options Strategy Backtester — v2 (Extended)
======================================================
12 Strategies + Entry Time Optimizer
Daily-use strategies with variable entry times
All Free | Black-Scholes Pricing | 10-Year Backtest
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import time, timedelta, datetime
import warnings, os, json, sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOT_SIZE   = 65
RISK_FREE  = 0.065
SLIPPAGE   = 0.005
TXN_COST   = 20

CAPITAL_TIERS = {
    "Small  (< Rs.50k)":  {"capital": 50_000,   "lots": 3},
    "Medium (< Rs.1.5L)": {"capital": 1_50_000,  "lots": 10},
    "Large  (> Rs.1.5L)": {"capital": 2_00_000,  "lots": 20},
}

DATA_DIR    = r"c:\samarth\projects\nifty 50 option"
OUTPUT_FILE = os.path.join(DATA_DIR, "strategy_backtest_v2_dashboard.html")

# ─────────────────────────────────────────────
# STRATEGY REGISTRY
# ─────────────────────────────────────────────
STRATEGY_NAMES = {
    # Original 6
    "S1": "ORB-15 Breakout",
    "S2": "VIX + Gap Signal",
    "S3": "Expiry Theta Crusher",
    "S4": "Monday Gap Reversal",
    "S5": "3-Day Momentum",
    "S6": "High-VIX Straddle Buy",
    # New 6 (daily-use, optimized entry times)
    "S7":  "First Candle Direction",
    "S8":  "10AM Confirmed Trend",
    "S9":  "11:30AM Reversal",
    "S10": "Power Hour Follow",
    "S11": "1.5hr Range Breakout",
    "S12": "Morning Surge Scalp",
}

STRATEGY_DESC = {
    "S1":  "Buy ATM Call/Put when 9:30 AM price breaks the first 15-min high/low range. Exit 1:00 PM.",
    "S2":  "Buy Call/Put based on VIX level + overnight gap direction. Exit 12:30 PM.",
    "S3":  "Sell ATM Straddle on expiry Thursdays at 9:20 AM. Collect theta decay. Exit 2:30 PM.",
    "S4":  "Fade Monday gaps > 0.5% — buy opposite side. Exit 11:00 AM.",
    "S5":  "Follow 3-day price trend at 9:20 AM. Exit 12:00 PM.",
    "S6":  "When VIX > 20, buy ATM Straddle for big move. Exit 1:00 PM.",
    "S7":  "DAILY: First 5-min candle direction at 9:15 AM sets bias. Buy Call (bullish) or Put (bearish). Entry 9:20 AM, Exit 11:00 AM.",
    "S8":  "DAILY: Wait for 10:00 AM trend confirmation vs open. Enter if price moved > 0.2% in either direction. Exit 12:30 PM.",
    "S9":  "DAILY: At 11:30 AM, fade the morning move if it exceeded 0.5%. Buy opposite side. Exit 1:30 PM.",
    "S10": "DAILY: At 2:15 PM, follow the day's trend. Short-duration power-hour play. Exit 3:20 PM.",
    "S11": "DAILY: If 10:45 AM price breaks above/below the first 1.5-hour range, enter in breakout direction. Exit 1:15 PM.",
    "S12": "DAILY: If 9:15-9:45 AM has volatile range > 0.4%, enter in breakout direction. Exit 11:30 AM.",
}

STRATEGY_DAILY = {s: True for s in ["S7","S8","S9","S10","S11","S12"]}

COLORS = ["#00d4ff","#ff6b6b","#51cf66","#ffd43b","#cc5de8","#ff922b",
          "#20c997","#f06595","#74c0fc","#a9e34b","#e599f7","#ffa94d"]

# ─────────────────────────────────────────────
# 1. BLACK-SCHOLES PRICER
# ─────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, option_type="call"):
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
    return round(spot / step) * step

def next_thursday(dt):
    days_ahead = 3 - dt.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return dt + timedelta(days=days_ahead)

# Time-to-expiry fraction helper
def t_frac(hours_elapsed):
    """Convert intraday hours elapsed to year fraction."""
    return hours_elapsed / (252 * 6.25)

TIME_FRACS = {
    time(9, 15):  t_frac(0),
    time(9, 20):  t_frac(5/60),
    time(9, 30):  t_frac(15/60),
    time(9, 45):  t_frac(30/60),
    time(10, 0):  t_frac(45/60),
    time(10, 15): t_frac(60/60),
    time(10, 30): t_frac(75/60),
    time(10, 45): t_frac(90/60),
    time(11, 0):  t_frac(105/60),
    time(11, 30): t_frac(135/60),
    time(12, 0):  t_frac(165/60),
    time(12, 30): t_frac(195/60),
    time(13, 0):  t_frac(225/60),
    time(13, 15): t_frac(240/60),
    time(13, 30): t_frac(255/60),
    time(14, 0):  t_frac(285/60),
    time(14, 15): t_frac(300/60),
    time(14, 30): t_frac(315/60),
    time(15, 15): t_frac(360/60),
    time(15, 20): t_frac(365/60),
}

# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    print("[*] Loading data...")

    df5 = pd.read_csv(os.path.join(DATA_DIR, "NIFTY50_INTRADAY_5m_10YEARS.csv"))
    df5.columns = df5.columns.str.strip()
    df5.rename(columns={df5.columns[0]: "timestamp"}, inplace=True)
    df5["timestamp"] = pd.to_datetime(df5["timestamp"])
    df5 = df5.dropna(subset=["timestamp","open","high","low","close"])

    dfa = pd.read_csv(os.path.join(DATA_DIR, "NIFTY50_5min_angelone.csv"))
    dfa.columns = dfa.columns.str.strip()
    dfa.rename(columns={dfa.columns[0]: "timestamp"}, inplace=True)
    dfa["timestamp"] = pd.to_datetime(dfa["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    dfa = dfa.dropna(subset=["timestamp","open","high","low","close"])

    df5min = pd.concat([df5, dfa], ignore_index=True)
    df5min = df5min.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df5min["date"] = df5min["timestamp"].dt.date
    df5min["time"] = df5min["timestamp"].dt.time

    daily = pd.read_csv(os.path.join(DATA_DIR, "NIFTY50_MASTER_DAILY.csv"))
    daily.columns = daily.columns.str.strip()
    daily.rename(columns={daily.columns[0]: "date"}, inplace=True)
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily = daily[["date","open","close","vix"]].rename(
        columns={"open":"daily_open","close":"daily_close","vix":"vix"})
    daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    daily["vix"] = daily["vix"].fillna(method="ffill").fillna(15.0)
    daily["vix_iv"] = daily["vix"] / 100.0
    daily["prev_close"] = daily["daily_close"].shift(1)
    daily["gap_pct"]    = (daily["daily_open"] - daily["prev_close"]) / daily["prev_close"]
    daily["date_dt"]    = pd.to_datetime(daily["date"])
    daily["day_of_week"]= daily["date_dt"].dt.dayofweek
    daily["is_thursday"]= (daily["day_of_week"] == 3).astype(int)
    daily["is_monday"]  = (daily["day_of_week"] == 0).astype(int)
    daily["ret"]        = daily["daily_close"].pct_change()
    daily["trend_3day"] = (daily["ret"].rolling(3).apply(
        lambda x: 1 if all(v > 0 for v in x) else (-1 if all(v < 0 for v in x) else 0)
    )).fillna(0).astype(int)

    print(f"  [OK] Intraday: {len(df5min):,} rows | {df5min['date'].min()} to {df5min['date'].max()}")
    print(f"  [OK] Daily:    {len(daily):,} rows | VIX available")
    return df5min, daily

# ─────────────────────────────────────────────
# 3. TRADE SIMULATOR
# ─────────────────────────────────────────────
def simulate_trade(spot_entry, spot_exit, vix_iv, expiry_date, trade_date,
                   t_entry_frac, t_exit_frac, option_side, action, lots,
                   lot_size=LOT_SIZE, r=RISK_FREE, slippage=SLIPPAGE, txn=TXN_COST):
    K     = get_atm_strike(spot_entry)
    sigma = max(vix_iv, 0.05)
    T_base= max((expiry_date - trade_date).days / 365.0, 1/365.0)
    T_e   = max(T_base - t_entry_frac, 1/365.0)
    T_x   = max(T_base - t_exit_frac,  0.0)

    pe = bs_price(spot_entry, K, T_e, r, sigma, option_side)
    px = bs_price(spot_exit,  K, T_x, r, sigma, option_side)

    if action == "buy":
        pnl_unit = px * (1 - slippage) - pe * (1 + slippage)
    else:
        pnl_unit = pe * (1 - slippage) - px * (1 + slippage)

    return pnl_unit * lots * lot_size - txn, pe, px

# ─────────────────────────────────────────────
# 4. HELPER
# ─────────────────────────────────────────────
def price_at(day_df, target_time):
    row = day_df[day_df["time"] == target_time]
    if len(row):
        return float(row.iloc[-1]["close"])
    times = pd.to_datetime([str(t) for t in day_df["time"].values])
    tgt   = pd.to_datetime(str(target_time))
    diffs = abs(times - tgt)
    if diffs.min().seconds > 600:
        return None
    return float(day_df.iloc[diffs.argmin()]["close"])

def tf(t): return TIME_FRACS.get(t, t_frac((datetime.combine(datetime.today(), t) -
                                              datetime.combine(datetime.today(), time(9,15))).seconds / 3600))

# ─────────────────────────────────────────────
# 5. ENTRY-TIME OPTIMIZER
# ─────────────────────────────────────────────
SCAN_TIMES = [
    time(9,15), time(9,30), time(9,45),
    time(10,0), time(10,15), time(10,30), time(10,45),
    time(11,0), time(11,15), time(11,30),
    time(12,0), time(12,30),
    time(13,0), time(13,30),
    time(14,0), time(14,30),
]
HOLD_HOURS = 1.5  # hold for 90 minutes after entry

def run_entry_optimizer(df5min, daily, lots=10):
    """
    For each scan time, run simple trend-follow:
    If price > daily open -> buy Call, else -> buy Put
    Hold for HOLD_HOURS. Record win rate, avg P&L, trade count.
    """
    print("\n[>>] Running Entry Time Optimizer...")
    grouped = {str(d): g for d, g in df5min.groupby("date")}
    results  = {str(t): {"wins":0,"losses":0,"pnl":0.0,"trades":0} for t in SCAN_TIMES}

    for _, row in daily.iterrows():
        date    = row["date"]
        dk      = str(date)
        vix_iv  = row["vix_iv"]
        d_open  = row["daily_open"]
        if dk not in grouped or pd.isna(d_open):
            continue
        dg      = grouped[dk]
        expiry  = next_thursday(row["date_dt"]).date()

        for entry_time in SCAN_TIMES:
            # Calculate exit time (entry + 90 min)
            entry_dt = datetime.combine(date, entry_time)
            exit_dt  = entry_dt + timedelta(hours=HOLD_HOURS)
            exit_time= min(exit_dt.time(), time(15,15))

            pe_spot = price_at(dg, entry_time)
            px_spot = price_at(dg, exit_time)
            if pe_spot is None or px_spot is None:
                continue

            side = "call" if pe_spot >= d_open else "put"
            try:
                pnl, _, _ = simulate_trade(
                    pe_spot, px_spot, vix_iv, expiry, date,
                    tf(entry_time), tf(exit_time), side, "buy", lots
                )
                r = results[str(entry_time)]
                r["trades"] += 1
                r["pnl"]    += pnl
                if pnl > 0: r["wins"] += 1
                else:       r["losses"] += 1
            except Exception:
                pass

    optimizer_output = []
    for t in SCAN_TIMES:
        r = results[str(t)]
        n = r["trades"]
        if n > 0:
            win_rate = r["wins"] / n * 100
            avg_pnl  = r["pnl"] / n
            optimizer_output.append({
                "time":     t.strftime("%H:%M"),
                "trades":   n,
                "win_rate": round(win_rate, 1),
                "avg_pnl":  round(avg_pnl, 0),
                "total_pnl":round(r["pnl"], 0),
            })
            print(f"  {t.strftime('%H:%M')}: {n:4d} trades | WinRate={win_rate:.1f}% | AvgP&L=Rs.{avg_pnl:,.0f}")

    # Best time
    if optimizer_output:
        best = max(optimizer_output, key=lambda x: x["win_rate"])
        print(f"\n  [BEST] Entry: {best['time']} | WinRate: {best['win_rate']}% | AvgP&L: Rs.{best['avg_pnl']:,.0f}/trade")
    return optimizer_output

# ─────────────────────────────────────────────
# 6. MAIN STRATEGY RUNNER
# ─────────────────────────────────────────────
def run_strategies(df5min, daily, lots):
    print(f"\n[>>] Running strategies ({lots} lots)...")
    results = {k: [] for k in STRATEGY_NAMES}
    grouped = {str(d): g for d, g in df5min.groupby("date")}

    for _, row in daily.iterrows():
        date      = row["date"]
        dk        = str(date)
        vix_iv    = row["vix_iv"]
        gap_pct   = row["gap_pct"] if pd.notna(row["gap_pct"]) else 0
        is_thu    = row["is_thursday"]
        is_mon    = row["is_monday"]
        trend3    = row["trend_3day"]
        d_open    = row["daily_open"] if pd.notna(row["daily_open"]) else None

        if dk not in grouped: continue
        dg = grouped[dk]
        if len(dg) < 5: continue

        expiry = next_thursday(row["date_dt"]).date()
        p = lambda t: price_at(dg, t)

        def trade(pe_spot, px_spot, side, action, t_entry, t_exit):
            return simulate_trade(pe_spot, px_spot, vix_iv, expiry, date,
                                  tf(t_entry), tf(t_exit), side, action, lots)

        def log(strat, pnl, side, pe, px):
            results[strat].append({"date": date, "pnl": pnl, "side": side,
                                    "entry": pe, "exit": px})

        # ── ORIGINAL STRATEGIES ───────────────────────────────────────────

        # S1: ORB-15
        try:
            p915,p920,p925,p930,p1300 = p(time(9,15)),p(time(9,20)),p(time(9,25)),p(time(9,30)),p(time(13,0))
            if all(v is not None for v in [p915,p920,p925,p930,p1300]):
                rh = max(p915,p920,p925); rl = min(p915,p920,p925)
                side = "call" if p930 > rh else "put" if p930 < rl else None
                if side:
                    pnl,pe,px = trade(p930,p1300,side,"buy",time(9,30),time(13,0))
                    log("S1",pnl,side,pe,px)
        except: pass

        # S2: VIX + Gap
        try:
            p920,p1230 = p(time(9,20)),p(time(12,30))
            if p920 and p1230 and pd.notna(gap_pct):
                if vix_iv*100>18: side = "put" if gap_pct>0.003 else "call" if gap_pct<-0.003 else None
                else:             side = "call" if gap_pct>0.003 else "put"  if gap_pct<-0.003 else None
                if side:
                    pnl,pe,px = trade(p920,p1230,side,"buy",time(9,20),time(12,30))
                    log("S2",pnl,side,pe,px)
        except: pass

        # S3: Expiry Theta Crusher
        try:
            if is_thu:
                p920,p1430 = p(time(9,20)),p(time(14,30))
                if p920 and p1430:
                    pnl_c,pe_c,_ = trade(p920,p1430,"call","sell",time(9,20),time(14,30))
                    pnl_p,pe_p,_ = trade(p920,p1430,"put", "sell",time(9,20),time(14,30))
                    log("S3",pnl_c+pnl_p,"straddle_sell",pe_c+pe_p,0)
        except: pass

        # S4: Monday Gap Reversal
        try:
            if is_mon and abs(gap_pct)>0.005:
                p920,p1100 = p(time(9,20)),p(time(11,0))
                if p920 and p1100:
                    side = "put" if gap_pct>0 else "call"
                    pnl,pe,px = trade(p920,p1100,side,"buy",time(9,20),time(11,0))
                    log("S4",pnl,side,pe,px)
        except: pass

        # S5: 3-Day Momentum
        try:
            if trend3!=0:
                p920,p1200 = p(time(9,20)),p(time(12,0))
                if p920 and p1200:
                    side = "call" if trend3==1 else "put"
                    pnl,pe,px = trade(p920,p1200,side,"buy",time(9,20),time(12,0))
                    log("S5",pnl,side,pe,px)
        except: pass

        # S6: High-VIX Straddle Buy
        try:
            if vix_iv*100>20:
                p920,p1300 = p(time(9,20)),p(time(13,0))
                if p920 and p1300:
                    pnl_c,pe_c,_ = trade(p920,p1300,"call","buy",time(9,20),time(13,0))
                    pnl_p,pe_p,_ = trade(p920,p1300,"put", "buy",time(9,20),time(13,0))
                    log("S6",pnl_c+pnl_p,"straddle_buy",pe_c+pe_p,0)
        except: pass

        # ── NEW DAILY STRATEGIES ─────────────────────────────────────────

        # S7: First Candle Direction (9:20 AM → 11:00 AM) — DAILY
        # No filter. First 5-min candle at 9:15 AM sets direction.
        try:
            p915,p920,p1100 = p(time(9,15)),p(time(9,20)),p(time(11,0))
            if p915 and p920 and p1100:
                # First candle: open is first tick; we compare 9:15 open vs 9:15 close (p920 ≈ second tick)
                first_open  = float(dg[dg["time"]==time(9,15)].iloc[0]["open"]) if len(dg[dg["time"]==time(9,15)])>0 else p915
                side = "call" if p915 >= first_open else "put"
                pnl,pe,px = trade(p920,p1100,side,"buy",time(9,20),time(11,0))
                log("S7",pnl,side,pe,px)
        except: pass

        # S8: 10 AM Confirmed Trend (10:00 AM → 12:30 PM) — DAILY with signal filter
        # Wait for opening noise to settle. Enter only if price moved >0.2% from open.
        try:
            p1000,p1230 = p(time(10,0)),p(time(12,30))
            if p1000 and p1230 and d_open:
                move = (p1000 - d_open) / d_open
                if move > 0.002:  side = "call"
                elif move < -0.002: side = "put"
                else: side = None
                if side:
                    pnl,pe,px = trade(p1000,p1230,side,"buy",time(10,0),time(12,30))
                    log("S8",pnl,side,pe,px)
        except: pass

        # S9: 11:30 AM Mean Reversion (11:30 AM → 1:30 PM) — DAILY with signal filter
        # Fade the morning move if > 0.5%. Mean reversion is strong in Nifty.
        try:
            p1130,p1330 = p(time(11,30)),p(time(13,30))
            if p1130 and p1330 and d_open:
                move = (p1130 - d_open) / d_open
                if move > 0.005:    side = "put"   # fade the rally
                elif move < -0.005: side = "call"  # fade the selloff
                else: side = None
                if side:
                    pnl,pe,px = trade(p1130,p1330,side,"buy",time(11,30),time(13,30))
                    log("S9",pnl,side,pe,px)
        except: pass

        # S10: Power Hour (2:15 PM → 3:20 PM) — DAILY
        # Follow the day's established trend in final trading hour.
        # Gamma is high, small moves give big % gains.
        try:
            p1415,p1520 = p(time(14,15)),p(time(15,20))
            if p1415 and p1520 and d_open:
                move = (p1415 - d_open) / d_open
                if move > 0.001:    side = "call"
                elif move < -0.001: side = "put"
                else: side = None
                if side:
                    pnl,pe,px = trade(p1415,p1520,side,"buy",time(14,15),time(15,20))
                    log("S10",pnl,side,pe,px)
        except: pass

        # S11: 1.5-Hour Range Breakout (10:45 AM → 1:15 PM) — DAILY
        # Calculate first 1.5-hour range. Enter on confirmed breakout of that range.
        # More reliable than ORB-15 — range is larger so false breakouts are fewer.
        try:
            range_bars = dg[dg["time"] <= time(10,45)]
            p1045 = p(time(10,45))
            p1315 = p(time(13,15))
            if len(range_bars)>=5 and p1045 and p1315:
                rh = float(range_bars["high"].max())
                rl = float(range_bars["low"].min())
                rng = rh - rl
                if p1045 > rh - 0.1*rng:  side = "call"  # breaking out up
                elif p1045 < rl + 0.1*rng: side = "put"  # breaking out down
                else: side = None
                if side:
                    pnl,pe,px = trade(p1045,p1315,side,"buy",time(10,45),time(13,15))
                    log("S11",pnl,side,pe,px)
        except: pass

        # S12: Morning Surge Scalp (9:45 AM → 11:30 AM) — DAILY with signal filter
        # If the first 30 minutes have a large range (> 0.4%), enter in the direction
        # of the surge. Strong morning moves tend to continue for 1-2 hours.
        try:
            morning = dg[dg["time"] <= time(9,45)]
            p945  = p(time(9,45))
            p1130 = p(time(11,30))
            if len(morning)>=3 and p945 and p1130 and p915:
                morning_rng = (float(morning["high"].max()) - float(morning["low"].min())) / p915
                if morning_rng > 0.004:  # volatile open
                    side = "call" if p945 >= p915 else "put"
                    pnl,pe,px = trade(p945,p1130,side,"buy",time(9,45),time(11,30))
                    log("S12",pnl,side,pe,px)
        except: pass

    # Convert to DataFrames
    dfs = {}
    for k,v in results.items():
        if v:
            df = pd.DataFrame(v)
            df["date"]    = pd.to_datetime(df["date"])
            df["month"]   = df["date"].dt.to_period("M")
            df["year"]    = df["date"].dt.year
            df["cum_pnl"] = df["pnl"].cumsum()
            dfs[k] = df
            print(f"  {k} [{STRATEGY_NAMES[k]}]: {len(df):4d} trades | Net P&L: Rs.{df['pnl'].sum():>12,.0f}")
        else:
            print(f"  {k}: 0 trades")
    return dfs

# ─────────────────────────────────────────────
# 7. METRICS
# ─────────────────────────────────────────────
def compute_metrics(df, capital):
    if df is None or len(df) == 0: return {}
    pnl = df["pnl"].values
    cum = np.cumsum(pnl)
    total_return = cum[-1]
    n_years = (df["date"].max() - df["date"].min()).days / 365.25
    cagr = ((capital + total_return) / capital) ** (1 / max(n_years, 0.1)) - 1 if n_years > 0 else 0
    daily_pnl = df.groupby(df["date"].dt.date)["pnl"].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0
    running_max = np.maximum.accumulate(cum)
    max_dd = (cum - running_max).min()
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = len(wins) / len(pnl) * 100
    avg_win  = wins.mean()   if len(wins)  > 0 else 0
    avg_loss = losses.mean() if len(losses)> 0 else 0
    pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 99.0
    return {
        "Total P&L":      f"Rs.{total_return:,.0f}",
        "CAGR":           f"{cagr*100:.1f}%",
        "Sharpe":         f"{sharpe:.2f}",
        "Max Drawdown":   f"Rs.{max_dd:,.0f}",
        "Win Rate":       f"{win_rate:.1f}%",
        "Avg Win":        f"Rs.{avg_win:,.0f}",
        "Avg Loss":       f"Rs.{avg_loss:,.0f}",
        "Profit Factor":  f"{pf:.2f}",
        "Total Trades":   len(pnl),
        "raw_cagr":       cagr,
        "raw_sharpe":     sharpe,
        "raw_pnl":        total_return,
        "raw_maxdd":      max_dd,
        "raw_winrate":    win_rate,
    }

def monthly_heatmap(df):
    if df is None or len(df) == 0: return {}
    m = df.groupby(["year", df["date"].dt.month])["pnl"].sum().reset_index()
    m.columns = ["year","month_num","pnl"]
    return {f"{int(r.year)}-{int(r.month_num):02d}": round(r.pnl,0) for _,r in m.iterrows()}

# ─────────────────────────────────────────────
# 8. DASHBOARD BUILDER
# ─────────────────────────────────────────────
def build_dashboard(all_tier_results, all_tier_metrics, all_tier_heatmaps, capital_tiers, optimizer_data):
    print("\n[>>] Building interactive dashboard...")

    tier_keys = list(capital_tiers.keys())

    def serialize(dfs):
        out = {}
        for k, df in dfs.items():
            if df is not None and len(df) > 0:
                out[k] = {
                    "dates":   df["date"].dt.strftime("%Y-%m-%d").tolist(),
                    "cum_pnl": df["cum_pnl"].round(0).tolist(),
                    "pnl":     df["pnl"].round(0).tolist(),
                }
        return out

    tdata  = {t: serialize(all_tier_results[t]) for t in tier_keys}
    tmets  = {t: all_tier_metrics[t]  for t in tier_keys}
    theats = {t: all_tier_heatmaps[t] for t in tier_keys}

    # Precompute metric splits
    tmets_public = {
        t: {s: {k: v for k,v in m.items() if not k.startswith("raw_")}
            for s,m in d.items() if m}
        for t,d in tmets.items()
    }
    tmets_raw = {
        t: {s: {k: v for k,v in m.items() if k.startswith("raw_")}
            for s,m in d.items() if m}
        for t,d in tmets.items()
    }
    capital_tiers_js = {t: {"capital": v["capital"], "lots": v["lots"]} for t,v in capital_tiers.items()}

    # Find best per tier (score: win rate 40% + sharpe 30% + cagr 30%)
    best_per_tier = {}
    for t in tier_keys:
        best_s, best_score = None, -999999
        for s, m in all_tier_metrics[t].items():
            if m and "raw_cagr" in m:
                wr    = m.get("raw_winrate", 0)
                sh    = m.get("raw_sharpe",  0)
                ca    = m.get("raw_cagr",    0)
                score = wr * 0.4 + sh * 10 * 0.3 + ca * 100 * 0.3
                if score > best_score:
                    best_score = score
                    best_s = s
        best_per_tier[t] = best_s

    # Find globally best entry time from optimizer
    best_opt = max(optimizer_data, key=lambda x: x["win_rate"]) if optimizer_data else {}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nifty 50 Options — Strategy Backtester v2</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:#0a0e1a; --bg2:#111827; --bg3:#1a2234; --card:#1e2a3d; --border:#2a3850;
  --accent:#00d4ff; --accent2:#7c3aed; --green:#10b981; --red:#ef4444;
  --yellow:#f59e0b; --text:#e2e8f0; --muted:#64748b; --gold:#fbbf24;
  --orange:#ff922b;
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);}}
.header{{background:linear-gradient(135deg,#0a0e1a,#1a1040,#0a0e1a);border-bottom:1px solid var(--border);padding:28px 48px;display:flex;align-items:center;justify-content:space-between;position:relative;overflow:hidden;}}
.header::before{{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 80% 60% at 50% -20%,rgba(0,212,255,0.12),transparent);}}
.header-left h1{{font-size:1.9rem;font-weight:900;background:linear-gradient(135deg,#00d4ff,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.02em;}}
.header-left p{{color:var(--muted);margin-top:6px;font-size:0.85rem;}}
.header-badges{{display:flex;gap:12px;}}
.badge{{background:linear-gradient(135deg,rgba(0,212,255,0.12),rgba(124,58,237,0.12));border:1px solid rgba(0,212,255,0.25);border-radius:10px;padding:10px 16px;text-align:center;}}
.badge .num{{font-size:1.6rem;font-weight:900;color:var(--accent);}}
.badge .lbl{{font-size:0.7rem;color:var(--muted);margin-top:2px;}}
/* Optimizer Banner */
.opt-banner{{background:linear-gradient(135deg,rgba(255,146,43,0.1),rgba(0,212,255,0.06));border:1px solid rgba(255,146,43,0.3);border-radius:16px;padding:20px 28px;margin:24px 48px 0;display:flex;align-items:center;gap:20px;}}
.opt-icon{{font-size:2.4rem;}}
.opt-title{{font-size:0.72rem;color:var(--muted);font-weight:700;letter-spacing:0.05em;text-transform:uppercase;}}
.opt-main{{font-size:1.4rem;font-weight:900;color:var(--orange);margin:3px 0;}}
.opt-sub{{font-size:0.82rem;color:var(--muted);}}
.opt-stats{{margin-left:auto;display:flex;gap:20px;}}
.opt-stat .val{{font-size:1.2rem;font-weight:800;color:var(--accent);}}
.opt-stat .key{{font-size:0.68rem;color:var(--muted);margin-top:2px;}}
/* Tabs */
.tabs-wrap{{padding:20px 48px 0;background:var(--bg2);border-bottom:1px solid var(--border);display:flex;gap:6px;}}
.tab-btn{{padding:9px 22px;border:none;border-radius:8px 8px 0 0;cursor:pointer;font-family:inherit;font-size:0.85rem;font-weight:600;background:transparent;color:var(--muted);transition:all 0.2s;}}
.tab-btn.active{{background:var(--card);color:var(--accent);border-bottom:2px solid var(--accent);}}
.tab-btn:hover:not(.active){{color:var(--text);background:var(--bg3);}}
.main{{padding:28px 48px;max-width:1700px;margin:0 auto;}}
.tab-content{{display:none;}}
.tab-content.active{{display:block;}}
.section-label{{font-size:0.72rem;font-weight:700;letter-spacing:0.1em;color:var(--accent);text-transform:uppercase;margin-bottom:14px;}}
/* Best Banner */
.best-banner{{background:linear-gradient(135deg,rgba(16,185,129,0.1),rgba(0,212,255,0.06));border:1px solid rgba(16,185,129,0.25);border-radius:14px;padding:20px 28px;margin-bottom:24px;display:flex;align-items:center;gap:20px;}}
.best-trophy{{font-size:2.8rem;}}
.best-title{{font-size:0.72rem;color:var(--muted);font-weight:700;letter-spacing:0.05em;text-transform:uppercase;}}
.best-name{{font-size:1.4rem;font-weight:900;color:var(--green);margin:3px 0;}}
.best-desc{{font-size:0.82rem;color:var(--muted);max-width:480px;}}
.best-stats{{margin-left:auto;display:flex;gap:20px;flex-shrink:0;}}
.best-stat .val{{font-size:1.1rem;font-weight:800;color:var(--accent);text-align:center;}}
.best-stat .key{{font-size:0.68rem;color:var(--muted);text-align:center;margin-top:2px;}}
/* KPI */
.kpi-row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:24px;}}
.kpi{{background:var(--bg3);border:1px solid var(--border);border-radius:12px;padding:14px 18px;flex:1;min-width:120px;}}
.kpi .label{{font-size:0.68rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:0.05em;}}
.kpi .value{{font-size:1.25rem;font-weight:800;margin-top:4px;}}
/* Card */
.card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:22px;position:relative;overflow:hidden;margin-bottom:24px;}}
.card::after{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),var(--accent2));}}
.card-title{{font-size:0.85rem;font-weight:700;color:var(--muted);margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em;}}
.card-subtitle{{font-size:0.76rem;color:var(--muted);opacity:0.7;margin-bottom:18px;}}
.chart-wrap{{position:relative;height:260px;}}
/* Strategy cards */
.strat-cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;}}
.strat-card{{background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:14px 16px;cursor:pointer;transition:all 0.2s;}}
.strat-card:hover{{border-color:var(--accent);}}
.strat-card.active{{border-color:var(--accent);background:var(--card);box-shadow:0 0 0 1px rgba(0,212,255,0.15);}}
.strat-card .sc-num{{font-size:0.68rem;color:var(--muted);font-weight:700;}}
.strat-card .sc-name{{font-weight:700;font-size:0.82rem;margin:3px 0;}}
.strat-card .sc-pnl{{font-size:0.95rem;font-weight:800;}}
.strat-card .sc-wr{{font-size:0.72rem;color:var(--muted);margin-top:2px;}}
.daily-badge{{display:inline-block;background:rgba(16,185,129,0.15);color:var(--green);border:1px solid rgba(16,185,129,0.3);border-radius:4px;font-size:0.58rem;font-weight:700;padding:1px 5px;margin-left:4px;vertical-align:middle;letter-spacing:0.04em;text-transform:uppercase;}}
/* Metrics table */
.metrics-table{{width:100%;border-collapse:collapse;font-size:0.8rem;}}
.metrics-table th{{color:var(--muted);font-weight:600;padding:8px 10px;text-align:left;border-bottom:1px solid var(--border);font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;white-space:nowrap;}}
.metrics-table td{{padding:9px 10px;border-bottom:1px solid rgba(42,56,80,0.4);white-space:nowrap;}}
.metrics-table tr:last-child td{{border-bottom:none;}}
.metrics-table tr:hover td{{background:rgba(0,212,255,0.03);}}
.best-row td{{background:rgba(16,185,129,0.06)!important;}}
.pos{{color:var(--green);font-weight:600;}}
.neg{{color:var(--red);font-weight:600;}}
/* Heatmap */
.heatmap-grid{{display:grid;grid-template-columns:54px repeat(12,1fr);gap:3px;font-size:0.68rem;}}
.hm-header{{color:var(--muted);text-align:center;padding:4px 0;font-weight:600;}}
.hm-year{{color:var(--muted);font-weight:700;display:flex;align-items:center;}}
.hm-cell{{border-radius:4px;text-align:center;padding:5px 2px;font-weight:600;cursor:default;transition:transform 0.15s;}}
.hm-cell:hover{{transform:scale(1.1);z-index:10;position:relative;}}
.hm-empty{{background:var(--bg3);}}
/* Optimizer chart */
.optimizer-wrap{{height:240px;position:relative;}}
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:var(--bg2);}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px;}}
@media(max-width:1100px){{
  .header,.tabs-wrap,.main{{padding-left:20px;padding-right:20px;}}
  .opt-banner{{margin:16px 20px 0;}}
  .strat-cards{{grid-template-columns:repeat(2,1fr);}}
  .best-stats,.opt-stats{{display:none;}}
}}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>Nifty 50 Options Backtester v2</h1>
    <p>12 Strategies (6 Daily-Use) + Entry Time Optimizer &bull; Black-Scholes Pricing &bull; 10-Year Analysis (2016-2026) &bull; Lot Size 65 units</p>
  </div>
  <div class="header-badges">
    <div class="badge"><div class="num">12</div><div class="lbl">Strategies</div></div>
    <div class="badge"><div class="num" style="color:#10b981">6</div><div class="lbl">Daily Use</div></div>
    <div class="badge"><div class="num">3</div><div class="lbl">Capital Tiers</div></div>
    <div class="badge"><div class="num">10Y</div><div class="lbl">Backtest</div></div>
  </div>
</div>

<div class="opt-banner">
  <div class="opt-icon">&#x23F1;</div>
  <div>
    <div class="opt-title">Entry Time Optimizer Result (Trend-Follow, 90-min hold)</div>
    <div class="opt-main">Best Entry: {best_opt.get('time','—')} AM &rarr; Win Rate: {best_opt.get('win_rate','—')}%</div>
    <div class="opt-sub">Avg P&L: Rs.{best_opt.get('avg_pnl',0):,.0f} per trade (10 lots) &bull; {best_opt.get('trades',0)} qualifying days in 10 years</div>
  </div>
  <div class="opt-stats">
    <div class="opt-stat"><div class="val">{best_opt.get('win_rate','—')}%</div><div class="key">Win Rate</div></div>
    <div class="opt-stat"><div class="val">Rs.{best_opt.get('avg_pnl',0):>,.0f}</div><div class="key">Avg P&L/Trade</div></div>
    <div class="opt-stat"><div class="val">90 min</div><div class="key">Hold Time</div></div>
  </div>
</div>

<div class="tabs-wrap" id="tierTabs">
  {" ".join(f'<button class="tab-btn {"active" if i==0 else ""}" onclick="switchTier({i})" id="tierBtn{i}">{t}</button>' for i,t in enumerate(tier_keys))}
  <button class="tab-btn" onclick="switchTier({len(tier_keys)})" id="tierBtn{len(tier_keys)}">&#x23F1; Entry Optimizer</button>
</div>

<div class="main">
  {chr(10).join(f'<div class="tab-content {"active" if i==0 else ""}" id="tier{i}"></div>' for i,t in enumerate(tier_keys))}
  <div class="tab-content" id="tier{len(tier_keys)}"></div>
</div>

<script>
const TIER_KEYS   = {json.dumps(tier_keys)};
const STRAT_NAMES = {json.dumps(STRATEGY_NAMES)};
const STRAT_DESC  = {json.dumps(STRATEGY_DESC)};
const STRAT_DAILY = {json.dumps({k: True for k in ["S7","S8","S9","S10","S11","S12"]})};
const COLORS      = {json.dumps(COLORS)};
const TDATA       = {json.dumps(tdata)};
const TMETS       = {json.dumps(tmets_public)};
const TMETS_RAW   = {json.dumps(tmets_raw)};
const THEATS      = {json.dumps(theats)};
const BEST_TIER   = {json.dumps(best_per_tier)};
const LOT_SIZE    = {LOT_SIZE};
const CAPITAL_TIERS = {json.dumps(capital_tiers_js)};
const OPT_DATA    = {json.dumps(optimizer_data)};
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const N_TIERS = TIER_KEYS.length;

function pnlColor(v){{ return v>=0?'#10b981':'#ef4444'; }}
function pnlClass(v){{ return v>=0?'pos':'neg'; }}

const chartReg = {{}};
function destroyChart(id){{ if(chartReg[id]){{ chartReg[id].destroy(); delete chartReg[id]; }} }}

// ── Equity Chart ──────────────────────────────
function drawEquity(tierKey, canvasId){{
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if(!ctx) return;
  const data = TDATA[tierKey];
  const datasets = Object.entries(data).map(([s,sd],i)=>{{
    const isDaily = STRAT_DAILY[s];
    return {{
      label: (STRAT_NAMES[s]||s) + (isDaily?' *':''),
      data: sd.dates.map((d,j)=>({{x:d,y:sd.cum_pnl[j]}})),
      borderColor: COLORS[i%COLORS.length],
      backgroundColor: 'transparent',
      borderWidth: isDaily ? 2.5 : 1.5,
      borderDash: isDaily ? [] : [4,3],
      pointRadius: 0,
      tension: 0.3,
    }};
  }});
  chartReg[canvasId] = new Chart(ctx, {{
    type:'line', data:{{datasets}},
    options:{{
      responsive:true, maintainAspectRatio:false,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{
        legend:{{labels:{{color:'#94a3b8',font:{{size:10}},boxWidth:12,generateLabels:function(chart){{
          const def = Chart.defaults.plugins.legend.labels.generateLabels(chart);
          def.forEach(l=>{{ if(l.text.endsWith(' *')) l.lineWidth=2.5; }});
          return def;
        }}}}}},
        tooltip:{{
          backgroundColor:'#1e2a3d',borderColor:'#2a3850',borderWidth:1,
          callbacks:{{label:ctx=>`  ${{ctx.dataset.label}}: Rs.${{ctx.parsed.y.toLocaleString('en-IN')}}`}}
        }}
      }},
      scales:{{
        x:{{type:'time',time:{{unit:'month'}},grid:{{color:'#1a2234'}},ticks:{{color:'#64748b',maxTicksLimit:20}}}},
        y:{{grid:{{color:'#1a2234'}},ticks:{{color:'#64748b',callback:v=>'Rs.'+v.toLocaleString('en-IN')}},
          title:{{display:true,text:'Cumulative P&L (Rs.)',color:'#64748b',font:{{size:10}}}}}}
      }}
    }}
  }});
}}

// ── Heatmap ───────────────────────────────────
function drawHeatmap(tierKey, stratKey, containerId){{
  const container = document.getElementById(containerId);
  if(!container) return;
  const data = (THEATS[tierKey]||{{}})[stratKey]||{{}};
  const keys = Object.keys(data);
  if(!keys.length){{ container.innerHTML='<p style="color:#64748b;text-align:center;padding:32px;">No trades for this strategy.</p>'; return; }}
  const years=[...new Set(keys.map(k=>parseInt(k.split('-')[0])))].sort();
  const maxAbs=Math.max(...Object.values(data).map(Math.abs),1);
  let html='<div class="heatmap-grid">';
  html+='<div class="hm-header"></div>';
  MONTHS.forEach(m=>html+=`<div class="hm-header">${{m}}</div>`);
  years.forEach(yr=>{{
    html+=`<div class="hm-year">${{yr}}</div>`;
    for(let mo=1;mo<=12;mo++){{
      const key=`${{yr}}-${{String(mo).padStart(2,'0')}}`;
      if(data[key]!==undefined){{
        const v=data[key];
        const ratio=v/maxAbs;
        let bg,txt;
        if(v>=0){{ const g=Math.round(100+ratio*105); bg=`rgba(16,${{g}},129,${{0.2+ratio*0.6}})`; txt='#10b981'; }}
        else {{ const r=Math.round(180+Math.abs(ratio)*75); bg=`rgba(${{r}},60,60,${{0.2+Math.abs(ratio)*0.6}})`; txt='#ef4444'; }}
        html+=`<div class="hm-cell" style="background:${{bg}};color:${{txt}};" title="${{key}}: Rs.${{v.toLocaleString('en-IN')}}">${{v>=0?'+':''}}${{Math.round(v/1000)}}k</div>`;
      }} else html+='<div class="hm-cell hm-empty">-</div>';
    }}
  }});
  html+='</div>';
  container.innerHTML=html;
}}

// ── Metrics Table ─────────────────────────────
function buildMetricsTable(tierKey,bestS){{
  const strats=Object.keys(STRAT_NAMES);
  const mets=TMETS[tierKey]||{{}};
  let html=`<table class="metrics-table"><thead><tr>
    <th>Strategy</th><th>Type</th><th>Total P&L</th><th>CAGR</th>
    <th>Sharpe</th><th>Win %</th><th>Profit Factor</th><th>Max DD</th><th>Trades</th>
  </tr></thead><tbody>`;
  strats.forEach(s=>{{
    const m=mets[s]; if(!m) return;
    const isBest=s===bestS;
    const isDaily=!!STRAT_DAILY[s];
    const pnlNum=parseFloat((m['Total P&L']||'0').replace(/[Rs.,]/g,'').replace(/Rs/g,''));
    const wr=parseFloat((m['Win Rate']||'0'));
    html+=`<tr class="${{isBest?'best-row':''}}">
      <td><span style="color:${{COLORS[parseInt(s[1])+( s.length>2?parseInt(s[2])||0:0 )-1]}}">${{s}}</span>
        ${{isBest?'&#x1F451;':''}} ${{STRAT_NAMES[s]}}
        ${{isDaily?'<span class="daily-badge">daily</span>':''}}
      </td>
      <td style="color:${{isDaily?'#10b981':'#64748b'}};font-size:0.72rem;font-weight:600">${{isDaily?'DAILY':'COND'}}</td>
      <td class="${{pnlNum>=0?'pos':'neg'}}">${{m['Total P&L']||'-'}}</td>
      <td>${{m['CAGR']||'-'}}</td>
      <td class="${{parseFloat(m['Sharpe'])>=1?'pos':parseFloat(m['Sharpe'])<0?'neg':''}}">${{m['Sharpe']||'-'}}</td>
      <td class="${{wr>=55?'pos':wr>=50?'':'neg'}}">${{m['Win Rate']||'-'}}</td>
      <td class="${{parseFloat(m['Profit Factor'])>=1?'pos':'neg'}}">${{m['Profit Factor']||'-'}}</td>
      <td class="neg">${{m['Max Drawdown']||'-'}}</td>
      <td style="color:#64748b">${{m['Total Trades']||0}}</td>
    </tr>`;
  }});
  html+='</tbody></table>';
  return html;
}}

// ── KPIs ─────────────────────────────────────
function buildKPIs(tierKey,bestS){{
  const m=TMETS[tierKey][bestS]||{{}};
  const ct=CAPITAL_TIERS[tierKey]||{{}};
  const pnlNum=parseFloat((m['Total P&L']||'0').replace(/[Rs.,]/g,'').replace(/Rs/g,''));
  const wr=parseFloat(m['Win Rate']||0);
  return `<div class="kpi-row">
    <div class="kpi"><div class="label">Capital</div><div class="value" style="color:#00d4ff">Rs.${{ct.capital?.toLocaleString('en-IN')}}</div></div>
    <div class="kpi"><div class="label">Lots</div><div class="value" style="color:#7c3aed">${{ct.lots}} lots</div></div>
    <div class="kpi"><div class="label">Best P&L</div><div class="value" style="color:${{pnlColor(pnlNum)}}">${{m['Total P&L']||'-'}}</div></div>
    <div class="kpi"><div class="label">CAGR</div><div class="value" style="color:#10b981">${{m['CAGR']||'-'}}</div></div>
    <div class="kpi"><div class="label">Sharpe</div><div class="value" style="color:#f59e0b">${{m['Sharpe']||'-'}}</div></div>
    <div class="kpi"><div class="label">Win Rate</div><div class="value" style="color:${{wr>=55?'#10b981':'#f59e0b'}}">${{m['Win Rate']||'-'}}</div></div>
    <div class="kpi"><div class="label">Max Drawdown</div><div class="value" style="color:#ef4444">${{m['Max Drawdown']||'-'}}</div></div>
  </div>`;
}}

// ── Strategy mini-cards ─────────────────────
function buildStratCards(tierKey,bestS,activeSIdx){{
  const strats=Object.keys(STRAT_NAMES).filter(s=>TMETS[tierKey]&&TMETS[tierKey][s]);
  let html='<div class="strat-cards">';
  strats.forEach((s,i)=>{{
    const m=TMETS[tierKey][s]; if(!m) return;
    const pnlNum=parseFloat((m['Total P&L']||'0').replace(/[Rs.,]/g,'').replace(/Rs/g,''));
    const wr=parseFloat(m['Win Rate']||0);
    const isDaily=!!STRAT_DAILY[s];
    html+=`<div class="strat-card ${{i===activeSIdx?'active':''}}" onclick="switchStrat(currentTier,${{i}})" id="sc-${{currentTier}}-${{i}}">
      <div class="sc-num">${{s}} ${{s===bestS?'&#x1F451;':''}}${{isDaily?'<span class="daily-badge">daily</span>':''}}</div>
      <div class="sc-name" style="color:${{COLORS[i%COLORS.length]}}">${{STRAT_NAMES[s]}}</div>
      <div class="sc-pnl" style="color:${{pnlColor(pnlNum)}}">${{m['Total P&L']||'-'}}</div>
      <div class="sc-wr">Win: <b style="color:${{wr>=55?'#10b981':wr>=50?'#f59e0b':'#ef4444'}}">${{wr.toFixed(1)}}%</b></div>
    </div>`;
  }});
  html+='</div>';
  return html;
}}

// ── Best Banner ──────────────────────────────
function buildBestBanner(tierKey){{
  const bestS=BEST_TIER[tierKey];
  if(!bestS) return '';
  const m=TMETS[tierKey][bestS]||{{}};
  const pnlNum=parseFloat((m['Total P&L']||'0').replace(/[Rs.,]/g,'').replace(/Rs/g,''));
  const isDaily=!!STRAT_DAILY[bestS];
  return `<div class="best-banner">
    <div class="best-trophy">&#x1F3C6;</div>
    <div>
      <div class="best-title">Best Strategy for this Tier ${{isDaily?'[DAILY USE]':''}}</div>
      <div class="best-name">${{STRAT_NAMES[bestS]}}</div>
      <div class="best-desc">${{STRAT_DESC[bestS]}}</div>
    </div>
    <div class="best-stats">
      <div class="best-stat"><div class="val">${{m['CAGR']||'-'}}</div><div class="key">CAGR</div></div>
      <div class="best-stat"><div class="val">${{m['Sharpe']||'-'}}</div><div class="key">Sharpe</div></div>
      <div class="best-stat"><div class="val">${{m['Win Rate']||'-'}}</div><div class="key">Win Rate</div></div>
      <div class="best-stat"><div class="val" style="color:${{pnlColor(pnlNum)}}">${{m['Total P&L']||'-'}}</div><div class="key">Total P&L</div></div>
    </div>
  </div>`;
}}

// ── Optimizer Tab ────────────────────────────
function renderOptimizer(){{
  const container=document.getElementById(`tier${{N_TIERS}}`);
  if(!container||container.dataset.rendered) return;
  container.dataset.rendered='1';

  const times=OPT_DATA.map(d=>d.time);
  const winRates=OPT_DATA.map(d=>d.win_rate);
  const avgPnls=OPT_DATA.map(d=>d.avg_pnl);
  const maxWR=Math.max(...winRates);
  const bgColors=winRates.map(wr=>wr===maxWR?'#ff922b':wr>=55?'rgba(16,185,129,0.7)':wr>=50?'rgba(0,212,255,0.5)':'rgba(239,68,68,0.5)');

  container.innerHTML=`
  <div style="margin-bottom:24px">
    <p class="section-label">&#x23F1; Entry Time Optimizer — Simple Trend-Follow (Hold 90 min)</p>
    <div class="card">
      <div class="card-title">Win Rate by Entry Time</div>
      <div class="card-subtitle">If price > daily open: buy Call | else: buy Put | Exit after 90 minutes. Best time highlighted in orange.</div>
      <div class="optimizer-wrap"><canvas id="opt-wr-chart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Average P&L per Trade by Entry Time (10 lots)</div>
      <div class="card-subtitle">Positive = profitable on average. The combination of high win rate + high avg P&L gives the optimal entry.</div>
      <div class="optimizer-wrap"><canvas id="opt-pnl-chart"></canvas></div>
    </div>
    <div class="card" style="overflow-x:auto">
      <div class="card-title">Full Optimizer Results Table</div>
      <table class="metrics-table">
        <thead><tr><th>Entry Time</th><th>Trades</th><th>Win Rate</th><th>Avg P&L/Trade</th><th>Total P&L</th><th>Verdict</th></tr></thead>
        <tbody>
          ${{OPT_DATA.map(d=>{{
            const isBest=d.win_rate===maxWR;
            return `<tr class="${{isBest?'best-row':''}}">
              <td style="font-weight:700;color:${{isBest?'#ff922b':'#e2e8f0'}}">${{d.time}}${{isBest?' &#x1F3C6;':''}}</td>
              <td style="color:#64748b">${{d.trades}}</td>
              <td class="${{d.win_rate>=55?'pos':d.win_rate>=50?'':'neg'}}">${{d.win_rate}}%</td>
              <td class="${{d.avg_pnl>=0?'pos':'neg'}}">Rs.${{d.avg_pnl.toLocaleString('en-IN')}}</td>
              <td class="${{d.total_pnl>=0?'pos':'neg'}}">Rs.${{d.total_pnl.toLocaleString('en-IN')}}</td>
              <td style="font-size:0.72rem;color:${{d.win_rate>=55?'#10b981':d.win_rate>=50?'#f59e0b':'#ef4444'}}">${{d.win_rate>=55?'Strong':d.win_rate>=50?'Moderate':'Weak'}}</td>
            </tr>`;
          }}).join('')}}
        </tbody>
      </table>
    </div>
  </div>`;

  // WR chart
  destroyChart('opt-wr-chart');
  chartReg['opt-wr-chart']=new Chart(document.getElementById('opt-wr-chart'),{{
    type:'bar',
    data:{{labels:times,datasets:[{{
      label:'Win Rate %',data:winRates,
      backgroundColor:bgColors,borderRadius:6,
    }}]}},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>`Win Rate: ${{ctx.parsed.y}}%`}}}}}},
      scales:{{
        x:{{grid:{{color:'#1a2234'}},ticks:{{color:'#64748b'}}}},
        y:{{min:40,grid:{{color:'#1a2234'}},ticks:{{color:'#64748b',callback:v=>v+'%'}},
          title:{{display:true,text:'Win Rate (%)',color:'#64748b',font:{{size:10}}}}}}
      }}
    }}
  }});

  // PnL chart
  destroyChart('opt-pnl-chart');
  chartReg['opt-pnl-chart']=new Chart(document.getElementById('opt-pnl-chart'),{{
    type:'bar',
    data:{{labels:times,datasets:[{{
      label:'Avg P&L (Rs.)',data:avgPnls,
      backgroundColor:avgPnls.map(v=>v>=0?'rgba(16,185,129,0.6)':'rgba(239,68,68,0.5)'),
      borderRadius:6,
    }}]}},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>`Avg P&L: Rs.${{ctx.parsed.y.toLocaleString('en-IN')}}`}}}}}},
      scales:{{
        x:{{grid:{{color:'#1a2234'}},ticks:{{color:'#64748b'}}}},
        y:{{grid:{{color:'#1a2234'}},ticks:{{color:'#64748b',callback:v=>'Rs.'+v.toLocaleString('en-IN')}},
          title:{{display:true,text:'Avg P&L per Trade (Rs.)',color:'#64748b',font:{{size:10}}}}}}
      }}
    }}
  }});
}}

// ── Render Tier ──────────────────────────────
let currentTier=0;
let currentStrat={{}};

function renderTier(tierIdx){{
  if(tierIdx===N_TIERS){{ renderOptimizer(); return; }}
  const tierKey=TIER_KEYS[tierIdx];
  const bestS=BEST_TIER[tierKey];
  const strats=Object.keys(STRAT_NAMES).filter(s=>TMETS[tierKey]&&TMETS[tierKey][s]);
  const sIdx=currentStrat[tierIdx]||0;
  const activeS=strats[sIdx]||strats[0];
  const container=document.getElementById(`tier${{tierIdx}}`);
  if(!container) return;

  container.innerHTML=`
    ${{buildBestBanner(tierKey)}}
    ${{buildKPIs(tierKey,bestS)}}
    <p class="section-label">Equity Curves (solid=daily, dashed=conditional)</p>
    <div class="card">
      <div class="card-title">All 12 Strategies Comparison</div>
      <div class="card-subtitle">* = Daily-use strategies (green badge). Cumulative P&L over 10 years for ${{tierKey}}</div>
      <div class="chart-wrap"><canvas id="eq-${{tierIdx}}"></canvas></div>
    </div>
    <p class="section-label">Strategy Performance Table</p>
    <div class="card" style="overflow-x:auto">${{buildMetricsTable(tierKey,bestS)}}</div>
    <p class="section-label">Monthly P&L Heatmap</p>
    ${{buildStratCards(tierKey,bestS,sIdx)}}
    <div class="card">
      <div class="card-title" id="hm-title-${{tierIdx}}">${{STRAT_NAMES[activeS]}} - Monthly P&L</div>
      <div class="card-subtitle" id="hm-sub-${{tierIdx}}">${{STRAT_DESC[activeS]}}</div>
      <div id="hm-${{tierIdx}}" style="margin-top:14px;overflow-x:auto"></div>
    </div>`;

  drawEquity(tierKey,`eq-${{tierIdx}}`);
  drawHeatmap(tierKey,activeS,`hm-${{tierIdx}}`);
}}

function switchTier(idx){{
  document.querySelectorAll('.tab-btn').forEach((b,i)=>b.classList.toggle('active',i===idx));
  document.querySelectorAll('.tab-content').forEach((c,i)=>c.classList.toggle('active',i===idx));
  currentTier=idx;
  renderTier(idx);
}}

function switchStrat(tierIdx,sIdx){{
  currentStrat[tierIdx]=sIdx;
  const tierKey=TIER_KEYS[tierIdx];
  const strats=Object.keys(STRAT_NAMES).filter(s=>TMETS[tierKey]&&TMETS[tierKey][s]);
  const activeS=strats[sIdx]||strats[0];
  document.querySelectorAll(`[id^="sc-${{tierIdx}}-"]`).forEach((el,i)=>el.classList.toggle('active',i===sIdx));
  drawHeatmap(tierKey,activeS,`hm-${{tierIdx}}`);
  const te=document.getElementById(`hm-title-${{tierIdx}}`);
  if(te) te.textContent=`${{STRAT_NAMES[activeS]}} - Monthly P&L`;
  const se=document.getElementById(`hm-sub-${{tierIdx}}`);
  if(se) se.textContent=STRAT_DESC[activeS]||'';
}}

window.addEventListener('load',()=>{{
  renderTier(0);
  TIER_KEYS.forEach((_,i)=>{{ if(i>0) setTimeout(()=>renderTier(i),200*i); }});
}});
</script>
</body>
</html>"""
    return html

# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
def main():
    print("="*60)
    print("  Nifty 50 Options Backtester v2 — 12 Strategies")
    print("="*60)

    df5min, daily = load_data()

    # Run entry time optimizer (use medium tier = 10 lots for consistency)
    optimizer_data = run_entry_optimizer(df5min, daily, lots=10)

    all_tier_results  = {}
    all_tier_metrics  = {}
    all_tier_heatmaps = {}

    for tier_name, tier_cfg in CAPITAL_TIERS.items():
        lots    = tier_cfg["lots"]
        capital = tier_cfg["capital"]
        print(f"\n{'─'*55}")
        print(f"  Tier: {tier_name} | Capital: Rs.{capital:,} | Lots: {lots}")
        print(f"{'─'*55}")

        dfs = run_strategies(df5min, daily, lots)
        metrics  = {s: compute_metrics(df, capital) for s, df in dfs.items()}
        heatmaps = {s: monthly_heatmap(df)          for s, df in dfs.items()}

        all_tier_results[tier_name]  = dfs
        all_tier_metrics[tier_name]  = metrics
        all_tier_heatmaps[tier_name] = heatmaps

    # Summary
    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    ref_tier = list(CAPITAL_TIERS.keys())[1]  # medium
    print(f"\n  [Medium Tier Results]")
    mets = all_tier_metrics[ref_tier]
    rows = []
    for s,m in mets.items():
        if m:
            rows.append((s, STRATEGY_NAMES[s], m.get("raw_winrate",0), m.get("raw_sharpe",0), m.get("raw_pnl",0)))
    rows.sort(key=lambda x: x[2], reverse=True)  # sort by win rate
    for s, name, wr, sh, pnl in rows:
        print(f"    {s:3s} [{name:<25s}]: WinRate={wr:.1f}% | Sharpe={sh:.2f} | P&L=Rs.{pnl:>12,.0f}")

    html = build_dashboard(all_tier_results, all_tier_metrics, all_tier_heatmaps,
                           CAPITAL_TIERS, optimizer_data)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[OK] Dashboard saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

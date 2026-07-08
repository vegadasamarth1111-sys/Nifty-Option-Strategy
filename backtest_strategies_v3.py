"""
Nifty 50 Options Strategy Backtester — v3
==========================================
15 Strategies | Fixed Thursday Expiry | Black-Scholes Pricing
New: S13 Daily Straddle Sell+Stop | S14 Iron Condor | S15 Mid-Week Strangle Sell
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import time, timedelta, datetime
import warnings, os, json, sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
LOT_SIZE  = 65
RISK_FREE = 0.065
SLIPPAGE  = 0.005
TXN_COST  = 20

CAPITAL_TIERS = {
    "Small  (< Rs.50k)":  {"capital": 50_000,   "lots": 3},
    "Medium (< Rs.1.5L)": {"capital": 1_50_000,  "lots": 10},
    "Large  (> Rs.1.5L)": {"capital": 2_00_000,  "lots": 20},
}
DATA_DIR    = r"c:\samarth\projects\nifty 50 option"
OUTPUT_FILE = os.path.join(DATA_DIR, "strategy_backtest_v3_dashboard.html")

# ─────────────────────────────────────────────────────────────
# STRATEGY REGISTRY
# ─────────────────────────────────────────────────────────────
STRATEGY_NAMES = {
    "S1":  "ORB-15 Breakout",
    "S2":  "VIX + Gap Signal",
    "S3":  "Expiry Straddle Sell",
    "S4":  "Monday Gap Reversal",
    "S5":  "3-Day Momentum Buy",
    "S6":  "High-VIX Straddle Buy",
    "S7":  "First Candle Direction",
    "S8":  "10AM Trend Follow",
    "S9":  "11:30AM Reversal Buy",
    "S10": "Power Hour Follow",
    "S11": "1.5hr Range Breakout",
    "S12": "Morning Surge Scalp",
    "S13": "Daily Straddle Sell+Stop",
    "S14": "Iron Condor Sell Daily",
    "S15": "Mid-Week Strangle Sell",
}

STRATEGY_DESC = {
    "S1":  "Buy ATM breakout of first 15-min range at 9:30 AM. Exit 1:00 PM.",
    "S2":  "Buy Call/Put based on VIX regime + overnight gap. Exit 12:30 PM.",
    "S3":  "SELL: Sell ATM Straddle on Thursdays at 9:20 AM. Collect all-day theta. Exit 2:30 PM.",
    "S4":  "Buy ATM reversal on Mondays with gap > 0.5%. Exit 11:00 AM.",
    "S5":  "Buy ATM Call/Put following 3 consecutive green/red days. Exit 12:00 PM.",
    "S6":  "Buy ATM Straddle when VIX > 20. Bet on big move either side. Exit 1:00 PM.",
    "S7":  "DAILY: First candle direction at 9:15 AM sets bias. Entry 9:20 AM, Exit 11:00 AM.",
    "S8":  "DAILY: Confirm trend at 10:00 AM vs open (> 0.2% move). Exit 12:30 PM.",
    "S9":  "DAILY: Fade the morning move at 11:30 AM if > 0.5% from open. Exit 1:30 PM.",
    "S10": "DAILY: Follow day trend at 2:15 PM in last trading hour. Exit 3:20 PM.",
    "S11": "DAILY: Break of 1.5-hour range at 10:45 AM in breakout direction. Exit 1:15 PM.",
    "S12": "DAILY: Volatile opening (> 0.4% range) surge continuation at 9:45 AM. Exit 11:30 AM.",
    "S13": "SELL DAILY: Sell ATM Straddle at 9:30 AM every day. Stop if 0.6% move by 12 PM. Exit 2:30 PM.",
    "S14": "SELL DAILY: Iron Condor at 10 AM (sell 0.5% OTM call+put, buy 1.5% OTM as hedge). Exit 2:30 PM.",
    "S15": "SELL TUE/WED/THU: Sell 1% OTM Strangle at 9:30 AM. Stop at 1.2% move. Exit 2:30 PM.",
}

SELL_STRATS = {"S3","S13","S14","S15"}
DAILY_STRATS= {"S7","S8","S9","S10","S11","S12","S13","S14","S15"}
COLORS = [
    "#00d4ff","#ff6b6b","#51cf66","#ffd43b","#cc5de8","#ff922b",
    "#20c997","#f06595","#74c0fc","#a9e34b","#e599f7","#ffa94d",
    "#38d9a9","#ff8787","#63e6be",
]

# ─────────────────────────────────────────────────────────────
# BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, opt="call"):
    if T <= 0 or sigma <= 0:
        intr = max(S-K,0) if opt=="call" else max(K-S,0)
        return max(intr, 0.01)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt=="call": return max(S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2), 0.01)
    return max(K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1), 0.01)

def atm(spot): return round(spot/50)*50
def otm_c(spot, pct): return round(spot*(1+pct)/50)*50
def otm_p(spot, pct): return round(spot*(1-pct)/50)*50

# ─────────────────────────────────────────────────────────────
# EXPIRY — FIXED: returns THIS week's Thursday
# ─────────────────────────────────────────────────────────────
def get_expiry(dt):
    """Returns this week's Thursday. If today IS Thursday, returns today."""
    days = (3 - dt.weekday()) % 7   # 0 on Thu, 1 on Wed, 2 on Tue, 3 on Mon, 6 on Fri
    return (dt + timedelta(days=days)).date()

# Intraday time fraction (fraction of year elapsed since market open)
def tof(hr, mn=0):
    elapsed = hr - 9 + (mn - 15) / 60     # hours since 9:15 AM
    return max(elapsed, 0) / (252 * 6.25)

TIME_FRAC = {
    time(9,15): tof(9,15), time(9,20): tof(9,20), time(9,25): tof(9,25),
    time(9,30): tof(9,30), time(9,45): tof(9,45),
    time(10,0): tof(10,0), time(10,15):tof(10,15), time(10,30):tof(10,30),
    time(10,45):tof(10,45),time(11,0): tof(11,0),  time(11,15):tof(11,15),
    time(11,30):tof(11,30),time(12,0): tof(12,0),  time(12,30):tof(12,30),
    time(13,0): tof(13,0), time(13,15):tof(13,15), time(13,30):tof(13,30),
    time(14,0): tof(14,0), time(14,15):tof(14,15), time(14,30):tof(14,30),
    time(15,15):tof(15,15),time(15,20):tof(15,20),
}
def tf(t): return TIME_FRAC.get(t, tof(t.hour, t.minute))

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_data():
    print("[*] Loading data...")
    df5 = pd.read_csv(os.path.join(DATA_DIR,"NIFTY50_INTRADAY_5m_10YEARS.csv"))
    df5.columns = df5.columns.str.strip()
    df5.rename(columns={df5.columns[0]:"timestamp"}, inplace=True)
    df5["timestamp"] = pd.to_datetime(df5["timestamp"])

    dfa = pd.read_csv(os.path.join(DATA_DIR,"NIFTY50_5min_angelone.csv"))
    dfa.columns = dfa.columns.str.strip()
    dfa.rename(columns={dfa.columns[0]:"timestamp"}, inplace=True)
    dfa["timestamp"] = pd.to_datetime(dfa["timestamp"],utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    df5min = pd.concat([df5,dfa],ignore_index=True)
    df5min = df5min.dropna(subset=["timestamp","close"])
    df5min = df5min.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df5min["date"] = df5min["timestamp"].dt.date
    df5min["time"] = df5min["timestamp"].dt.time

    daily = pd.read_csv(os.path.join(DATA_DIR,"NIFTY50_MASTER_DAILY.csv"))
    daily.columns = daily.columns.str.strip()
    daily.rename(columns={daily.columns[0]:"date"}, inplace=True)
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily = daily[["date","open","close","vix"]].rename(columns={"open":"d_open","close":"d_close","vix":"vix"})
    daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    daily["vix"]     = daily["vix"].fillna(method="ffill").fillna(15.0)
    daily["vix_iv"]  = daily["vix"]/100.0
    daily["prev_c"]  = daily["d_close"].shift(1)
    daily["gap_pct"] = (daily["d_open"]-daily["prev_c"])/daily["prev_c"]
    daily["date_dt"] = pd.to_datetime(daily["date"])
    daily["dow"]     = daily["date_dt"].dt.dayofweek
    daily["is_thu"]  = (daily["dow"]==3).astype(int)
    daily["is_mon"]  = (daily["dow"]==0).astype(int)
    daily["ret"]     = daily["d_close"].pct_change()
    daily["trend3"]  = (daily["ret"].rolling(3).apply(
        lambda x: 1 if all(v>0 for v in x) else (-1 if all(v<0 for v in x) else 0)
    )).fillna(0).astype(int)

    print(f"  [OK] Intraday: {len(df5min):,} rows | {df5min['date'].min()} to {df5min['date'].max()}")
    print(f"  [OK] Daily:    {len(daily):,} rows | VIX available")
    return df5min, daily

# ─────────────────────────────────────────────────────────────
# TRADE SIMULATOR
# ─────────────────────────────────────────────────────────────
def sim(spot_e, spot_x, vix_iv, expiry, date, t_e, t_x, side, action, lots):
    sigma  = max(vix_iv, 0.05)
    T_base = max((expiry-date).days/365.0, 0)
    Te     = max(T_base - t_e, 1/365.0)
    Tx     = max(T_base - t_x, 0.0)
    K      = atm(spot_e)
    pe     = bs_price(spot_e, K, Te, RISK_FREE, sigma, side)
    px     = bs_price(spot_x, K, Tx, RISK_FREE, sigma, side)
    if action=="buy":
        u = px*(1-SLIPPAGE) - pe*(1+SLIPPAGE)
    else:
        u = pe*(1-SLIPPAGE) - px*(1+SLIPPAGE)
    return u*lots*LOT_SIZE - TXN_COST, pe, px

def sim_otm(spot_e, spot_x, K, vix_iv, expiry, date, t_e, t_x, side, action, lots):
    """Simulate with a specified strike (for OTM options)."""
    sigma  = max(vix_iv, 0.05)
    T_base = max((expiry-date).days/365.0, 0)
    Te     = max(T_base - t_e, 1/365.0)
    Tx     = max(T_base - t_x, 0.0)
    pe     = bs_price(spot_e, K, Te, RISK_FREE, sigma, side)
    px     = bs_price(spot_x, K, Tx, RISK_FREE, sigma, side)
    if action=="buy":
        u = px*(1-SLIPPAGE) - pe*(1+SLIPPAGE)
    else:
        u = pe*(1-SLIPPAGE) - px*(1+SLIPPAGE)
    return u*lots*LOT_SIZE - TXN_COST, pe, px

# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────
def p_at(dg, t):
    row = dg[dg["time"]==t]
    if len(row): return float(row.iloc[-1]["close"])
    ts = pd.to_datetime([str(x) for x in dg["time"].values])
    tgt= pd.to_datetime(str(t))
    d  = abs(ts-tgt)
    if d.min().seconds>600: return None
    return float(dg.iloc[d.argmin()]["close"])

def p_open(dg):
    row = dg[dg["time"]==time(9,15)]
    if len(row): return float(row.iloc[0]["open"])
    return float(dg.sort_values("time").iloc[0]["close"])

# ─────────────────────────────────────────────────────────────
# ENTRY TIME OPTIMIZER
# ─────────────────────────────────────────────────────────────
SCAN_TIMES = [
    time(9,15),time(9,30),time(9,45),
    time(10,0),time(10,15),time(10,30),time(10,45),
    time(11,0),time(11,30),time(12,0),time(12,30),
    time(13,0),time(13,30),time(14,0),time(14,30),
]

def run_optimizer(df5min, daily, lots=10):
    print("\n[>>] Entry Time Optimizer (trend-follow, 90-min hold)...")
    grouped = {str(d):g for d,g in df5min.groupby("date")}
    res = {str(t):{"w":0,"l":0,"pnl":0.0,"n":0} for t in SCAN_TIMES}

    for _,row in daily.iterrows():
        date   = row["date"]; dk=str(date)
        vix_iv = row["vix_iv"]; d_open=row["d_open"]
        if dk not in grouped or pd.isna(d_open): continue
        dg     = grouped[dk]
        expiry = get_expiry(row["date_dt"])
        for et in SCAN_TIMES:
            ed = datetime.combine(date, et)
            xt = min((ed+timedelta(hours=1.5)).time(), time(15,15))
            pe_s = p_at(dg, et)
            px_s = p_at(dg, xt)
            if pe_s is None or px_s is None: continue
            side = "call" if pe_s>=d_open else "put"
            try:
                pnl,_,_ = sim(pe_s,px_s,vix_iv,expiry,date,tf(et),tf(xt),side,"buy",lots)
                r=res[str(et)]
                r["n"]+=1; r["pnl"]+=pnl
                if pnl>0: r["w"]+=1
                else: r["l"]+=1
            except: pass

    out=[]
    for t in SCAN_TIMES:
        r=res[str(t)]
        n=r["n"]
        if n>0:
            wr=r["w"]/n*100
            ap=r["pnl"]/n
            out.append({"time":t.strftime("%H:%M"),"trades":n,
                        "win_rate":round(wr,1),"avg_pnl":round(ap,0),
                        "total_pnl":round(r["pnl"],0)})
            print(f"  {t.strftime('%H:%M')}: {n:4d} trades | WinRate={wr:.1f}% | AvgPnL=Rs.{ap:>8,.0f}")
    if out:
        best=max(out,key=lambda x:x["win_rate"])
        print(f"\n  [BEST] {best['time']} | WinRate:{best['win_rate']}% | AvgPnL:Rs.{best['avg_pnl']:,.0f}")
    return out

# ─────────────────────────────────────────────────────────────
# MAIN STRATEGY RUNNER
# ─────────────────────────────────────────────────────────────
def run_strategies(df5min, daily, lots):
    print(f"\n[>>] Running all 15 strategies ({lots} lots)...")
    results = {k:[] for k in STRATEGY_NAMES}
    grouped = {str(d):g for d,g in df5min.groupby("date")}

    def log(s, pnl, side, pe, px):
        results[s].append({"date":date,"pnl":pnl,"side":side,"entry":pe,"exit":px})

    for _,row in daily.iterrows():
        date   = row["date"]; dk=str(date)
        vix_iv = row["vix_iv"]
        gap    = row["gap_pct"] if pd.notna(row["gap_pct"]) else 0
        d_open = row["d_open"]  if pd.notna(row["d_open"])  else None
        dow    = row["dow"]
        is_thu = row["is_thu"]; is_mon=row["is_mon"]
        trend3 = row["trend3"]

        if dk not in grouped: continue
        dg = grouped[dk]
        if len(dg)<5: continue

        expiry = get_expiry(row["date_dt"])  # FIXED: this week's Thursday
        p = lambda t: p_at(dg,t)

        # ── S1: ORB-15 ─────────────────────────────────────────────
        try:
            p915=p(time(9,15)); p920=p(time(9,20)); p925=p(time(9,25))
            p930=p(time(9,30)); p1300=p(time(13,0))
            if all(v is not None for v in [p915,p920,p925,p930,p1300]):
                rh=max(p915,p920,p925); rl=min(p915,p920,p925)
                side="call" if p930>rh else "put" if p930<rl else None
                if side:
                    pnl,pe,px=sim(p930,p1300,vix_iv,expiry,date,tf(time(9,30)),tf(time(13,0)),side,"buy",lots)
                    log("S1",pnl,side,pe,px)
        except: pass

        # ── S2: VIX + Gap ──────────────────────────────────────────
        try:
            p920=p(time(9,20)); p1230=p(time(12,30))
            if p920 and p1230 and pd.notna(gap):
                side=(("put" if gap>0.003 else "call" if gap<-0.003 else None) if vix_iv*100>18
                      else ("call" if gap>0.003 else "put" if gap<-0.003 else None))
                if side:
                    pnl,pe,px=sim(p920,p1230,vix_iv,expiry,date,tf(time(9,20)),tf(time(12,30)),side,"buy",lots)
                    log("S2",pnl,side,pe,px)
        except: pass

        # ── S3: Expiry Straddle Sell (Thursday) ────────────────────
        try:
            if is_thu:
                p920=p(time(9,20)); p1430=p(time(14,30))
                if p920 and p1430:
                    pc,pec,_=sim(p920,p1430,vix_iv,expiry,date,tf(time(9,20)),tf(time(14,30)),"call","sell",lots)
                    pp,pep,_=sim(p920,p1430,vix_iv,expiry,date,tf(time(9,20)),tf(time(14,30)),"put","sell",lots)
                    log("S3",pc+pp,"straddle_sell",pec+pep,0)
        except: pass

        # ── S4: Monday Gap Reversal ─────────────────────────────────
        try:
            if is_mon and abs(gap)>0.005:
                p920=p(time(9,20)); p1100=p(time(11,0))
                if p920 and p1100:
                    side="put" if gap>0 else "call"
                    pnl,pe,px=sim(p920,p1100,vix_iv,expiry,date,tf(time(9,20)),tf(time(11,0)),side,"buy",lots)
                    log("S4",pnl,side,pe,px)
        except: pass

        # ── S5: 3-Day Momentum ──────────────────────────────────────
        try:
            if trend3!=0:
                p920=p(time(9,20)); p1200=p(time(12,0))
                if p920 and p1200:
                    side="call" if trend3==1 else "put"
                    pnl,pe,px=sim(p920,p1200,vix_iv,expiry,date,tf(time(9,20)),tf(time(12,0)),side,"buy",lots)
                    log("S5",pnl,side,pe,px)
        except: pass

        # ── S6: High-VIX Straddle Buy ──────────────────────────────
        try:
            if vix_iv*100>20:
                p920=p(time(9,20)); p1300=p(time(13,0))
                if p920 and p1300:
                    pc,pec,_=sim(p920,p1300,vix_iv,expiry,date,tf(time(9,20)),tf(time(13,0)),"call","buy",lots)
                    pp,pep,_=sim(p920,p1300,vix_iv,expiry,date,tf(time(9,20)),tf(time(13,0)),"put","buy",lots)
                    log("S6",pc+pp,"straddle_buy",pec+pep,0)
        except: pass

        # ── S7: First Candle Direction (daily) ─────────────────────
        try:
            bars915=dg[dg["time"]==time(9,15)]
            p920=p(time(9,20)); p1100=p(time(11,0))
            if len(bars915) and p920 and p1100:
                fo=float(bars915.iloc[0]["open"]); fc=float(bars915.iloc[0]["close"])
                side="call" if fc>=fo else "put"
                pnl,pe,px=sim(p920,p1100,vix_iv,expiry,date,tf(time(9,20)),tf(time(11,0)),side,"buy",lots)
                log("S7",pnl,side,pe,px)
        except: pass

        # ── S8: 10AM Confirmed Trend (daily) ───────────────────────
        try:
            p1000=p(time(10,0)); p1230=p(time(12,30))
            if p1000 and p1230 and d_open:
                mv=(p1000-d_open)/d_open
                side="call" if mv>0.002 else "put" if mv<-0.002 else None
                if side:
                    pnl,pe,px=sim(p1000,p1230,vix_iv,expiry,date,tf(time(10,0)),tf(time(12,30)),side,"buy",lots)
                    log("S8",pnl,side,pe,px)
        except: pass

        # ── S9: 11:30AM Reversal (daily) ───────────────────────────
        try:
            p1130=p(time(11,30)); p1330=p(time(13,30))
            if p1130 and p1330 and d_open:
                mv=(p1130-d_open)/d_open
                side="put" if mv>0.005 else "call" if mv<-0.005 else None
                if side:
                    pnl,pe,px=sim(p1130,p1330,vix_iv,expiry,date,tf(time(11,30)),tf(time(13,30)),side,"buy",lots)
                    log("S9",pnl,side,pe,px)
        except: pass

        # ── S10: Power Hour (daily) ─────────────────────────────────
        try:
            p1415=p(time(14,15)); p1520=p(time(15,20))
            if p1415 and p1520 and d_open:
                mv=(p1415-d_open)/d_open
                side="call" if mv>0.001 else "put" if mv<-0.001 else None
                if side:
                    pnl,pe,px=sim(p1415,p1520,vix_iv,expiry,date,tf(time(14,15)),tf(time(15,20)),side,"buy",lots)
                    log("S10",pnl,side,pe,px)
        except: pass

        # ── S11: 1.5hr Range Breakout (daily) ──────────────────────
        try:
            rbars=dg[dg["time"]<=time(10,45)]
            p1045=p(time(10,45)); p1315=p(time(13,15))
            if len(rbars)>=5 and p1045 and p1315:
                rh=float(rbars["high"].max()); rl=float(rbars["low"].min())
                rng=rh-rl
                side="call" if p1045>rh-0.1*rng else "put" if p1045<rl+0.1*rng else None
                if side:
                    pnl,pe,px=sim(p1045,p1315,vix_iv,expiry,date,tf(time(10,45)),tf(time(13,15)),side,"buy",lots)
                    log("S11",pnl,side,pe,px)
        except: pass

        # ── S12: Morning Surge Scalp (daily) ───────────────────────
        try:
            p915=p(time(9,15)); p945=p(time(9,45)); p1130=p(time(11,30))
            mbars=dg[dg["time"]<=time(9,45)]
            if p915 and p945 and p1130 and len(mbars)>=3:
                mrng=(float(mbars["high"].max())-float(mbars["low"].min()))/p915
                if mrng>0.004:
                    side="call" if p945>=p915 else "put"
                    pnl,pe,px=sim(p945,p1130,vix_iv,expiry,date,tf(time(9,45)),tf(time(11,30)),side,"buy",lots)
                    log("S12",pnl,side,pe,px)
        except: pass

        # ── S13: Daily Straddle Sell + Stop ────────────────────────
        # Sell ATM straddle at 9:30 AM every day.
        # Check 12:00 PM: if spot moved > 0.6% from ATM strike, take early exit.
        # Otherwise hold to 2:30 PM.
        try:
            p930=p(time(9,30)); p1200=p(time(12,0)); p1430=p(time(14,30))
            if p930 and p1200 and p1430:
                K=atm(p930)
                mid_mv=abs(p1200-K)/K
                if mid_mv>0.006:                          # stopped out at 12:00
                    xs=p1200; xt=tf(time(12,0))
                else:                                     # hold to 2:30
                    xs=p1430; xt=tf(time(14,30))
                pc,pec,_=sim(p930,xs,vix_iv,expiry,date,tf(time(9,30)),xt,"call","sell",lots)
                pp,pep,_=sim(p930,xs,vix_iv,expiry,date,tf(time(9,30)),xt,"put","sell",lots)
                log("S13",pc+pp,"straddle_sell",pec+pep,0)
        except: pass

        # ── S14: Iron Condor Sell (daily) ──────────────────────────
        # At 10 AM: sell 0.5% OTM call + put, buy 1.5% OTM as hedge.
        # Defined-risk spread. Exit 2:30 PM.
        try:
            p1000=p(time(10,0)); p1430=p(time(14,30))
            if p1000 and p1430:
                sigma=max(vix_iv,0.05)
                T_base=max((expiry-date).days/365.0, 0)
                Te=max(T_base-tf(time(10,0)), 1/365.0)
                Tx=max(T_base-tf(time(14,30)), 0.0)
                # Strikes: sell 0.5% OTM, buy 1.5% OTM (hedge)
                K_sc=otm_c(p1000,0.005); K_lc=otm_c(p1000,0.015)
                K_sp=otm_p(p1000,0.005); K_lp=otm_p(p1000,0.015)
                # Entry prices
                pe_sc=bs_price(p1000,K_sc,Te,RISK_FREE,sigma,"call")
                pe_lc=bs_price(p1000,K_lc,Te,RISK_FREE,sigma,"call")
                pe_sp=bs_price(p1000,K_sp,Te,RISK_FREE,sigma,"put")
                pe_lp=bs_price(p1000,K_lp,Te,RISK_FREE,sigma,"put")
                # Net credit at entry: sell – hedge
                net_cr=(pe_sc+pe_sp-pe_lc-pe_lp)*(1-SLIPPAGE)
                # Exit prices
                px_sc=bs_price(p1430,K_sc,Tx,RISK_FREE,sigma,"call")
                px_lc=bs_price(p1430,K_lc,Tx,RISK_FREE,sigma,"call")
                px_sp=bs_price(p1430,K_sp,Tx,RISK_FREE,sigma,"put")
                px_lp=bs_price(p1430,K_lp,Tx,RISK_FREE,sigma,"put")
                # Net debit at exit: buy back sold – sell hedges
                net_db=(px_sc+px_sp-px_lc-px_lp)*(1+SLIPPAGE)
                pnl=(net_cr-net_db)*lots*LOT_SIZE - TXN_COST*4
                if net_cr>0:  # only take trade if credit is positive
                    log("S14",pnl,"iron_condor",net_cr,net_db)
        except: pass

        # ── S15: Mid-Week Strangle Sell ────────────────────────────
        # Tue/Wed/Thu only. Sell 1% OTM call + put at 9:30 AM.
        # Stop-check at 1:00 PM: if > 1.2% move from entry, exit early.
        # Otherwise hold to 2:30 PM.
        try:
            if dow in [1,2,3]:   # Tue=1, Wed=2, Thu=3
                p930=p(time(9,30)); p1300=p(time(13,0)); p1430=p(time(14,30))
                if p930 and p1300 and p1430:
                    sigma=max(vix_iv,0.05)
                    T_base=max((expiry-date).days/365.0, 0)
                    Te=max(T_base-tf(time(9,30)), 1/365.0)
                    K_c=otm_c(p930,0.01); K_p=otm_p(p930,0.01)
                    pe_c=bs_price(p930,K_c,Te,RISK_FREE,sigma,"call")
                    pe_p=bs_price(p930,K_p,Te,RISK_FREE,sigma,"put")
                    # Check stop
                    stop_mv=abs(p1300-p930)/p930
                    if stop_mv>0.012:              # stopped at 1 PM
                        xs=p1300; xt=tf(time(13,0))
                    else:
                        xs=p1430; xt=tf(time(14,30))
                    Tx=max(T_base-xt, 0.0)
                    px_c=bs_price(xs,K_c,Tx,RISK_FREE,sigma,"call")
                    px_p=bs_price(xs,K_p,Tx,RISK_FREE,sigma,"put")
                    credit=(pe_c+pe_p)*(1-SLIPPAGE)
                    debit =(px_c+px_p)*(1+SLIPPAGE)
                    pnl=(credit-debit)*lots*LOT_SIZE - TXN_COST*2
                    log("S15",pnl,"strangle_sell",credit,debit)
        except: pass

    # Convert to DataFrames
    dfs={}
    for k,v in results.items():
        if v:
            df=pd.DataFrame(v)
            df["date"]=pd.to_datetime(df["date"])
            df["year"]=df["date"].dt.year
            df["month"]=df["date"].dt.to_period("M")
            df["cum_pnl"]=df["pnl"].cumsum()
            dfs[k]=df
            tag = "[SELL]" if k in SELL_STRATS else "      "
            print(f"  {k:3s} {tag} [{STRATEGY_NAMES[k]:<26s}]: {len(df):5d} trades | P&L: Rs.{df['pnl'].sum():>14,.0f}")
        else:
            print(f"  {k}: 0 trades")
    return dfs

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def metrics(df, capital):
    if df is None or len(df)==0: return {}
    pnl=df["pnl"].values; cum=np.cumsum(pnl)
    tr=cum[-1]
    ny=(df["date"].max()-df["date"].min()).days/365.25
    cagr=((capital+tr)/capital)**(1/max(ny,0.1))-1 if ny>0 else 0
    dp=df.groupby(df["date"].dt.date)["pnl"].sum()
    sh=(dp.mean()/dp.std()*np.sqrt(252)) if dp.std()>0 else 0
    rm=np.maximum.accumulate(cum); mdd=(cum-rm).min()
    wins=pnl[pnl>0]; losses=pnl[pnl<0]
    wr=len(wins)/len(pnl)*100
    aw=wins.mean() if len(wins)>0 else 0
    al=losses.mean() if len(losses)>0 else 0
    pf=abs(wins.sum()/losses.sum()) if losses.sum()!=0 else 99.0
    return {
        "Total P&L":    f"Rs.{tr:,.0f}",
        "CAGR":         f"{cagr*100:.1f}%",
        "Sharpe":       f"{sh:.2f}",
        "Max Drawdown": f"Rs.{mdd:,.0f}",
        "Win Rate":     f"{wr:.1f}%",
        "Avg Win":      f"Rs.{aw:,.0f}",
        "Avg Loss":     f"Rs.{al:,.0f}",
        "Profit Factor":f"{pf:.2f}",
        "Total Trades": len(pnl),
        "raw_cagr": cagr, "raw_sharpe": sh, "raw_pnl": tr,
        "raw_maxdd": mdd, "raw_wr": wr,
    }

def heatmap(df):
    if df is None or len(df)==0: return {}
    m=df.groupby(["year",df["date"].dt.month])["pnl"].sum().reset_index()
    m.columns=["year","mo","pnl"]
    return {f"{int(r.year)}-{int(r.mo):02d}":round(r.pnl,0) for _,r in m.iterrows()}

# ─────────────────────────────────────────────────────────────
# DASHBOARD — Template-based (no f-string brace issues)
# ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Nifty 50 Options — Strategy Backtester v3</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0e1a;--bg2:#111827;--bg3:#1a2234;--card:#1e2a3d;--bdr:#2a3850;
  --acc:#00d4ff;--acc2:#7c3aed;--grn:#10b981;--red:#ef4444;--ylw:#f59e0b;
  --txt:#e2e8f0;--mut:#64748b;--org:#ff922b;}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--txt);}
/* Header */
.hdr{background:linear-gradient(135deg,#0a0e1a,#1a1040,#0a0e1a);
  border-bottom:1px solid var(--bdr);padding:24px 48px;
  display:flex;align-items:center;justify-content:space-between;position:relative;overflow:hidden;}
.hdr::before{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 80% 60% at 50% -20%,rgba(0,212,255,0.12),transparent);}
.hdr h1{font-size:1.8rem;font-weight:900;
  background:linear-gradient(135deg,#00d4ff,#7c3aed);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.02em;}
.hdr p{color:var(--mut);margin-top:5px;font-size:0.82rem;}
.badges{display:flex;gap:10px;}
.badge{background:linear-gradient(135deg,rgba(0,212,255,0.1),rgba(124,58,237,0.1));
  border:1px solid rgba(0,212,255,0.22);border-radius:10px;padding:10px 14px;text-align:center;}
.badge .n{font-size:1.5rem;font-weight:900;color:var(--acc);}
.badge .l{font-size:0.68rem;color:var(--mut);margin-top:2px;}
/* Opt banner */
.ob{background:linear-gradient(135deg,rgba(255,146,43,0.1),rgba(0,212,255,0.05));
  border:1px solid rgba(255,146,43,0.28);border-radius:14px;
  padding:18px 28px;margin:20px 48px 0;display:flex;align-items:center;gap:18px;}
.ob-icon{font-size:2.2rem;}
.ob-sub{font-size:0.7rem;color:var(--mut);font-weight:700;text-transform:uppercase;letter-spacing:.05em;}
.ob-main{font-size:1.3rem;font-weight:900;color:var(--org);margin:3px 0;}
.ob-txt{font-size:0.78rem;color:var(--mut);}
.ob-st{margin-left:auto;display:flex;gap:18px;}
.ob-st .v{font-size:1.1rem;font-weight:800;color:var(--acc);text-align:center;}
.ob-st .k{font-size:0.65rem;color:var(--mut);text-align:center;margin-top:2px;}
/* Tabs */
.tabs{padding:18px 48px 0;background:var(--bg2);border-bottom:1px solid var(--bdr);display:flex;gap:5px;flex-wrap:wrap;}
.tb{padding:8px 18px;border:none;border-radius:8px 8px 0 0;cursor:pointer;
  font-family:inherit;font-size:0.82rem;font-weight:600;background:transparent;
  color:var(--mut);transition:all .2s;}
.tb.active{background:var(--card);color:var(--acc);border-bottom:2px solid var(--acc);}
.tb:hover:not(.active){color:var(--txt);background:var(--bg3);}
.main{padding:24px 48px;max-width:1750px;margin:0 auto;}
.tc{display:none;}.tc.active{display:block;}
.sl{font-size:.7rem;font-weight:700;letter-spacing:.1em;color:var(--acc);
  text-transform:uppercase;margin-bottom:12px;}
/* Best banner */
.bb{background:linear-gradient(135deg,rgba(16,185,129,.1),rgba(0,212,255,.05));
  border:1px solid rgba(16,185,129,.22);border-radius:12px;padding:18px 26px;
  margin-bottom:20px;display:flex;align-items:center;gap:18px;}
.bb-ico{font-size:2.6rem;}
.bb-top{font-size:.68rem;color:var(--mut);font-weight:700;text-transform:uppercase;letter-spacing:.05em;}
.bb-nm{font-size:1.3rem;font-weight:900;color:var(--grn);margin:3px 0;}
.bb-ds{font-size:.78rem;color:var(--mut);max-width:460px;}
.bb-st{margin-left:auto;display:flex;gap:18px;flex-shrink:0;}
.bb-st .v{font-size:1.05rem;font-weight:800;color:var(--acc);text-align:center;}
.bb-st .k{font-size:.65rem;color:var(--mut);text-align:center;margin-top:2px;}
/* KPI */
.krow{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;}
.kpi{background:var(--bg3);border:1px solid var(--bdr);border-radius:10px;
  padding:12px 16px;flex:1;min-width:110px;}
.kpi .lb{font-size:.65rem;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.05em;}
.kpi .vl{font-size:1.18rem;font-weight:800;margin-top:3px;}
/* Card */
.card{background:var(--card);border:1px solid var(--bdr);border-radius:12px;
  padding:20px;position:relative;overflow:hidden;margin-bottom:20px;}
.card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--acc),var(--acc2));}
.ct{font-size:.82rem;font-weight:700;color:var(--mut);margin-bottom:3px;
  text-transform:uppercase;letter-spacing:.05em;}
.cs{font-size:.73rem;color:var(--mut);opacity:.7;margin-bottom:14px;}
.cw{position:relative;height:250px;}
/* Strategy mini-cards */
.scards{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:18px;}
.sc{background:var(--bg3);border:1px solid var(--bdr);border-radius:9px;
  padding:12px 14px;cursor:pointer;transition:all .2s;}
.sc:hover{border-color:var(--acc);}
.sc.active{border-color:var(--acc);background:var(--card);box-shadow:0 0 0 1px rgba(0,212,255,.14);}
.sc .sn{font-size:.65rem;color:var(--mut);font-weight:700;}
.sc .nm{font-weight:700;font-size:.78rem;margin:3px 0;}
.sc .pl{font-size:.9rem;font-weight:800;}
.sc .wr{font-size:.68rem;color:var(--mut);margin-top:2px;}
/* Badges */
.sell-b{display:inline-block;background:rgba(16,185,129,.15);color:var(--grn);
  border:1px solid rgba(16,185,129,.28);border-radius:3px;
  font-size:.55rem;font-weight:700;padding:1px 4px;margin-left:3px;
  vertical-align:middle;text-transform:uppercase;letter-spacing:.04em;}
.day-b{display:inline-block;background:rgba(0,212,255,.1);color:var(--acc);
  border:1px solid rgba(0,212,255,.25);border-radius:3px;
  font-size:.55rem;font-weight:700;padding:1px 4px;margin-left:3px;
  vertical-align:middle;text-transform:uppercase;letter-spacing:.04em;}
/* Metrics table */
.mt{width:100%;border-collapse:collapse;font-size:.78rem;}
.mt th{color:var(--mut);font-weight:600;padding:7px 9px;text-align:left;
  border-bottom:1px solid var(--bdr);font-size:.67rem;text-transform:uppercase;
  letter-spacing:.05em;white-space:nowrap;}
.mt td{padding:8px 9px;border-bottom:1px solid rgba(42,56,80,.35);white-space:nowrap;}
.mt tr:last-child td{border-bottom:none;}
.mt tr:hover td{background:rgba(0,212,255,.03);}
.br td{background:rgba(16,185,129,.06)!important;}
.pos{color:var(--grn);font-weight:600;}
.neg{color:var(--red);font-weight:600;}
/* Heatmap */
.hg{display:grid;grid-template-columns:50px repeat(12,1fr);gap:2px;font-size:.64rem;}
.hh{color:var(--mut);text-align:center;padding:3px 0;font-weight:600;}
.hy{color:var(--mut);font-weight:700;display:flex;align-items:center;}
.hc{border-radius:4px;text-align:center;padding:4px 1px;font-weight:600;
  cursor:default;transition:transform .15s;}
.hc:hover{transform:scale(1.12);z-index:10;position:relative;}
.he{background:var(--bg3);}
/* Optimizer */
.ow{height:220px;position:relative;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg2);}
::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:3px;}
@media(max-width:1100px){
  .hdr,.tabs,.main{padding-left:18px;padding-right:18px;}
  .ob{margin:14px 18px 0;}
  .scards{grid-template-columns:repeat(3,1fr);}
  .bb-st,.ob-st{display:none;}
}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>&#9889; Nifty 50 Options Backtester v3</h1>
    <p>15 Strategies (3 Sell-focused) + Entry Time Optimizer &bull; Fixed Expiry &bull; 10-Year Analysis &bull; Lot Size 65 units</p>
  </div>
  <div class="badges">
    <div class="badge"><div class="n">15</div><div class="l">Strategies</div></div>
    <div class="badge"><div class="n" style="color:#10b981">4</div><div class="l">Sell Strats</div></div>
    <div class="badge"><div class="n" style="color:#ff922b">11</div><div class="l">Daily-Use</div></div>
    <div class="badge"><div class="n">10Y</div><div class="l">Backtest</div></div>
  </div>
</div>
<div class="ob" id="opt-banner">
  <div class="ob-icon">&#9201;</div>
  <div>
    <div class="ob-sub">Entry Time Optimizer Result</div>
    <div class="ob-main" id="ob-main">Loading...</div>
    <div class="ob-txt" id="ob-txt"></div>
  </div>
  <div class="ob-st" id="ob-stats"></div>
</div>
<div class="tabs" id="tierTabs"></div>
<div class="main" id="mainContent"></div>

<script>
// ── ALL DATA INJECTED HERE ─────────────────────────────────
/*__DATA__*/

// ── Utilities ─────────────────────────────────────────────
const MO = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const CR = {};
function dch(id){if(CR[id]){CR[id].destroy();delete CR[id];}}
function pc(v){return v>=0?'#10b981':'#ef4444';}
function cls(v){return v>=0?'pos':'neg';}
function frs(s){return s.replace(/Rs\./g,'').replace(/[,]/g,'');}

// ── Init optimizer banner ──────────────────────────────────
(function(){
  if(!OPT_DATA||!OPT_DATA.length) return;
  const best=OPT_DATA.reduce((a,b)=>b.win_rate>a.win_rate?b:a,OPT_DATA[0]);
  document.getElementById('ob-main').textContent=
    'Best Entry: '+best.time+' AM  |  Win Rate: '+best.win_rate+'%';
  document.getElementById('ob-txt').textContent=
    'Avg P&L: Rs.'+best.avg_pnl.toLocaleString('en-IN')+' per trade (10 lots)  |  '+
    best.trades+' qualifying days in 10 years';
  document.getElementById('ob-stats').innerHTML=
    '<div class="ob-st"><div class="v">'+best.win_rate+'%</div><div class="k">Win Rate</div></div>'+
    '<div class="ob-st"><div class="v">Rs.'+best.avg_pnl.toLocaleString('en-IN')+'</div><div class="k">Avg P&L/Trade</div></div>'+
    '<div class="ob-st"><div class="v">90 min</div><div class="k">Hold Time</div></div>';
})();

// ── Build tier tabs ────────────────────────────────────────
(function(){
  const tabs=document.getElementById('tierTabs');
  const main=document.getElementById('mainContent');
  TIER_KEYS.forEach((t,i)=>{
    const b=document.createElement('button');
    b.className='tb'+(i===0?' active':'');
    b.textContent=t;
    b.onclick=()=>switchTier(i);
    b.id='tb'+i;
    tabs.appendChild(b);
    const d=document.createElement('div');
    d.className='tc'+(i===0?' active':'');
    d.id='tc'+i;
    main.appendChild(d);
  });
  // Optimizer tab
  const bo=document.createElement('button');
  bo.className='tb'; bo.textContent='&#9201; Entry Optimizer';
  bo.onclick=()=>switchTier(TIER_KEYS.length); bo.id='tb'+TIER_KEYS.length;
  tabs.appendChild(bo);
  const do2=document.createElement('div');
  do2.className='tc'; do2.id='tc'+TIER_KEYS.length;
  main.appendChild(do2);
})();

// ── Equity chart ──────────────────────────────────────────
function drawEq(tierKey, cid){
  dch(cid); const ctx=document.getElementById(cid); if(!ctx) return;
  const data=TDATA[tierKey];
  const ds=Object.entries(data).map(([s,sd],i)=>({
    label:(STRAT_NAMES[s]||s)+(SELL_STRATS[s]?' [S]':DAILY_STRATS[s]?' [D]':''),
    data:sd.dates.map((d,j)=>({x:d,y:sd.cum_pnl[j]})),
    borderColor:COLORS[i%COLORS.length],
    backgroundColor:'transparent',
    borderWidth:SELL_STRATS[s]?3:1.8,
    borderDash:SELL_STRATS[s]?[]:[4,3],
    pointRadius:0, tension:0.3,
  }));
  CR[cid]=new Chart(ctx,{
    type:'line',data:{datasets:ds},
    options:{
      responsive:true,maintainAspectRatio:false,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{labels:{color:'#94a3b8',font:{size:9.5},boxWidth:10}},
        tooltip:{backgroundColor:'#1e2a3d',borderColor:'#2a3850',borderWidth:1,
          callbacks:{label:c=>'  '+c.dataset.label+': Rs.'+c.parsed.y.toLocaleString('en-IN')}}
      },
      scales:{
        x:{type:'time',time:{unit:'month'},grid:{color:'#1a2234'},ticks:{color:'#64748b',maxTicksLimit:20}},
        y:{grid:{color:'#1a2234'},ticks:{color:'#64748b',callback:v=>'Rs.'+v.toLocaleString('en-IN')},
          title:{display:true,text:'Cumulative P&L (Rs.)',color:'#64748b',font:{size:10}}}
      }
    }
  });
}

// ── Heatmap ───────────────────────────────────────────────
function drawHm(tierKey, stratKey, cid){
  const c=document.getElementById(cid); if(!c) return;
  const data=(THEATS[tierKey]||{})[stratKey]||{};
  const keys=Object.keys(data);
  if(!keys.length){c.innerHTML='<p style="color:#64748b;text-align:center;padding:24px">No trades for this strategy.</p>';return;}
  const yrs=[...new Set(keys.map(k=>parseInt(k.split('-')[0])))].sort();
  const mx=Math.max(...Object.values(data).map(Math.abs),1);
  let h='<div class="hg"><div class="hh"></div>';
  MO.forEach(m=>h+='<div class="hh">'+m+'</div>');
  yrs.forEach(yr=>{
    h+='<div class="hy">'+yr+'</div>';
    for(let mo=1;mo<=12;mo++){
      const k=yr+'-'+String(mo).padStart(2,'0');
      if(data[k]!==undefined){
        const v=data[k],r=v/mx;
        const bg=v>=0?'rgba(16,'+(Math.round(100+r*105))+',129,'+(0.2+r*0.6)+')':
                      'rgba('+(Math.round(180+Math.abs(r)*75))+',60,60,'+(0.2+Math.abs(r)*0.6)+')';
        const tc=v>=0?'#10b981':'#ef4444';
        h+='<div class="hc" style="background:'+bg+';color:'+tc+';" title="'+k+': Rs.'+v.toLocaleString('en-IN')+'">'+(v>=0?'+':'')+Math.round(v/1000)+'k</div>';
      } else h+='<div class="hc he">-</div>';
    }
  });
  h+='</div>'; c.innerHTML=h;
}

// ── Metrics table ─────────────────────────────────────────
function buildMT(tierKey, bestS){
  const mets=TMETS[tierKey]||{};
  let h='<table class="mt"><thead><tr><th>Strategy</th><th>Type</th><th>Total P&L</th><th>CAGR</th><th>Sharpe</th><th>Win %</th><th>Profit Factor</th><th>Max DD</th><th>Trades</th></tr></thead><tbody>';
  Object.keys(STRAT_NAMES).forEach(s=>{
    const m=mets[s]; if(!m) return;
    const isBest=s===bestS;
    const isSell=!!SELL_STRATS[s]; const isDay=!!DAILY_STRATS[s];
    const pn=parseFloat(frs(m['Total P&L']||'0'));
    const wr=parseFloat(m['Win Rate']||0);
    h+='<tr class="'+(isBest?'br':'')+'"><td>';
    h+='<span style="color:'+COLORS[(parseInt(s.slice(1))-1)%COLORS.length]+'">'+s+'</span> ';
    if(isBest) h+='&#x1F451; ';
    h+=STRAT_NAMES[s];
    if(isSell) h+='<span class="sell-b">SELL</span>';
    else if(isDay) h+='<span class="day-b">DAILY</span>';
    h+='</td><td style="font-size:.7rem;font-weight:600;color:'+(isSell?'#10b981':isDay?'#00d4ff':'#64748b')+'">'+(isSell?'SELL':'BUY')+'</td>';
    h+='<td class="'+(pn>=0?'pos':'neg')+'">'+( m['Total P&L']||'-')+'</td>';
    h+='<td>'+(m['CAGR']||'-')+'</td>';
    h+='<td class="'+(parseFloat(m['Sharpe'])>=1?'pos':parseFloat(m['Sharpe'])<0?'neg':'')+'">'+( m['Sharpe']||'-')+'</td>';
    h+='<td class="'+(wr>=60?'pos':wr>=50?'':'neg')+'">'+( m['Win Rate']||'-')+'</td>';
    h+='<td class="'+(parseFloat(m['Profit Factor'])>=1?'pos':'neg')+'">'+( m['Profit Factor']||'-')+'</td>';
    h+='<td class="neg">'+( m['Max Drawdown']||'-')+'</td>';
    h+='<td style="color:#64748b">'+( m['Total Trades']||0)+'</td></tr>';
  });
  h+='</tbody></table>';
  return h;
}

// ── KPI row ───────────────────────────────────────────────
function buildKPIs(tierKey,bestS){
  const m=TMETS[tierKey][bestS]||{};
  const ct=CAPITAL_TIERS[tierKey]||{};
  const pn=parseFloat(frs(m['Total P&L']||'0'));
  const wr=parseFloat(m['Win Rate']||0);
  return '<div class="krow">'+
    '<div class="kpi"><div class="lb">Capital</div><div class="vl" style="color:#00d4ff">Rs.'+ct.capital?.toLocaleString('en-IN')+'</div></div>'+
    '<div class="kpi"><div class="lb">Lots</div><div class="vl" style="color:#7c3aed">'+ct.lots+'</div></div>'+
    '<div class="kpi"><div class="lb">Best P&L</div><div class="vl" style="color:'+pc(pn)+'">'+( m['Total P&L']||'-')+'</div></div>'+
    '<div class="kpi"><div class="lb">CAGR</div><div class="vl" style="color:#10b981">'+( m['CAGR']||'-')+'</div></div>'+
    '<div class="kpi"><div class="lb">Sharpe</div><div class="vl" style="color:#f59e0b">'+( m['Sharpe']||'-')+'</div></div>'+
    '<div class="kpi"><div class="lb">Win Rate</div><div class="vl" style="color:'+(wr>=60?'#10b981':wr>=50?'#f59e0b':'#ef4444')+'">'+( m['Win Rate']||'-')+'</div></div>'+
    '<div class="kpi"><div class="lb">Max Drawdown</div><div class="vl" style="color:#ef4444">'+( m['Max Drawdown']||'-')+'</div></div>'+
    '</div>';
}

// ── Strategy cards ────────────────────────────────────────
function buildSC(tierKey,bestS,active){
  const strats=Object.keys(STRAT_NAMES).filter(s=>TMETS[tierKey]&&TMETS[tierKey][s]);
  let h='<div class="scards">';
  strats.forEach((s,i)=>{
    const m=TMETS[tierKey][s]; if(!m) return;
    const pn=parseFloat(frs(m['Total P&L']||'0'));
    const wr=parseFloat(m['Win Rate']||0);
    const isSell=!!SELL_STRATS[s]; const isDay=!!DAILY_STRATS[s];
    h+='<div class="sc'+(i===active?' active':'')+(isSell?' sell-border':'')+
      '" onclick="switchS(cTier,'+i+')" id="sc-'+cTier+'-'+i+'">'+
      '<div class="sn">'+s+(s===bestS?' &#x1F451;':'')+(isSell?'<span class="sell-b">S</span>':isDay?'<span class="day-b">D</span>':'')+'</div>'+
      '<div class="nm" style="color:'+COLORS[(parseInt(s.slice(1))-1)%COLORS.length]+'">'+STRAT_NAMES[s]+'</div>'+
      '<div class="pl" style="color:'+pc(pn)+'">'+( m['Total P&L']||'-')+'</div>'+
      '<div class="wr">Win: <b style="color:'+(wr>=60?'#10b981':wr>=50?'#f59e0b':'#ef4444')+'">'+wr.toFixed(1)+'%</b></div>'+
      '</div>';
  });
  h+='</div>'; return h;
}

// ── Best banner ───────────────────────────────────────────
function buildBB(tierKey){
  const bs=BEST_TIER[tierKey]; if(!bs) return '';
  const m=TMETS[tierKey][bs]||{};
  const pn=parseFloat(frs(m['Total P&L']||'0'));
  const isSell=!!SELL_STRATS[bs];
  return '<div class="bb">'+
    '<div class="bb-ico">&#x1F3C6;</div>'+
    '<div>'+
      '<div class="bb-top">Best Strategy for this Tier '+(isSell?'[SELL - COLLECT PREMIUM]':'')+'</div>'+
      '<div class="bb-nm">'+STRAT_NAMES[bs]+'</div>'+
      '<div class="bb-ds">'+STRAT_DESC[bs]+'</div>'+
    '</div>'+
    '<div class="bb-st">'+
      '<div><div class="v">'+( m['CAGR']||'-')+'</div><div class="k">CAGR</div></div>'+
      '<div><div class="v">'+( m['Sharpe']||'-')+'</div><div class="k">Sharpe</div></div>'+
      '<div><div class="v">'+( m['Win Rate']||'-')+'</div><div class="k">Win Rate</div></div>'+
      '<div><div class="v" style="color:'+pc(pn)+'">'+( m['Total P&L']||'-')+'</div><div class="k">P&L</div></div>'+
    '</div></div>';
}

// ── Render tier ───────────────────────────────────────────
let cTier=0, cStrat={};
function renderTier(idx){
  if(idx===TIER_KEYS.length){renderOpt();return;}
  const tk=TIER_KEYS[idx], bs=BEST_TIER[tk];
  const strats=Object.keys(STRAT_NAMES).filter(s=>TMETS[tk]&&TMETS[tk][s]);
  const si=cStrat[idx]||0, as=strats[si]||strats[0];
  const c=document.getElementById('tc'+idx); if(!c) return;
  c.innerHTML=
    buildBB(tk)+
    buildKPIs(tk,bs)+
    '<p class="sl">&#128200; Equity Curves — All 15 Strategies (solid=SELL, dashed=BUY)</p>'+
    '<div class="card"><div class="ct">Cumulative P&L — All Strategies</div>'+
    '<div class="cs">SELL strategies (solid) vs BUY strategies (dashed). '+tk+'</div>'+
    '<div class="cw"><canvas id="eq'+idx+'"></canvas></div></div>'+
    '<p class="sl">&#128202; Performance Metrics Table</p>'+
    '<div class="card" style="overflow-x:auto">'+buildMT(tk,bs)+'</div>'+
    '<p class="sl">&#128197; Monthly P&amp;L Heatmap</p>'+
    buildSC(tk,bs,si)+
    '<div class="card">'+
    '<div class="ct" id="hmt'+idx+'">'+STRAT_NAMES[as]+' — Monthly P&L</div>'+
    '<div class="cs" id="hms'+idx+'">'+STRAT_DESC[as]+'</div>'+
    '<div id="hm'+idx+'" style="margin-top:12px;overflow-x:auto"></div></div>';
  drawEq(tk,'eq'+idx);
  drawHm(tk,as,'hm'+idx);
}

// ── Optimizer tab ─────────────────────────────────────────
function renderOpt(){
  const c=document.getElementById('tc'+TIER_KEYS.length);
  if(!c||c.dataset.r) return; c.dataset.r='1';
  const ts=OPT_DATA.map(d=>d.time);
  const wrs=OPT_DATA.map(d=>d.win_rate);
  const aps=OPT_DATA.map(d=>d.avg_pnl);
  const mx=Math.max(...wrs);
  c.innerHTML=
    '<p class="sl" style="margin-top:4px">&#9201; Entry Time Optimizer — Simple Trend-Follow (90-min hold)</p>'+
    '<div class="card"><div class="ct">Win Rate by Entry Time</div>'+
    '<div class="cs">Orange = best. Green = 55%+. If price > daily open, buy Call; else buy Put. Exit after 90 min.</div>'+
    '<div class="ow"><canvas id="opt-wr"></canvas></div></div>'+
    '<div class="card"><div class="ct">Average P&L per Trade by Entry Time (10 lots)</div>'+
    '<div class="cs">Positive = profitable on average. Best combo = high win rate + positive avg P&L.</div>'+
    '<div class="ow"><canvas id="opt-pnl"></canvas></div></div>'+
    '<div class="card" style="overflow-x:auto"><div class="ct">Full Optimizer Results</div>'+
    '<table class="mt"><thead><tr><th>Entry Time</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th><th>Total P&L</th><th>Grade</th></tr></thead><tbody>'+
    OPT_DATA.map(d=>'<tr class="'+(d.win_rate===mx?'br':'')+'">'+
      '<td style="font-weight:700;color:'+(d.win_rate===mx?'#ff922b':'#e2e8f0')+'">'+d.time+(d.win_rate===mx?' &#x1F3C6;':'')+'</td>'+
      '<td style="color:#64748b">'+d.trades+'</td>'+
      '<td class="'+(d.win_rate>=55?'pos':d.win_rate>=50?'':'neg')+'">'+d.win_rate+'%</td>'+
      '<td class="'+(d.avg_pnl>=0?'pos':'neg')+'">Rs.'+d.avg_pnl.toLocaleString('en-IN')+'</td>'+
      '<td class="'+(d.total_pnl>=0?'pos':'neg')+'">Rs.'+d.total_pnl.toLocaleString('en-IN')+'</td>'+
      '<td style="font-size:.7rem;font-weight:700;color:'+(d.win_rate>=55?'#10b981':d.win_rate>=50?'#f59e0b':'#ef4444')+'">'+(d.win_rate>=55?'Strong':d.win_rate>=50?'Moderate':'Weak')+'</td>'+
    '</tr>').join('')+
    '</tbody></table></div>';

  dch('opt-wr');
  CR['opt-wr']=new Chart(document.getElementById('opt-wr'),{
    type:'bar',
    data:{labels:ts,datasets:[{label:'Win Rate %',data:wrs,
      backgroundColor:wrs.map(w=>w===mx?'#ff922b':w>=55?'rgba(16,185,129,.7)':w>=50?'rgba(0,212,255,.5)':'rgba(239,68,68,.5)'),
      borderRadius:5}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>'Win Rate: '+c.parsed.y+'%'}}},
      scales:{x:{grid:{color:'#1a2234'},ticks:{color:'#64748b'}},
        y:{min:20,grid:{color:'#1a2234'},ticks:{color:'#64748b',callback:v=>v+'%'}}}}
  });
  dch('opt-pnl');
  CR['opt-pnl']=new Chart(document.getElementById('opt-pnl'),{
    type:'bar',
    data:{labels:ts,datasets:[{label:'Avg P&L',data:aps,
      backgroundColor:aps.map(v=>v>=0?'rgba(16,185,129,.6)':'rgba(239,68,68,.5)'),borderRadius:5}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>'Avg P&L: Rs.'+c.parsed.y.toLocaleString('en-IN')}}},
      scales:{x:{grid:{color:'#1a2234'},ticks:{color:'#64748b'}},
        y:{grid:{color:'#1a2234'},ticks:{color:'#64748b',callback:v=>'Rs.'+v.toLocaleString('en-IN')}}}}
  });
}

// ── Switching ─────────────────────────────────────────────
function switchTier(idx){
  document.querySelectorAll('.tb').forEach((b,i)=>b.classList.toggle('active',i===idx));
  document.querySelectorAll('.tc').forEach((c,i)=>c.classList.toggle('active',i===idx));
  cTier=idx; renderTier(idx);
}
function switchS(ti,si){
  cStrat[ti]=si;
  const tk=TIER_KEYS[ti];
  const strats=Object.keys(STRAT_NAMES).filter(s=>TMETS[tk]&&TMETS[tk][s]);
  const as=strats[si]||strats[0];
  document.querySelectorAll('[id^="sc-'+ti+'-"]').forEach((el,i)=>el.classList.toggle('active',i===si));
  drawHm(tk,as,'hm'+ti);
  const te=document.getElementById('hmt'+ti); if(te) te.textContent=STRAT_NAMES[as]+' - Monthly P&L';
  const se=document.getElementById('hms'+ti); if(se) se.textContent=STRAT_DESC[as]||'';
}
window.addEventListener('load',()=>{
  renderTier(0);
  TIER_KEYS.forEach((_,i)=>{if(i>0) setTimeout(()=>renderTier(i),200*i);});
});
</script>
</body>
</html>"""

def build_dashboard(all_results, all_metrics, all_heats, capital_tiers, opt_data):
    print("\n[>>] Building dashboard...")
    tier_keys=list(capital_tiers.keys())

    def ser(dfs):
        out={}
        for k,df in dfs.items():
            if df is not None and len(df)>0:
                out[k]={"dates":df["date"].dt.strftime("%Y-%m-%d").tolist(),
                        "cum_pnl":df["cum_pnl"].round(0).tolist(),
                        "pnl":df["pnl"].round(0).tolist()}
        return out

    tdata ={t:ser(all_results[t]) for t in tier_keys}
    tmets ={t:all_metrics[t]      for t in tier_keys}
    theats={t:all_heats[t]        for t in tier_keys}

    tmets_pub={t:{s:{k:v for k,v in m.items() if not k.startswith("raw_")}
                  for s,m in d.items() if m} for t,d in tmets.items()}
    tmets_raw={t:{s:{k:v for k,v in m.items() if k.startswith("raw_")}
                  for s,m in d.items() if m} for t,d in tmets.items()}
    ct_js={t:{"capital":v["capital"],"lots":v["lots"]} for t,v in capital_tiers.items()}

    best={}
    for t in tier_keys:
        bs,bsc=None,-999999
        for s,m in all_metrics[t].items():
            if m and "raw_wr" in m:
                sc=m["raw_wr"]*0.4+m["raw_sharpe"]*10*0.3+m["raw_cagr"]*100*0.3
                if sc>bsc: bsc=sc; bs=s
        best[t]=bs

    js = "\n".join([
        f"const TIER_KEYS    = {json.dumps(tier_keys)};",
        f"const STRAT_NAMES  = {json.dumps(STRATEGY_NAMES)};",
        f"const STRAT_DESC   = {json.dumps(STRATEGY_DESC)};",
        f"const SELL_STRATS  = {json.dumps({k:True for k in SELL_STRATS})};",
        f"const DAILY_STRATS = {json.dumps({k:True for k in DAILY_STRATS})};",
        f"const COLORS       = {json.dumps(COLORS)};",
        f"const TDATA        = {json.dumps(tdata)};",
        f"const TMETS        = {json.dumps(tmets_pub)};",
        f"const TMETS_RAW    = {json.dumps(tmets_raw)};",
        f"const THEATS       = {json.dumps(theats)};",
        f"const BEST_TIER    = {json.dumps(best)};",
        f"const CAPITAL_TIERS= {json.dumps(ct_js)};",
        f"const OPT_DATA     = {json.dumps(opt_data)};",
    ])
    return HTML_TEMPLATE.replace("/*__DATA__*/", js)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("="*62)
    print("  Nifty 50 Options Backtester v3 — 15 Strategies")
    print("="*62)

    df5min, daily = load_data()
    opt_data = run_optimizer(df5min, daily, lots=10)

    all_res, all_met, all_heat = {},{},{}

    for tn, tc in CAPITAL_TIERS.items():
        lots=tc["lots"]; cap=tc["capital"]
        print(f"\n{'─'*58}")
        print(f"  Tier: {tn} | Capital: Rs.{cap:,} | Lots: {lots}")
        print(f"{'─'*58}")
        dfs=run_strategies(df5min, daily, lots)
        all_res[tn] =dfs
        all_met[tn] ={s:metrics(df,cap) for s,df in dfs.items()}
        all_heat[tn]={s:heatmap(df)     for s,df in dfs.items()}

    # ── Summary table ──────────────────────────────────────
    print("\n"+"="*62)
    print("  FINAL RESULTS SUMMARY (Medium Tier)")
    print("="*62)
    ref=list(CAPITAL_TIERS.keys())[1]
    rows=[(s,STRATEGY_NAMES[s],
           all_met[ref][s].get("raw_wr",0),
           all_met[ref][s].get("raw_sharpe",0),
           all_met[ref][s].get("raw_pnl",0),
           s in SELL_STRATS)
          for s in STRATEGY_NAMES if all_met[ref].get(s)]
    rows.sort(key=lambda x:x[2],reverse=True)
    print(f"\n  {'Strat':<5} {'Type':<5} {'Name':<26} {'WinRate':>8} {'Sharpe':>8} {'P&L':>16}")
    print(f"  {'─'*75}")
    for s,nm,wr,sh,pnl,isSell in rows:
        tag="[SELL]" if isSell else "[BUY] "
        star=" ***" if isSell and wr>80 else ""
        print(f"  {s:<5} {tag} {nm:<26} {wr:>7.1f}% {sh:>8.2f} Rs.{pnl:>14,.0f}{star}")

    html=build_dashboard(all_res,all_met,all_heat,CAPITAL_TIERS,opt_data)
    with open(OUTPUT_FILE,"w",encoding="utf-8") as f:
        f.write(html)
    print(f"\n[OK] Dashboard: {OUTPUT_FILE}")
    print("     Open in Chrome/Edge for best experience")

if __name__=="__main__":
    main()

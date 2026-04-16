"""
Crypto Trading Assistant v3.2 — Web Edition
Flask backend — tanpa Discord
"""

from flask import Flask, jsonify, render_template, Response
import threading
import queue
import time
import json

app = Flask(__name__)

# ── Import semua logic dari core ──────────────────────────────────────────────
import ccxt
import pandas as pd
import numpy as np
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone

# ══════════════════════════════════════════════
#  KONFIGURASI
# ══════════════════════════════════════════════
CONFIG = {
    "exchange":             "bybit",
    "long_quote":           "USDT",
    "long_timeframe":       "4h",
    "long_candles":         300,
    "long_min_volume":      5_000_000,
    "long_top_n":           10,
    "short_quote":          "USDT",
    "short_timeframe":      "4h",
    "short_candles":        300,
    "short_min_volume":     1_000_000,
    "short_top_n":          10,
    "hot_timeframe":        "1h",
    "hot_candles":          168,
    "hot_min_volume":       500_000,
    "hot_spike_mult":       3.0,
    "hot_min_gain":         5.0,
    "hot_top_n":            10,
    "atr_sl_mult":          1.5,
    "atr_tp1_mult":         2.0,
    "atr_tp2_mult":         3.5,
    "atr_tp3_mult":         5.5,
    "min_rr":               2.0,
    "exclude": ["USDT","BUSD","USDC","DAI","TUSD","FDUSD","USDP","USDD","USDE","FDUSD"],
}

# ══════════════════════════════════════════════
#  INDIKATOR TEKNIKAL
# ══════════════════════════════════════════════
def ema(s, n):   return s.ewm(span=n, adjust=False).mean()
def sma(s, n):   return s.rolling(n).mean()

def calc_rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    return 100 - (100 / (1 + g / l))

def detect_divergence(close, rsi, lb=5):
    if len(close) < lb + 5: return "None"
    c = close.iloc[-(lb+5):]
    r = rsi.iloc[-(lb+5):]
    if c.iloc[-1] < c.min() * 1.02 and r.iloc[-1] > r.min() * 1.05: return "Bullish"
    if c.iloc[-1] > c.max() * 0.98 and r.iloc[-1] < r.max() * 0.95: return "Bearish"
    return "None"

def calc_macd(c, fast=12, slow=26, sig=9):
    m  = ema(c, fast) - ema(c, slow)
    sl = ema(m, sig)
    return m, sl, m - sl

def calc_bb(c, p=20, mult=2.0):
    mid  = sma(c, p)
    std  = c.rolling(p).std()
    up   = mid + mult * std
    lo   = mid - mult * std
    bw   = (up - lo) / mid * 100
    pctb = (c - lo) / (up - lo)
    return up, mid, lo, bw, pctb

def calc_atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def calc_adx(h, l, c, p=14):
    tr   = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    atr  = tr.ewm(span=p, adjust=False).mean()
    pdm  = h.diff().clip(lower=0)
    mdm  = (-l.diff()).clip(lower=0)
    pdm[pdm < mdm] = 0
    mdm[mdm < pdm] = 0
    pdi  = 100 * pdm.ewm(span=p, adjust=False).mean() / atr
    mdi  = 100 * mdm.ewm(span=p, adjust=False).mean() / atr
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(span=p, adjust=False).mean(), pdi, mdi

def calc_stoch_rsi(c, rp=14, sp=14, k=3, d=3):
    rsi   = calc_rsi(c, rp)
    mn    = rsi.rolling(sp).min()
    mx    = rsi.rolling(sp).max()
    stoch = (rsi - mn) / (mx - mn) * 100
    K     = stoch.rolling(k).mean()
    return K, K.rolling(d).mean()

def calc_ichimoku(h, l, c):
    tenkan = (h.rolling(9).max()  + l.rolling(9).min())  / 2
    kijun  = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, span_a, span_b

def calc_obv(c, vol):
    obv = (np.sign(c.diff()).fillna(0) * vol).cumsum()
    obv_e = ema(obv, 20)
    if obv.iloc[-1] > obv_e.iloc[-1] and obv.iloc[-5] < obv_e.iloc[-5]:
        return 2, "OBV Breakout ✓"
    if obv.iloc[-1] > obv_e.iloc[-1]:
        return 1, "OBV Bullish ✓"
    return -1, "OBV Bearish ✗"

def calc_vwap(df):
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).rolling(24).sum() / df["volume"].rolling(24).sum()
    p    = df["close"].iloc[-1]
    v    = vwap.iloc[-1]
    diff = (p - v) / v * 100
    sc   = 2 if p > v * 1.01 else (1 if p > v else -1)
    return sc, f"{'Above' if diff > 0 else 'Below'} VWAP {diff:+.1f}%"

def calc_pivots_advanced(df):
    h = df["high"].iloc[-2]
    l = df["low"].iloc[-2]
    c = df["close"].iloc[-2]
    rng = h - l
    fp = (h + l + c) / 3
    fib_levels = {
        "fib_pivot": fp, "fib_r1": fp + 0.382*rng, "fib_r2": fp + 0.618*rng,
        "fib_r3": fp + 1.000*rng, "fib_s1": fp - 0.382*rng,
        "fib_s2": fp - 0.618*rng, "fib_s3": fp - 1.000*rng,
    }
    cam_levels = {
        "cam_r4": c+rng*(1.1/2), "cam_r3": c+rng*(1.1/4),
        "cam_r2": c+rng*(1.1/6), "cam_r1": c+rng*(1.1/12),
        "cam_s1": c-rng*(1.1/12), "cam_s2": c-rng*(1.1/6),
        "cam_s3": c-rng*(1.1/4), "cam_s4": c-rng*(1.1/2),
    }
    all_levels = {**fib_levels, **cam_levels}
    all_levels["pivot"] = fp
    all_levels["r1"] = fib_levels["fib_r1"]; all_levels["r2"] = fib_levels["fib_r2"]
    all_levels["r3"] = fib_levels["fib_r3"]; all_levels["s1"] = fib_levels["fib_s1"]
    all_levels["s2"] = fib_levels["fib_s2"]; all_levels["s3"] = fib_levels["fib_s3"]
    return all_levels

def detect_market_regime(df):
    h, l, c = df["high"], df["low"], df["close"]
    adx_series, pdi, mdi = calc_adx(h, l, c)
    adx_val = adx_series.iloc[-1]; pdi_val = pdi.iloc[-1]; mdi_val = mdi.iloc[-1]
    atr_series = calc_atr(h, l, c)
    atr_val = atr_series.iloc[-1]; atr_ma = atr_series.rolling(20).mean().iloc[-1]
    atr_ratio = atr_val / atr_ma if atr_ma > 0 else 1.0
    if adx_val >= 30 and pdi_val > mdi_val:
        regime = "Trending Up"; sl_mult = 2.0; tp_mult = 1.15; label = f"Strong Uptrend (ADX {adx_val:.0f})"
    elif adx_val >= 30 and mdi_val > pdi_val:
        regime = "Trending Down"; sl_mult = 2.0; tp_mult = 1.15; label = f"Strong Downtrend (ADX {adx_val:.0f})"
    elif 20 <= adx_val < 30:
        regime = "Mild Trend"; sl_mult = 1.6; tp_mult = 1.0; label = f"Mild Trend (ADX {adx_val:.0f})"
    else:
        regime = "Ranging"; sl_mult = 1.2; tp_mult = 0.85; label = f"Ranging/Sideways (ADX {adx_val:.0f})"
    if atr_ratio >= 1.5: sl_mult += 0.3; label += f" + High Volatility (ATR {atr_ratio:.1f}x)"
    elif atr_ratio <= 0.7: sl_mult = max(sl_mult-0.2, 1.0); label += f" + Low Volatility (ATR {atr_ratio:.1f}x)"
    return {"regime": regime, "sl_mult": round(sl_mult,2), "tp_mult": round(tp_mult,2),
            "adx": round(adx_val,1), "atr_ratio": round(atr_ratio,2), "label": label}

def detect_liquidity_zones(df, price, direction="long", lookback=50):
    h = df["high"].iloc[-lookback:]; l = df["low"].iloc[-lookback:]
    swing_lows = []; swing_highs = []
    for i in range(2, len(l)-2):
        if l.iloc[i] < l.iloc[i-1] and l.iloc[i] < l.iloc[i-2] and l.iloc[i] < l.iloc[i+1] and l.iloc[i] < l.iloc[i+2]:
            swing_lows.append(l.iloc[i])
        if h.iloc[i] > h.iloc[i-1] and h.iloc[i] > h.iloc[i-2] and h.iloc[i] > h.iloc[i+1] and h.iloc[i] > h.iloc[i+2]:
            swing_highs.append(h.iloc[i])
    def nearest_round(p):
        magnitude = max(10**(len(str(int(p)))-2), 1)
        rounds = []
        for mult in [0.5,1,2,5,10]:
            step = magnitude*mult; below = (p//step)*step
            rounds.extend([below, below+step])
        return rounds
    psych_levels = nearest_round(price)
    body = (df["close"]-df["open"]).abs().iloc[-lookback:]
    wick_l = (df["open"].clip(upper=df["close"])-df["low"]).iloc[-lookback:]
    wick_h = (df["high"]-df["close"].clip(lower=df["open"])).iloc[-lookback:]
    avg_body = body.mean()
    wick_support_levels = l[wick_l > 2*avg_body].tolist()
    wick_resist_levels  = h[wick_h > 2*avg_body].tolist()
    if direction == "long":
        candidates = [s for s in swing_lows+wick_support_levels if s < price]
        psych_below = [p for p in psych_levels if p < price*0.999]
        if candidates:
            nearest_swing = max(candidates); liq_aware_sl = nearest_swing * 0.9975
            for r in psych_below:
                if abs(liq_aware_sl-r)/price < 0.003: liq_aware_sl = r*0.995; break
        else: liq_aware_sl = price*0.97
        liq_aware_sl = max(liq_aware_sl, price*0.90); liq_aware_sl = min(liq_aware_sl, price*0.98)
        near_psych = any(abs(liq_aware_sl-r)/price < 0.005 for r in psych_below)
        return liq_aware_sl, near_psych, len(candidates)
    else:
        candidates = [s for s in swing_highs+wick_resist_levels if s > price]
        psych_above = [p for p in psych_levels if p > price*1.001]
        if candidates:
            nearest_swing = min(candidates); liq_aware_sl = nearest_swing*1.0025
            for r in psych_above:
                if abs(liq_aware_sl-r)/price < 0.003: liq_aware_sl = r*1.005; break
        else: liq_aware_sl = price*1.03
        liq_aware_sl = min(liq_aware_sl, price*1.10); liq_aware_sl = max(liq_aware_sl, price*1.02)
        near_psych = any(abs(liq_aware_sl-r)/price < 0.005 for r in psych_above)
        return liq_aware_sl, near_psych, len(candidates)

def detect_candle_patterns(df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c-o).abs(); wu = h-c.clip(lower=o); wl = o.clip(upper=c)-l
    ab = body.rolling(10).mean(); patterns = []; score = 0
    if wl.iloc[-1]>2*body.iloc[-1] and wu.iloc[-1]<body.iloc[-1]*0.5 and c.iloc[-1]>o.iloc[-1]:
        patterns.append("Hammer 🔨"); score += 3
    if (c.iloc[-2]<o.iloc[-2] and c.iloc[-1]>o.iloc[-1] and o.iloc[-1]<c.iloc[-2] and c.iloc[-1]>o.iloc[-2]):
        patterns.append("Bull Engulfing ↑"); score += 4
    if (len(df)>=3 and c.iloc[-3]<o.iloc[-3] and body.iloc[-2]<ab.iloc[-2]*0.5
        and c.iloc[-1]>o.iloc[-1] and c.iloc[-1]>(o.iloc[-3]+c.iloc[-3])/2):
        patterns.append("Morning Star ⭐"); score += 5
    if (wu.iloc[-1]>2*body.iloc[-1] and wl.iloc[-1]<body.iloc[-1]*0.5 and c.iloc[-1]<o.iloc[-1]):
        patterns.append("Shooting Star 🌟"); score -= 3
    if (c.iloc[-2]>o.iloc[-2] and c.iloc[-1]<o.iloc[-1] and o.iloc[-1]>c.iloc[-2] and c.iloc[-1]<o.iloc[-2]):
        patterns.append("Bear Engulfing ↓"); score -= 4
    if (len(df)>=3 and c.iloc[-3]>o.iloc[-3] and body.iloc[-2]<ab.iloc[-2]*0.5
        and c.iloc[-1]<o.iloc[-1] and c.iloc[-1]<(o.iloc[-3]+c.iloc[-3])/2):
        patterns.append("Evening Star 🌙"); score -= 5
    if body.iloc[-1]<ab.iloc[-1]*0.1: patterns.append("Doji ~")
    return score, patterns if patterns else ["No Pattern"]

def _sltp_dict(price, sl, tp1, tp2, tp3, atr, direction, regime, trail_be, trail_tp1, trail_step, near_psych, swing_count, pivots):
    def pct(t): return round((t-price)/price*100, 2)
    def rr(t):  return round(abs(t-price)/max(abs(price-sl),price*0.001), 1)
    return {
        "sl": round(sl,8), "tp1": round(tp1,8), "tp2": round(tp2,8), "tp3": round(tp3,8),
        "sl_pct": pct(sl), "tp1_pct": pct(tp1), "tp2_pct": pct(tp2), "tp3_pct": pct(tp3),
        "rr1": rr(tp1), "rr2": rr(tp2), "rr3": rr(tp3),
        "trail_be": round(trail_be,8), "trail_tp1": round(trail_tp1,8), "trail_step": round(trail_step,8),
        "regime": regime["label"], "liq_warning": near_psych,
        "fib_r1": round(pivots.get("fib_r1",0),8), "fib_s1": round(pivots.get("fib_s1",0),8),
        "cam_r3": round(pivots.get("cam_r3",0),8), "cam_s3": round(pivots.get("cam_s3",0),8),
    }

def calc_sltp_long(price, df, atr_base):
    pivots = calc_pivots_advanced(df); regime = detect_market_regime(df)
    atr = atr_base * regime["sl_mult"]
    cam_s3 = pivots.get("cam_s3", price*0.97); fib_s1 = pivots.get("fib_s1", price*0.97)
    atr_sl = price - atr
    liq_sl, near_psych, swing_count = detect_liquidity_zones(df, price, "long")
    sl_candidates = [atr_sl, liq_sl, cam_s3*0.998, fib_s1*0.998]
    valid_sl = [s for s in sl_candidates if price*0.88 <= s < price*0.985]
    sl = max(valid_sl) if valid_sl else price*0.96
    risk = max(price-sl, price*0.015)
    atr_tp = atr_base * regime["tp_mult"]
    fib_r1 = pivots.get("fib_r1", price*1.04); fib_r2 = pivots.get("fib_r2", price*1.07)
    fib_r3 = pivots.get("fib_r3", price*1.12); cam_r3 = pivots.get("cam_r3", price*1.05)
    cam_r4 = pivots.get("cam_r4", price*1.10)
    tp1 = max(min(fib_r1,cam_r3), price+1.5*risk, price+2.0*atr_tp)
    tp2 = max(max(fib_r2,cam_r4), price+2.5*risk, price+3.5*atr_tp)
    tp3 = max(fib_r3, price+4.0*risk, price+5.5*atr_tp)
    fee_buffer = price*0.0012; trail_be = price+fee_buffer; trail_tp1 = tp1; trail_step = atr_base*0.5
    return _sltp_dict(price, sl, tp1, tp2, tp3, atr_base, "long", regime, trail_be, trail_tp1, trail_step, near_psych, swing_count, pivots)

def calc_sltp_short(price, df, atr_base):
    pivots = calc_pivots_advanced(df); regime = detect_market_regime(df)
    atr = atr_base * regime["sl_mult"]
    cam_r3 = pivots.get("cam_r3", price*1.03); fib_r1 = pivots.get("fib_r1", price*1.04)
    atr_sl = price + atr
    liq_sl, near_psych, swing_count = detect_liquidity_zones(df, price, "short")
    sl_candidates = [atr_sl, liq_sl, cam_r3*1.002, fib_r1*1.002]
    valid_sl = [s for s in sl_candidates if price*1.015 < s <= price*1.12]
    sl = min(valid_sl) if valid_sl else price*1.04
    risk = max(sl-price, price*0.015)
    atr_tp = atr_base * regime["tp_mult"]
    fib_s1 = pivots.get("fib_s1", price*0.96); fib_s2 = pivots.get("fib_s2", price*0.93)
    fib_s3 = pivots.get("fib_s3", price*0.88); cam_s3 = pivots.get("cam_s3", price*0.95)
    cam_s4 = pivots.get("cam_s4", price*0.90)
    tp1 = min(max(fib_s1,cam_s3), price-1.5*risk, price-2.0*atr_tp)
    tp2 = min(min(fib_s2,cam_s4), price-2.5*risk, price-3.5*atr_tp)
    tp3 = min(fib_s3, price-4.0*risk, price-5.5*atr_tp)
    fee_buffer = price*0.0012; trail_be = price-fee_buffer; trail_tp1 = tp1; trail_step = atr_base*0.5
    return _sltp_dict(price, sl, tp1, tp2, tp3, atr_base, "short", regime, trail_be, trail_tp1, trail_step, near_psych, swing_count, pivots)

def calc_volume_poc(df, lookback=50, bins=30):
    try:
        d = df.iloc[-lookback:]; pmin = d["low"].min(); pmax = d["high"].max()
        if pmax <= pmin: return None
        bsize = (pmax-pmin)/bins; edges = [pmin+i*bsize for i in range(bins+1)]
        vols = [0.0]*bins
        for _, row in d.iterrows():
            vol = row["volume"]; clo = row["low"]; chi = row["high"]; cr = chi-clo
            if cr <= 0: continue
            for b in range(bins):
                ol = max(clo,edges[b]); oh = min(chi,edges[b+1])
                if oh > ol: vols[b] += vol*(oh-ol)/cr
        poc_i = vols.index(max(vols)); poc = (edges[poc_i]+edges[poc_i+1])/2
        total = sum(vols); target = total*0.70; acc = vols[poc_i]; lo = poc_i; hi = poc_i
        while acc < target:
            l_v = vols[lo-1] if lo > 0 else 0; h_v = vols[hi+1] if hi < bins-1 else 0
            if l_v >= h_v and lo > 0: lo -= 1; acc += l_v
            elif hi < bins-1: hi += 1; acc += h_v
            else: break
        val = edges[lo]; vah = edges[hi+1]; price = df["close"].iloc[-1]
        dist = (price-poc)/poc*100
        if price > vah:       pos = "Above VAH (extended)"
        elif price < val:     pos = "Below VAL (oversold)"
        elif abs(dist) < 1.0: pos = "At POC (key level)"
        elif price > poc:     pos = "Above POC"
        else:                 pos = "Below POC"
        return {"poc":round(poc,8),"val":round(val,8),"vah":round(vah,8),
                "poc_dist":round(dist,2),"position":pos,"is_near_poc":abs(dist)<2.0}
    except: return None

def check_entry_trigger(df, direction="long"):
    o=df["open"]; h=df["high"]; l=df["low"]; c=df["close"]
    body=(c-o).abs(); ab=body.rolling(10).mean()
    e9=ema(c,9); triggers=[]; ts=0
    if direction == "long":
        if (c.iloc[-2]<o.iloc[-2] and c.iloc[-1]>o.iloc[-1] and o.iloc[-1]<=c.iloc[-2] and c.iloc[-1]>=o.iloc[-2]):
            triggers.append("Bullish Engulfing ← ENTRY SEKARANG"); ts+=3
        wl=o.clip(upper=c)-l; wu=h-c.clip(lower=o)
        if wl.iloc[-1]>2*body.iloc[-1] and wu.iloc[-1]<body.iloc[-1]*0.5 and c.iloc[-1]>o.iloc[-1]:
            triggers.append("Hammer — strong reversal"); ts+=2
        if (c.iloc[-3]<o.iloc[-3] and c.iloc[-2]<o.iloc[-2] and c.iloc[-1]>o.iloc[-1] and body.iloc[-1]>ab.iloc[-1]*1.2):
            triggers.append("3-Bar Reversal — momentum balik"); ts+=2
        if c.iloc[-1]>e9.iloc[-1] and c.iloc[-2]<e9.iloc[-2]:
            triggers.append("EMA9 Cross Up — trend balik"); ts+=2
        if body.iloc[-2]<ab.iloc[-2]*0.15 and c.iloc[-1]>o.iloc[-1]:
            triggers.append("Doji + Bullish Follow-through"); ts+=1
    else:
        if (c.iloc[-2]>o.iloc[-2] and c.iloc[-1]<o.iloc[-1] and o.iloc[-1]>=c.iloc[-2] and c.iloc[-1]<=o.iloc[-2]):
            triggers.append("Bearish Engulfing ← SHORT SEKARANG"); ts+=3
        wu=h-c.clip(lower=o); wl=o.clip(upper=c)-l
        if wu.iloc[-1]>2*body.iloc[-1] and wl.iloc[-1]<body.iloc[-1]*0.5 and c.iloc[-1]<o.iloc[-1]:
            triggers.append("Shooting Star — rejection di resistance"); ts+=2
        if (c.iloc[-3]>o.iloc[-3] and c.iloc[-2]>o.iloc[-2] and c.iloc[-1]<o.iloc[-1] and body.iloc[-1]>ab.iloc[-1]*1.2):
            triggers.append("3-Bar Bearish Reversal"); ts+=2
        if c.iloc[-1]<e9.iloc[-1] and c.iloc[-2]>e9.iloc[-2]:
            triggers.append("EMA9 Cross Down — momentum bearish"); ts+=2
    if not triggers: triggers.append("Belum ada konfirmasi candle — tunggu sinyal")
    if ts >= 3:   rec = "ENTRY SEKARANG — konfirmasi kuat"
    elif ts >= 2: rec = "SIAP ENTRY — ada konfirmasi, risiko moderat"
    elif ts >= 1: rec = "WASPADA — sinyal lemah, tunggu candle berikut"
    else:         rec = "TUNGGU — belum ada konfirmasi candle"
    return {"triggers":triggers,"trigger_score":ts,"recommendation":rec}

def calc_confidence(signals, direction="long"):
    bull_c = bear_c = neutral_c = 0; total = 0
    checks = [
        ("trend", "Uptrend" if direction in ("long","hot") else "Downtrend"),
        ("macd", "Bullish" if direction in ("long","hot") else "Bearish"),
        ("obv", "OBV Bull" if direction in ("long","hot") else "OBV Bear"),
        ("vwap", "Above" if direction in ("long","hot") else "Below"),
        ("stoch_bias", "Bull" if direction in ("long","hot") else "Bear"),
        ("ichimoku", "Bull" if direction in ("long","hot") else "Bear"),
    ]
    rsi_val = signals.get("rsi_raw",50)
    if direction in ("long","hot"):
        if 40 <= rsi_val <= 65: bull_c += 1
        elif rsi_val < 40: bull_c += 1
        else: bear_c += 1
    else:
        if rsi_val > 60: bull_c += 1
        elif rsi_val > 70: bull_c += 2
        else: bear_c += 1
    total += 1
    for key, bull_val in checks:
        v = str(signals.get(key,""))
        if bull_val in v: bull_c += 1
        elif any(x in v for x in ["Bear","Down","Below","Sell"]): bear_c += 1
        else: neutral_c += 1
        total += 1
    pat_score = signals.get("pat_score", 0)
    if pat_score > 0: bull_c += 1
    elif pat_score < 0: bear_c += 1
    else: neutral_c += 1
    total += 1
    adx_v = signals.get("adx_raw",0)
    if adx_v >= 25: bull_c += 1 if direction in ("long","hot") else 0; bear_c += 0 if direction in ("long","hot") else 1
    else: neutral_c += 1
    total += 1
    if direction in ("long","hot"): agree = bull_c; disagree = bear_c
    else: agree = bull_c; disagree = bear_c
    pct = agree/total*100 if total > 0 else 0
    if pct >= 75: level = "HIGH"; icon = "🟢"
    elif pct >= 55: level = "MEDIUM"; icon = "🟡"
    else: level = "LOW"; icon = "🔴"
    return {"level":level,"icon":icon,"pct":round(pct,1),"agree":agree,"disagree":disagree,
            "neutral":neutral_c,"total":total,"label":f"{icon} {level} ({agree}/{total} indikator sepakat)"}

def _make_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    for col in ["open","high","low","close","volume"]: df[col] = pd.to_numeric(df[col])
    return df

def fmt(p):
    if p < 0.001:  return f"${p:.8f}"
    if p < 0.1:    return f"${p:.6f}"
    if p < 1:      return f"${p:.4f}"
    if p < 100:    return f"${p:.4f}"
    if p < 10000:  return f"${p:,.2f}"
    return f"${p:,.1f}"

# ══════════════════════════════════════════════
#  ECONOMIC CALENDAR
# ══════════════════════════════════════════════
def get_economic_calendar():
    events = []; warning_level = "NONE"
    try:
        url = "https://fapi.coinglass.com/api/support/calender/lists"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0", "Accept": "application/json",
            "Referer": "https://www.coinglass.com/",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        now_ts = datetime.now(timezone.utc).timestamp() * 1000
        end_ts = now_ts + 24*3600*1000
        for item in (data.get("data") or []):
            event_ts = item.get("time",0) or item.get("timestamp",0)
            if not (now_ts <= event_ts <= end_ts): continue
            impact = str(item.get("impact","")).upper()
            if impact not in ["HIGH","3","3.0"] and "HIGH" not in impact: continue
            if item.get("currency","USD") not in ["USD","BTC"]: continue
            events.append({"time": datetime.fromtimestamp(event_ts/1000).strftime("%d/%m %H:%M"),
                           "title": item.get("event", item.get("title","Unknown")), "impact": impact})
    except:
        now = datetime.now(); wd = now.weekday(); hr = now.hour
        if wd == 2 and 20 <= hr <= 23:
            events.append({"time":"Malam ini","title":"Kemungkinan FOMC/Fed Event (cek manual)","impact":"HIGH"})
        elif wd == 4 and 18 <= hr <= 22:
            events.append({"time":"Sore ini","title":"Kemungkinan NFP/Jobs Data (cek manual)","impact":"HIGH"})
    if len(events) >= 3:   warning_level = "DANGER"
    elif len(events) >= 2: warning_level = "WARNING"
    elif len(events) >= 1: warning_level = "CAUTION"
    return {"events":events,"warning_level":warning_level}

# ══════════════════════════════════════════════
#  EXCHANGE CONNECTION
# ══════════════════════════════════════════════
def connect_spot():
    ex = ccxt.bybit({"options":{"defaultType":"spot"}})
    ex.load_markets(); return ex

def connect_futures():
    ex = ccxt.bybit({"options":{"defaultType":"linear"}})
    ex.load_markets(); return ex

def fetch_tickers_bybit(ex, market_type="spot"):
    try:
        if market_type == "spot":
            tickers = ex.fetch_tickers()
            return [{"symbol":s,"quoteVolume":v.get("quoteVolume",0) or 0}
                    for s,v in tickers.items() if s.endswith("/USDT")
                    and not any(x in s for x in CONFIG["exclude"])]
        else:
            tickers = ex.fetch_tickers(params={"category":"linear"})
            return [{"symbol":s,"quoteVolume":v.get("quoteVolume",0) or 0}
                    for s,v in tickers.items() if s.endswith("/USDT:USDT")
                    and not any(x in s.replace("/USDT:USDT","") for x in CONFIG["exclude"])]
    except: return []

def get_eligible(tickers, min_vol):
    r = [t for t in tickers if t["quoteVolume"] >= min_vol]
    r.sort(key=lambda x: x["quoteVolume"], reverse=True)
    return r

def get_btc_context(ex):
    try:
        ohlcv = ex.fetch_ohlcv("BTC/USDT", "4h", limit=200)
        df = _make_df(ohlcv)
        c = df["close"]; h = df["high"]; l = df["low"]
        e21 = ema(c,21).iloc[-1]; e50 = ema(c,50).iloc[-1]; e200 = ema(c,200).iloc[-1]
        rsi_v = round(calc_rsi(c).iloc[-1], 1)
        adx_s,_,_ = calc_adx(h,l,c); adx_v = round(adx_s.iloc[-1],1)
        price = c.iloc[-1]
        ch24 = round((price/c.iloc[-7]-1)*100,2); ch7d = round((price/c.iloc[-42]-1)*100,2)
        long_ok = price > e21 > e50 and rsi_v > 45
        short_ok = price < e21 < e50 and rsi_v < 55
        sideways = not long_ok and not short_ok
        if price > e50 > e200: trend = "UPTREND 📈"
        elif price < e50 < e200: trend = "DOWNTREND 📉"
        else: trend = "SIDEWAYS ↔️"
        return {"price":price,"e21":e21,"e50":e50,"e200":e200,"rsi":rsi_v,"adx":adx_v,
                "trend":trend,"ch24":ch24,"ch7d":ch7d,"long_ok":long_ok,"short_ok":short_ok,"sideways":sideways}
    except: return {"price":0,"trend":"N/A","rsi":"N/A","adx":"N/A","e21":0,"e50":0,"e200":0,
                    "ch24":0,"ch7d":0,"long_ok":True,"short_ok":False,"sideways":False}

# ══════════════════════════════════════════════
#  SCORING FUNCTIONS
# ══════════════════════════════════════════════
def score_long(df, symbol):
    try:
        c=df["close"]; h=df["high"]; l=df["low"]; o=df["open"]; vol=df["volume"]
        price=c.iloc[-1]; score=0; signals={}
        e9v=ema(c,9).iloc[-1]; e21v=ema(c,21).iloc[-1]; e50v=ema(c,50).iloc[-1]; e200v=ema(c,200).iloc[-1]
        if price>e21v>e50v: score+=20; signals["trend"]="Uptrend ✓"
        elif price>e21v: score+=10; signals["trend"]="Weak Uptrend"
        else: score-=10; signals["trend"]="Downtrend ✗"
        rsi_s=calc_rsi(c); rsi_v=round(rsi_s.iloc[-1],1); signals["rsi"]=rsi_v; signals["rsi_raw"]=rsi_v
        if 40<=rsi_v<=60: score+=15
        elif rsi_v<40: score+=20
        elif rsi_v>70: score-=15
        macd_l,macd_sig,macd_hist=calc_macd(c)
        if macd_hist.iloc[-1]>0 and macd_hist.iloc[-2]<=0: score+=15; signals["macd"]="Bullish Cross ✓"
        elif macd_hist.iloc[-1]>0: score+=8; signals["macd"]="Bullish ✓"
        else: score-=5; signals["macd"]="Bearish ✗"
        adx_s,pdi,mdi=calc_adx(h,l,c); adx_v=round(adx_s.iloc[-1],1)
        signals["adx"]=adx_v; signals["adx_raw"]=adx_v
        signals["adx_bias"]="Bull" if pdi.iloc[-1]>mdi.iloc[-1] else "Bear"
        if adx_v>25 and pdi.iloc[-1]>mdi.iloc[-1]: score+=10
        up,mid,lo,bw,pctb=calc_bb(c)
        if pctb.iloc[-1]<0.3: score+=10
        elif pctb.iloc[-1]>0.8: score-=5
        stoch_k,stoch_d=calc_stoch_rsi(c)
        sk=round(stoch_k.iloc[-1],1); sd=round(stoch_d.iloc[-1],1)
        signals["stoch_k"]=sk
        if sk<20: score+=10; signals["stoch_bias"]="Bull"
        elif sk>80: score-=10; signals["stoch_bias"]="Bear"
        else: signals["stoch_bias"]="Neutral"
        ten,kij,spa,spb=calc_ichimoku(h,l,c)
        if price>spa.iloc[-1] and price>spb.iloc[-1]: score+=10; signals["ichimoku"]="Bull Cloud ✓"
        elif price<spa.iloc[-1] and price<spb.iloc[-1]: score-=10; signals["ichimoku"]="Bear Cloud ✗"
        else: signals["ichimoku"]="In Cloud"
        obv_sc,obv_lbl=calc_obv(c,vol); score+=obv_sc*3; signals["obv"]=obv_lbl
        vwap_sc,vwap_lbl=calc_vwap(df); score+=vwap_sc*2; signals["vwap"]=vwap_lbl
        vol_ma=vol.rolling(20).mean(); vol_ratio=round(vol.iloc[-1]/vol_ma.iloc[-1],2) if vol_ma.iloc[-1]>0 else 1
        signals["vol_ratio"]=vol_ratio
        if vol_ratio>1.5: score+=5
        pat_sc,patterns=detect_candle_patterns(df); score+=pat_sc; signals["patterns"]=patterns; signals["pat_score"]=pat_sc
        golden=ema(c,50).iloc[-1]<ema(c,200).iloc[-1] and ema(c,50).iloc[-2]>=ema(c,200).iloc[-2]
        if golden: score+=15; signals["golden"]=True
        div=detect_divergence(c,calc_rsi(c)); signals["div"]=div
        if div=="Bullish": score+=10
        atr_base=calc_atr(h,l,c).iloc[-1]
        sltp=calc_sltp_long(price,df,atr_base)
        if sltp["rr2"]<CONFIG["min_rr"]: return None
        pivots=calc_pivots_advanced(df)
        e21_v=ema(c,21).iloc[-1]; recent_high=h.iloc[-20:].max(); recent_low=l.iloc[-20:].min()
        fib_range=recent_high-recent_low; fib_382=recent_high-0.382*fib_range; fib_618=recent_high-0.618*fib_range
        ideal_low=min(e21_v,fib_382)*0.998; ideal_high=max(e21_v,fib_382)*1.002
        skip_above=ideal_high*1.05; entry_max=price
        if ideal_low<=price<=ideal_high: ez_status="✅ IDEAL — Harga di zona entry"
        elif price<ideal_low: ez_status="⏳ BELUM — Tunggu koreksi lebih dalam"
        elif price<=skip_above: ez_status="⚠️ TERLAMBAT — Harga di atas zona ideal"
        else: ez_status="🚫 SKIP — Harga terlalu jauh di atas zona"
        entry_zone={"status":ez_status,"ideal_low":round(ideal_low,8),"ideal_high":round(ideal_high,8),
                    "ideal_pct_low":round((ideal_low-price)/price*100,2),"ideal_pct_high":round((ideal_high-price)/price*100,2),
                    "skip_above":round(skip_above,8),"skip_pct":round((skip_above-price)/price*100,2),
                    "entry_max":round(entry_max,8),"e21":round(e21_v,8),"fib_382":round(fib_382,8),
                    "fib_618":round(fib_618,8),"e9":round(e9v,8),"direction":"long"}
        volume_poc=calc_volume_poc(df); entry_trigger=check_entry_trigger(df,"long")
        score=max(0,min(100,score)); prob=min(95,max(5,int(score*0.85+10)))
        if score>=70: action="STRONG BUY"
        elif score>=55: action="BUY"
        elif score>=40: action="WATCH"
        elif score>=25: action="CAUTION"
        else: action="AVOID"
        ticker=None
        try: ticker=connect_spot().fetch_ticker(symbol)
        except: pass
        ch24=round(ticker["percentage"],2) if ticker and ticker.get("percentage") else round((price/c.iloc[-7]-1)*100,2)
        confidence=calc_confidence(signals,"long")
        return {"symbol":symbol,"price":price,"score":score,"probability":prob,"action":action,
                "change_24h":ch24,"signals":signals,"sltp":sltp,"pivots":pivots,"entry_zone":entry_zone,
                "volume_poc":volume_poc,"entry_trigger":entry_trigger,"confidence":confidence}
    except: return None

def score_short(df, symbol):
    try:
        c=df["close"]; h=df["high"]; l=df["low"]; o=df["open"]; vol=df["volume"]
        price=c.iloc[-1]; score=0; signals={}
        e21v=ema(c,21).iloc[-1]; e50v=ema(c,50).iloc[-1]; e200v=ema(c,200).iloc[-1]
        if price<e21v<e50v: score+=20; signals["trend"]="Downtrend ✓"
        elif price<e21v: score+=10; signals["trend"]="Weak Downtrend"
        else: score-=10; signals["trend"]="Uptrend ✗"
        rsi_s=calc_rsi(c); rsi_v=round(rsi_s.iloc[-1],1); signals["rsi"]=rsi_v; signals["rsi_raw"]=rsi_v
        if rsi_v>60: score+=15
        elif rsi_v>70: score+=20
        elif rsi_v<40: score-=15
        macd_l,macd_sig,macd_hist=calc_macd(c)
        if macd_hist.iloc[-1]<0 and macd_hist.iloc[-2]>=0: score+=15; signals["macd"]="Bearish Cross ✓"
        elif macd_hist.iloc[-1]<0: score+=8; signals["macd"]="Bearish ✓"
        else: score-=5; signals["macd"]="Bullish ✗"
        adx_s,pdi,mdi=calc_adx(h,l,c); adx_v=round(adx_s.iloc[-1],1)
        signals["adx"]=adx_v; signals["adx_raw"]=adx_v
        signals["adx_bias"]="Bear" if mdi.iloc[-1]>pdi.iloc[-1] else "Bull"
        if adx_v>25 and mdi.iloc[-1]>pdi.iloc[-1]: score+=10
        up,mid,lo,bw,pctb=calc_bb(c)
        if pctb.iloc[-1]>0.8: score+=10
        elif pctb.iloc[-1]<0.2: score-=5
        stoch_k,stoch_d=calc_stoch_rsi(c)
        sk=round(stoch_k.iloc[-1],1); signals["stoch_k"]=sk
        if sk>80: score+=10; signals["stoch_bias"]="Bear"
        elif sk<20: score-=10; signals["stoch_bias"]="Bull"
        else: signals["stoch_bias"]="Neutral"
        ten,kij,spa,spb=calc_ichimoku(h,l,c)
        if price<spa.iloc[-1] and price<spb.iloc[-1]: score+=10; signals["ichimoku"]="Bear Cloud ✓"
        elif price>spa.iloc[-1] and price>spb.iloc[-1]: score-=10; signals["ichimoku"]="Bull Cloud ✗"
        else: signals["ichimoku"]="In Cloud"
        obv_sc,obv_lbl=calc_obv(c,vol); score-=obv_sc*3; signals["obv"]=obv_lbl
        vwap_sc,vwap_lbl=calc_vwap(df); score-=vwap_sc*2; signals["vwap"]=vwap_lbl
        pat_sc,patterns=detect_candle_patterns(df); score-=pat_sc; signals["patterns"]=patterns; signals["pat_score"]=pat_sc
        death=ema(c,50).iloc[-1]>ema(c,200).iloc[-1] and ema(c,50).iloc[-2]<=ema(c,200).iloc[-2]
        if death: score+=15; signals["death"]=True
        div=detect_divergence(c,calc_rsi(c)); signals["div"]=div
        if div=="Bearish": score+=10
        atr_base=calc_atr(h,l,c).iloc[-1]
        sltp=calc_sltp_short(price,df,atr_base)
        if sltp["rr2"]<CONFIG["min_rr"]: return None
        pivots=calc_pivots_advanced(df)
        e21_v=ema(c,21).iloc[-1]; recent_high=h.iloc[-20:].max(); recent_low=l.iloc[-20:].min()
        fib_range=recent_high-recent_low; fib_382=recent_low+0.382*fib_range; fib_618=recent_low+0.618*fib_range
        ideal_low=min(e21_v,fib_382)*0.998; ideal_high=max(e21_v,fib_382)*1.002
        skip_below=ideal_low*0.95; entry_max=price
        if ideal_low<=price<=ideal_high: ez_status="✅ IDEAL — Zona short ideal"
        elif price>ideal_high: ez_status="⏳ BELUM — Tunggu bounce lebih tinggi"
        elif price>=skip_below: ez_status="⚠️ TERLAMBAT — Harga sudah turun terlalu jauh"
        else: ez_status="🚫 SKIP — Terlalu jauh dari zona short"
        entry_zone={"status":ez_status,"ideal_low":round(ideal_low,8),"ideal_high":round(ideal_high,8),
                    "ideal_pct_low":round((ideal_low-price)/price*100,2),"ideal_pct_high":round((ideal_high-price)/price*100,2),
                    "skip_below":round(skip_below,8),"skip_pct":round((skip_below-price)/price*100,2),
                    "entry_max":round(entry_max,8),"e21":round(e21_v,8),"fib_382":round(fib_382,8),
                    "fib_618":round(fib_618,8),"direction":"short"}
        volume_poc=calc_volume_poc(df); entry_trigger=check_entry_trigger(df,"short")
        score=max(0,min(100,score)); prob=min(95,max(5,int(score*0.85+10)))
        if score>=70: action="STRONG SHORT"
        elif score>=55: action="SHORT"
        elif score>=40: action="WATCH SHORT"
        else: action="NO SHORT"
        ch24=round((price/c.iloc[-7]-1)*100,2)
        confidence=calc_confidence(signals,"short")
        return {"symbol":symbol,"price":price,"score":score,"probability":prob,"action":action,
                "change_24h":ch24,"signals":signals,"sltp":sltp,"pivots":pivots,"entry_zone":entry_zone,
                "volume_poc":volume_poc,"entry_trigger":entry_trigger,"confidence":confidence}
    except: return None

def score_hot(df, symbol, avg_vol_7d):
    try:
        c=df["close"]; h=df["high"]; l=df["low"]; vol=df["volume"]
        price=c.iloc[-1]
        recent_vol=vol.iloc[-3:].mean()
        spike_mult=round(recent_vol/avg_vol_7d,1) if avg_vol_7d>0 else 0
        if spike_mult<CONFIG["hot_spike_mult"]: return None
        ch24=round((price/c.iloc[-25]-1)*100,2) if len(c)>25 else 0
        ch1h=round((price/c.iloc[-2]-1)*100,2) if len(c)>2 else 0
        if ch24<CONFIG["hot_min_gain"]: return None
        score=0; signals={}
        rsi_v=round(calc_rsi(c).iloc[-1],1); signals["rsi"]=rsi_v; signals["rsi_raw"]=rsi_v
        macd_l,macd_sig,macd_hist=calc_macd(c)
        if macd_hist.iloc[-1]>0: score+=20; signals["macd"]="Bullish ✓"
        else: signals["macd"]="Bearish ✗"
        e9v=ema(c,9).iloc[-1]; e21v=ema(c,21).iloc[-1]
        if price>e9v>e21v: score+=20; signals["trend"]="Strong Momentum ✓"
        elif price>e9v: score+=10; signals["trend"]="Momentum ✓"
        else: signals["trend"]="Weak"
        hh=h.iloc[-20:].max(); ll=l.iloc[-20:].min()
        if price>hh*0.98: score+=20; signals["breakout"]="Near 20-bar High 🚀"
        elif price>hh*0.95: score+=10; signals["breakout"]="Approaching High"
        else: signals["breakout"]="No Breakout"
        vol_ma=vol.rolling(168).mean()
        vol_ratio=round(vol.iloc[-1]/vol_ma.iloc[-1],1) if vol_ma.iloc[-1]>0 else 1
        if vol_ratio>5: score+=20
        elif vol_ratio>3: score+=10
        if rsi_v>80: score-=15; signals["risk"]="RSI Overbought — risiko reversal tinggi ⚠️"
        elif rsi_v>70: score-=5; signals["risk"]="RSI tinggi — waspadai"
        else: signals["risk"]="RSI normal"
        pat_sc,patterns=detect_candle_patterns(df); score+=max(0,pat_sc); signals["patterns"]=patterns; signals["pat_score"]=pat_sc
        atr_base=calc_atr(h,l,c).iloc[-1]
        sltp=calc_sltp_long(price,df,atr_base)
        if not sltp or sltp["rr2"]<1.5: return None
        e21_v=ema(c,21).iloc[-1]; recent_high=h.iloc[-20:].max(); recent_low=l.iloc[-20:].min()
        fib_range=recent_high-recent_low; fib_382=recent_high-0.382*fib_range
        ideal_low=min(e21_v,fib_382)*0.998; ideal_high=max(e21_v,fib_382)*1.002
        skip_above=ideal_high*1.05
        if ideal_low<=price<=ideal_high: ez_status="✅ IDEAL"
        elif price<ideal_low: ez_status="⏳ BELUM"
        elif price<=skip_above: ez_status="⚠️ TERLAMBAT"
        else: ez_status="🚫 SKIP"
        entry_zone={"status":ez_status,"ideal_low":round(ideal_low,8),"ideal_high":round(ideal_high,8),
                    "ideal_pct_low":round((ideal_low-price)/price*100,2),"ideal_pct_high":round((ideal_high-price)/price*100,2),
                    "e21":round(e21_v,8),"fib_382":round(fib_382,8),"direction":"hot"}
        volume_poc=calc_volume_poc(df); entry_trigger=check_entry_trigger(df,"long")
        score=max(0,min(100,score)); prob=min(95,max(5,int(score*0.80+15)))
        if score>=70: action="🔥 STRONG MOMENTUM"
        elif score>=50: action="🔥 MOMENTUM"
        elif score>=35: action="👀 WATCH HOT"
        else: action="❄️ FADING"
        confidence=calc_confidence(signals,"hot")
        return {"symbol":symbol,"price":price,"score":score,"probability":prob,"action":action,
                "change_24h":ch24,"change_1h":ch1h,"spike_mult":spike_mult,"signals":signals,
                "sltp":sltp,"entry_zone":entry_zone,"volume_poc":volume_poc,"entry_trigger":entry_trigger,"confidence":confidence}
    except: return None

# ══════════════════════════════════════════════
#  SCAN RUNNERS
# ══════════════════════════════════════════════
def run_long(progress_cb=None):
    try:
        ex = connect_spot()
        tickers = fetch_tickers_bybit(ex, "spot")
    except Exception as e:
        return {"error": str(e), "results": []}
    btc = get_btc_context(ex)
    cal = get_economic_calendar()
    eligible = get_eligible(tickers, CONFIG["long_min_volume"])[:100]
    results = []
    for i, mkt in enumerate(eligible):
        if progress_cb: progress_cb(i, len(eligible), mkt["symbol"])
        try:
            ohlcv = ex.fetch_ohlcv(mkt["symbol"], CONFIG["long_timeframe"], limit=CONFIG["long_candles"])
            if len(ohlcv) < 150: continue
            df = _make_df(ohlcv)
            r = score_long(df, mkt["symbol"])
            if r:
                r["btc"] = btc
                r["_cal"] = cal
                r["_btc"] = btc
                results.append(r)
        except: pass
        time.sleep(0.05)
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"btc": btc, "cal": cal, "results": results[:CONFIG["long_top_n"]]}

def run_short(progress_cb=None):
    try:
        ex = connect_futures()
        ex_spot = connect_spot()
        tickers = fetch_tickers_bybit(ex, "linear")
    except Exception as e:
        return {"error": str(e), "results": []}
    btc = get_btc_context(ex_spot)
    cal = get_economic_calendar()
    eligible = get_eligible(tickers, CONFIG["short_min_volume"])[:80]
    results = []
    for i, mkt in enumerate(eligible):
        if progress_cb: progress_cb(i, len(eligible), mkt["symbol"])
        try:
            ohlcv = ex.fetch_ohlcv(mkt["symbol"], CONFIG["short_timeframe"],
                                   limit=CONFIG["short_candles"], params={"category":"linear"})
            if len(ohlcv) < 150: continue
            df = _make_df(ohlcv)
            r = score_short(df, mkt["symbol"])
            if r:
                r["btc"] = btc
                r["_cal"] = cal
                r["_btc"] = btc
                results.append(r)
        except: pass
        time.sleep(0.05)
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"btc": btc, "cal": cal, "results": results[:CONFIG["short_top_n"]]}

def run_hot(progress_cb=None):
    try:
        ex = connect_spot()
        tickers = fetch_tickers_bybit(ex, "spot")
    except Exception as e:
        return {"error": str(e), "results": []}
    cal = get_economic_calendar()
    eligible = get_eligible(tickers, CONFIG["hot_min_volume"])[:150]
    results = []
    for i, mkt in enumerate(eligible):
        if progress_cb: progress_cb(i, len(eligible), mkt["symbol"])
        try:
            ohlcv = ex.fetch_ohlcv(mkt["symbol"], CONFIG["hot_timeframe"], limit=CONFIG["hot_candles"])
            if len(ohlcv) < 100: continue
            df = _make_df(ohlcv)
            vol = df["volume"]
            avg_vol_7d = vol.iloc[-168:].mean() if len(vol) >= 168 else vol.mean()
            r = score_hot(df, mkt["symbol"], avg_vol_7d)
            if r:
                r["_cal"] = cal
                results.append(r)
        except: pass
        time.sleep(0.05)
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"cal": cal, "results": results[:CONFIG["hot_top_n"]]}

# ══════════════════════════════════════════════
#  SSE SCAN STATE
# ══════════════════════════════════════════════
scan_state = {
    "running": False,
    "mode": None,
    "progress": 0,
    "total": 0,
    "current_symbol": "",
    "result": None,
    "error": None,
}
scan_lock = threading.Lock()

def run_scan_thread(mode):
    global scan_state
    with scan_lock:
        scan_state["running"] = True
        scan_state["progress"] = 0
        scan_state["total"] = 0
        scan_state["result"] = None
        scan_state["error"] = None
        scan_state["mode"] = mode

    def progress_cb(i, total, symbol):
        with scan_lock:
            scan_state["progress"] = i + 1
            scan_state["total"] = total
            scan_state["current_symbol"] = symbol

    try:
        if mode == "long":
            result = run_long(progress_cb)
        elif mode == "short":
            result = run_short(progress_cb)
        elif mode == "hot":
            result = run_hot(progress_cb)
        elif mode == "all":
            r_long  = run_long(progress_cb)
            r_short = run_short(progress_cb)
            r_hot   = run_hot(progress_cb)
            result  = {"long": r_long, "short": r_short, "hot": r_hot}
        else:
            result = {"error": "Mode tidak dikenal"}

        # Sanitize result for JSON serialization
        result = sanitize(result)
        with scan_lock:
            scan_state["result"] = result
    except Exception as e:
        with scan_lock:
            scan_state["error"] = str(e)
    finally:
        with scan_lock:
            scan_state["running"] = False

def sanitize(obj):
    """Convert numpy types to Python native for JSON."""
    import math
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.ndarray,)): return sanitize(obj.tolist())
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj

# ══════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/scan/<mode>", methods=["POST"])
def start_scan(mode):
    if mode not in ("long","short","hot","all"):
        return jsonify({"error": "Mode tidak valid"}), 400
    with scan_lock:
        if scan_state["running"]:
            return jsonify({"error": "Scan sedang berjalan"}), 409
    t = threading.Thread(target=run_scan_thread, args=(mode,), daemon=True)
    t.start()
    return jsonify({"status": "started", "mode": mode})

@app.route("/api/status")
def get_status():
    with scan_lock:
        return jsonify(dict(scan_state))

@app.route("/api/result")
def get_result():
    with scan_lock:
        if scan_state["running"]:
            return jsonify({"status": "running"})
        if scan_state["error"]:
            return jsonify({"status": "error", "error": scan_state["error"]})
        if scan_state["result"] is not None:
            return jsonify({"status": "done", "result": scan_state["result"], "mode": scan_state["mode"]})
        return jsonify({"status": "idle"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

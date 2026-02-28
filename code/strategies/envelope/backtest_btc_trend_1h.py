
import argparse
import pandas as pd
import numpy as np
import ccxt
from pathlib import Path

def fetch(symbol, timeframe, start, end):
    ex = ccxt.bitget({"enableRateLimit": True})
    since = ex.parse8601(start + "T00:00:00Z")
    end_ts = ex.parse8601(end + "T00:00:00Z")
    all_data = []
    while since < end_ts:
        data = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=200)
        if not data:
            break
        all_data += data
        since = data[-1][0] + 1
    df = pd.DataFrame(all_data, columns=["timestamp","open","high","low","close","volume"])
    return df

def backtest(df):
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()

    balance = 1000
    pos = None
    entry = 0

    for i in range(200, len(df)):
        row = df.iloc[i]

        if pos is None:
            if row["ema20"] > row["ema200"] and row["close"] <= row["ema20"]:
                pos = "long"
                entry = row["close"]
            elif row["ema20"] < row["ema200"] and row["close"] >= row["ema20"]:
                pos = "short"
                entry = row["close"]
        else:
            atr = row["atr"]
            if pos == "long":
                if row["close"] >= entry + 2*atr:
                    balance += (row["close"]-entry)
                    pos=None
                elif row["close"] <= entry - 1.5*atr:
                    balance += (row["close"]-entry)
                    pos=None
            else:
                if row["close"] <= entry - 2*atr:
                    balance += (entry-row["close"])
                    pos=None
                elif row["close"] >= entry + 1.5*atr:
                    balance += (entry-row["close"])
                    pos=None
    return balance

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    df = fetch("BTC/USDT", "1h", args.start, args.end)
    result = backtest(df)
    print("Final balance:", result)

# ============================================================
# data.py — BTC NextGen | Multi-Source Data Pipeline
# Supports: yfinance, Binance API, CSV, CoinGecko
# ============================================================

import numpy as np
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

# ── Default source ─────────────────────────────────────────
# Options: "yfinance" | "binance" | "coingecko" | "csv"
DEFAULT_SOURCE = "yfinance"
CSV_PATH = "btc_prices.csv"   # used only when source="csv"


# ────────────────────────────────────────────────────────────
# LIVE PRICE — Binance real-time ticker
# yfinance se alag — yeh actual live market price hai
# ────────────────────────────────────────────────────────────
def fetch_live_price() -> dict:
    """
    Binance se REAL-TIME BTC/USDT price fetch karo.
    Har second update hota hai — daily close nahi, live market price.
    No API key needed.

    Returns dict:
        price, change_24h, change_pct_24h,
        high_24h, low_24h, volume_24h, timestamp
    """
    try:
        url  = "https://api.binance.com/api/v3/ticker/24hr"
        resp = requests.get(url, params={"symbol": "BTCUSDT"}, timeout=8)
        resp.raise_for_status()
        d = resp.json()

        price      = float(d["lastPrice"])
        open_price = float(d["openPrice"])
        change     = price - open_price

        return {
            "price":          round(price, 2),
            "change_24h":     round(change, 2),
            "change_pct_24h": round(float(d["priceChangePercent"]), 2),
            "high_24h":       round(float(d["highPrice"]), 2),
            "low_24h":        round(float(d["lowPrice"]), 2),
            "volume_24h":     round(float(d["volume"]), 2),
            "source":         "Binance Live",
            "timestamp":      datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    except Exception as e:
        # Fallback: CoinGecko
        try:
            url2  = "https://api.coingecko.com/api/v3/simple/price"
            resp2 = requests.get(url2, timeout=8,
                                 params={"ids": "bitcoin", "vs_currencies": "usd",
                                         "include_24hr_change": "true"})
            resp2.raise_for_status()
            d2 = resp2.json()["bitcoin"]
            return {
                "price":          round(float(d2["usd"]), 2),
                "change_pct_24h": round(float(d2.get("usd_24h_change", 0)), 2),
                "change_24h":     None,
                "high_24h":       None,
                "low_24h":        None,
                "volume_24h":     None,
                "source":         "CoinGecko fallback",
                "timestamp":      datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
        except Exception as e2:
            raise RuntimeError(f"Live price failed: {e} | fallback: {e2}")


# ────────────────────────────────────────────────────────────
# 1. yfinance  (FREE, no API key, most reliable)
# ────────────────────────────────────────────────────────────
def fetch_yfinance(days: int = 365) -> np.ndarray:
    """
    Fetch BTC-USD daily close prices via yfinance.
    Install: pip install yfinance
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("BTC-USD")
        df = ticker.history(period=f"{days}d", interval="1d")
        if df.empty:
            raise ValueError("yfinance returned empty dataframe")
        prices = df["Close"].dropna().values.astype(np.float32)
        print(f"[yfinance] Fetched {len(prices)} daily prices")
        return prices
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")


# ────────────────────────────────────────────────────────────
# 2. Binance Public API (FREE, no API key needed)
# ────────────────────────────────────────────────────────────
def fetch_binance(days: int = 365) -> np.ndarray:
    """
    Fetch BTCUSDT daily klines from Binance public API.
    No registration or API key required.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": min(days, 1000)  # Binance max per request
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # index 4 = close price in kline response
    prices = np.array([float(candle[4]) for candle in data], dtype=np.float32)
    print(f"[Binance] Fetched {len(prices)} daily prices")
    return prices


# ────────────────────────────────────────────────────────────
# 3. CoinGecko API (FREE tier, rate-limited)
# ────────────────────────────────────────────────────────────
def fetch_coingecko(days: int = 365) -> np.ndarray:
    """
    Fetch BTC/USD prices from CoinGecko public API.
    Rate limit: ~10-50 calls/min on free tier.
    """
    HEADERS = {
        "User-Agent": "btc-nextgen/2.0 (contact: dev@example.com)",
        "Accept": "application/json"
    }
    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        f"?vs_currency=usd&days={days}&interval=daily"
    )
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 429:
        raise ValueError("CoinGecko rate limit hit. Switch source to 'binance' or 'yfinance'.")
    resp.raise_for_status()
    data = resp.json()
    if "prices" not in data:
        raise ValueError("Invalid CoinGecko response: 'prices' key missing")
    prices = np.array([p[1] for p in data["prices"]], dtype=np.float32)
    print(f"[CoinGecko] Fetched {len(prices)} daily prices")
    return prices


# ────────────────────────────────────────────────────────────
# 4. Local CSV File (completely offline)
# ────────────────────────────────────────────────────────────
def fetch_csv(days: int = 365, path: str = CSV_PATH) -> np.ndarray:
    """
    Load BTC prices from a local CSV file.
    CSV must have a 'Close' (or 'close') column.
    Download free historical data from: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found at '{path}'. "
            "Download from Kaggle: 'Bitcoin Historical Data' and place as btc_prices.csv"
        )
    df = pd.read_csv(path)

    # Auto-detect close column name
    close_col = None
    for col in df.columns:
        if col.strip().lower() == "close":
            close_col = col
            break
    if close_col is None:
        raise ValueError(f"No 'Close' column found. Available: {list(df.columns)}")

    df = df.dropna(subset=[close_col])
    prices = df[close_col].tail(days).values.astype(np.float32)
    print(f"[CSV] Loaded {len(prices)} prices from {path}")
    return prices


# ────────────────────────────────────────────────────────────
# MAIN FUNCTION — auto-fallback chain
# ────────────────────────────────────────────────────────────
def fetch_data(days: int = 365, source: str = DEFAULT_SOURCE) -> np.ndarray:
    """
    Unified data fetcher with automatic fallback.

    Priority order (if source='auto'):
      yfinance → Binance → CoinGecko → CSV

    Args:
        days:   Number of daily candles to fetch
        source: 'yfinance' | 'binance' | 'coingecko' | 'csv' | 'auto'

    Returns:
        np.ndarray of float32 close prices, shape (N,)
    """
    fetchers = {
        "yfinance":   fetch_yfinance,
        "binance":    fetch_binance,
        "coingecko":  fetch_coingecko,
        "csv":        fetch_csv,
    }

    if source in fetchers:
        return fetchers[source](days)

    if source == "auto":
        for name, fn in fetchers.items():
            try:
                return fn(days)
            except Exception as e:
                print(f"[auto-fallback] {name} failed: {e}")
        raise RuntimeError("All data sources failed. Check internet connection.")

    raise ValueError(f"Unknown source '{source}'. Choose: {list(fetchers.keys()) + ['auto']}")


# ────────────────────────────────────────────────────────────
# OHLCV fetch (for candlestick charts in dashboard)
# ────────────────────────────────────────────────────────────
def fetch_ohlcv(days: int = 90, source: str = "binance") -> pd.DataFrame:
    """
    Returns OHLCV DataFrame with columns: date, open, high, low, close, volume
    Used for candlestick charting in the dashboard.
    """
    if source == "binance":
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "1d", "limit": min(days, 1000)}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(raw, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
        ])
        df["date"]   = pd.to_datetime(df["open_time"], unit="ms")
        df["open"]   = df["open"].astype(float)
        df["high"]   = df["high"].astype(float)
        df["low"]    = df["low"].astype(float)
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df[["date","open","high","low","close","volume"]].reset_index(drop=True)

    elif source == "yfinance":
        import yfinance as yf
        ticker = yf.Ticker("BTC-USD")
        df = ticker.history(period=f"{days}d", interval="1d").reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"index": "date"})
        return df[["date","open","high","low","close","volume"]].dropna()

    else:
        # fallback: return close-only DataFrame
        prices = fetch_data(days, source)
        dates = pd.date_range(end=datetime.today(), periods=len(prices), freq="D")
        return pd.DataFrame({"date": dates, "close": prices})


# ────────────────────────────────────────────────────────────
# Quick test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing data sources...")
    try:
        p = fetch_data(90, source="yfinance")
        print(f"  yfinance OK — shape: {p.shape}, latest: ${p[-1]:,.2f}")
    except Exception as e:
        print(f"  yfinance FAILED: {e}")

    try:
        p = fetch_data(90, source="binance")
        print(f"  Binance  OK — shape: {p.shape}, latest: ${p[-1]:,.2f}")
    except Exception as e:
        print(f"  Binance  FAILED: {e}")
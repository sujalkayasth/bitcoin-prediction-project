# ============================================================
# main.py — BTC NextGen | FastAPI Backend
# Endpoints: /predict, /historical, /backtest, /benchmark, /health
# ============================================================

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
import os, json, traceback, time, math, statistics
import joblib
import psutil

# MongoDB integration
from database import (
    get_db, is_connected, get_dashboard_stats,
    prediction_save, prediction_history, prediction_signal_stats, prediction_accuracy_trend,
    backtest_save, backtest_history, backtest_best,
    benchmark_save, benchmark_history,
)

# ────────────────────────────────────────────────────────────
# App setup
# ────────────────────────────────────────────────────────────
app = FastAPI(
    title="BTC NextGen Forecast API",
    description="Industry-grade Bitcoin price forecasting with Transformer models",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────
# PATHS (auto-detect project root)
# ────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "transformer.keras")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

SEQ_LEN           = 60
DEFAULT_THRESHOLD = 150.0   # USD difference to trigger BUY/SELL

# Global model state
model  = None
scaler = None

# ────────────────────────────────────────────────────────────
# Custom Keras objects registration
# ────────────────────────────────────────────────────────────
def get_custom_objects():
    """Returns dict of custom Keras layers needed for model loading."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers

        @tf.keras.utils.register_keras_serializable(package="BTCNextGen")
        class PositionalEncoding(layers.Layer):
            def __init__(self, max_len=5000, d_model=64, **kwargs):
                super().__init__(**kwargs)
                self.max_len = max_len
                self.d_model = d_model
            def build(self, input_shape):
                position = np.arange(self.max_len)[:, np.newaxis]
                div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
                pe = np.zeros((self.max_len, self.d_model))
                pe[:, 0::2] = np.sin(position * div_term)
                pe[:, 1::2] = np.cos(position * div_term)
                self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)
                super().build(input_shape)
            def call(self, x):
                return x + self.pe[:, :tf.shape(x)[1], :]
            def get_config(self):
                cfg = super().get_config()
                cfg.update({"max_len": self.max_len, "d_model": self.d_model})
                return cfg

        @tf.keras.utils.register_keras_serializable(package="BTCNextGen")
        class TransformerEncoderBlock(layers.Layer):
            def __init__(self, d_model=64, num_heads=4, ff_dim=256, dropout=0.1, **kwargs):
                super().__init__(**kwargs)
                self.d_model = d_model; self.num_heads = num_heads
                self.ff_dim  = ff_dim;  self.dropout   = dropout
                self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)
                self.ffn  = tf.keras.Sequential([layers.Dense(ff_dim, activation="gelu"), layers.Dense(d_model)])
                self.norm1 = layers.LayerNormalization(epsilon=1e-6)
                self.norm2 = layers.LayerNormalization(epsilon=1e-6)
                self.drop1 = layers.Dropout(dropout)
                self.drop2 = layers.Dropout(dropout)
            def call(self, x, training=False):
                attn = self.attention(x, x, training=training)
                attn = self.drop1(attn, training=training)
                x    = self.norm1(x + attn)
                ff   = self.ffn(x)
                ff   = self.drop2(ff, training=training)
                return self.norm2(x + ff)
            def get_config(self):
                cfg = super().get_config()
                cfg.update({"d_model": self.d_model, "num_heads": self.num_heads,
                             "ff_dim": self.ff_dim, "dropout": self.dropout})
                return cfg

        return {"PositionalEncoding": PositionalEncoding, "TransformerEncoderBlock": TransformerEncoderBlock}
    except Exception:
        return {}


# ────────────────────────────────────────────────────────────
# Load model on startup
# ────────────────────────────────────────────────────────────
def load_resources():
    global model, scaler
    print("\n╔══════════════════════════════════════╗")
    print("║   BTC NextGen API — Loading Resources  ║")
    print("╚══════════════════════════════════════╝")

    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            custom = get_custom_objects()
            model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom,
    compile=False,
    safe_mode=False
)
            model.compile(optimizer="adam", loss="huber", metrics=["mae"])
            print(f"✅ Model loaded  : {MODEL_PATH}")
        else:
            print(f"⚠️  Model not found: {MODEL_PATH}")
            print("   Run `python train.py` first to train the model.")

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ Scaler loaded : {SCALER_PATH}")
        else:
            print(f"⚠️  Scaler not found: {SCALER_PATH}")

    except Exception as e:
        print(f"❌ Loading failed: {e}")
        traceback.print_exc()

    status = "READY" if (model and scaler) else "MODEL NOT TRAINED — run train.py"
    print(f"\nStatus: {status}\n")


load_resources()


# ────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────
def _fetch(days: int = 365):
    """Fetch historical price data for model input."""
    from data import fetch_data
    return fetch_data(days, source="yfinance")


def _get_live_price() -> dict:
    """
    Binance se real-time current price lo.
    Prediction ke liye current_price yahan se aata hai — yfinance se nahi.
    """
    from data import fetch_live_price
    try:
        return fetch_live_price()
    except Exception as e:
        print(f"Live price warning: {e}")
        return None


def _predict_next(prices_raw: np.ndarray) -> float:
    """Run one forward pass and return predicted price in USD."""
    seq   = prices_raw[-SEQ_LEN:].reshape(-1, 1)
    scaled = scaler.transform(seq).reshape(1, SEQ_LEN, 1)
    pred   = model.predict(scaled, verbose=0)
    return float(scaler.inverse_transform(pred)[0][0])


def _compute_risk(prices: np.ndarray) -> dict:
    """Annualized volatility, Sharpe ratio, max drawdown."""
    returns    = np.diff(prices) / prices[:-1]
    volatility = float(np.std(returns) * np.sqrt(252))
    sharpe     = float((np.mean(returns) * 252 - 0.02) / (np.std(returns) * np.sqrt(252)))

    # Max drawdown
    peak       = np.maximum.accumulate(prices)
    drawdown   = (prices - peak) / peak
    max_dd     = float(np.min(drawdown))

    return {
        "volatility_annual": round(volatility, 4),
        "sharpe_ratio":      round(sharpe, 4),
        "max_drawdown":      round(max_dd, 4),
    }


def _moving_averages(prices: np.ndarray) -> dict:
    """Compute 7, 25, 50, 99 day moving averages."""
    return {
        "ma7":  round(float(np.mean(prices[-7:])),  2) if len(prices) >= 7  else None,
        "ma25": round(float(np.mean(prices[-25:])), 2) if len(prices) >= 25 else None,
        "ma50": round(float(np.mean(prices[-50:])), 2) if len(prices) >= 50 else None,
        "ma99": round(float(np.mean(prices[-99:])), 2) if len(prices) >= 99 else None,
    }


def _signal(diff: float, threshold: float) -> str:
    if diff > threshold:
        return "BUY"
    if diff < -threshold:
        return "SELL"
    return "HOLD"


# ────────────────────────────────────────────────────────────
# ENDPOINTS
# ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":         "BTC NextGen Forecast API",
        "version":      "2.0.0",
        "model_ready":  model is not None,
        "scaler_ready": scaler is not None,
        "endpoints":    ["/predict", "/historical/{days}", "/backtest", "/benchmark", "/health", "/metrics"]
    }


@app.get("/health")
def health():
    """System health check."""
    return {
        "status":        "ok" if (model and scaler) else "model_not_loaded",
        "model_loaded":  model is not None,
        "scaler_loaded": scaler is not None,
        "cpu_%":         psutil.cpu_percent(),
        "ram_%":         psutil.virtual_memory().percent,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/metrics")
def get_metrics():
    """Return saved training metrics."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {"error": "metrics.json not found. Run train.py first."}


@app.get("/predict")
def predict(
    threshold: float = Query(DEFAULT_THRESHOLD, description="USD diff to trigger BUY/SELL"),
    days:      int   = Query(120,               description="Days of history to fetch")
):
    """
    Main prediction endpoint.
    Returns current price, predicted next-day price, signal, technical indicators.
    """
    if not model or not scaler:
        return {"error": "Model not loaded. Run: python train.py"}

    try:
        raw = _fetch(max(days, SEQ_LEN + 10))

        # ── Live price (Binance real-time) ──
        live_data     = _get_live_price()
        current_price = live_data["price"] if live_data else float(raw[-1])

        # ── Model prediction (uses historical sequence) ──
        pred_price = _predict_next(raw)
        diff       = pred_price - current_price
        signal     = _signal(diff, threshold)

        # Technical indicators
        ma   = _moving_averages(raw)
        risk = _compute_risk(raw[-90:])

        # RSI (14-day)
        delta  = np.diff(raw[-15:])
        gains  = delta[delta > 0]
        losses = -delta[delta < 0]
        avg_g  = float(np.mean(gains))  if len(gains)  > 0 else 0.0
        avg_l  = float(np.mean(losses)) if len(losses) > 0 else 0.001
        rsi    = round(100 - 100 / (1 + avg_g / avg_l), 2)

        # 30-day trend
        trend_30 = round((float(raw[-1]) - float(raw[-30])) / float(raw[-30]) * 100, 2)

        result = {
            "current_price":   round(current_price, 2),
            "predicted_price": round(pred_price,    2),
            "difference":      round(diff,           2),
            "signal":          signal,
            "confidence":      round(min(abs(diff) / threshold * 100, 99), 1),
            "rsi_14":          rsi,
            "trend_30d_%":     trend_30,
            "moving_averages": ma,
            "risk_metrics":    risk,
            "threshold_used":  threshold,
            "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
            # Live price details
            "live": {
                "price":          current_price,
                "change_24h":     live_data.get("change_24h")     if live_data else None,
                "change_pct_24h": live_data.get("change_pct_24h") if live_data else None,
                "high_24h":       live_data.get("high_24h")       if live_data else None,
                "low_24h":        live_data.get("low_24h")        if live_data else None,
                "volume_24h":     live_data.get("volume_24h")     if live_data else None,
                "source":         live_data.get("source", "yfinance fallback") if live_data else "yfinance fallback",
            }
        }

        # ── Save to MongoDB Atlas ──
        try:
            saved_id = prediction_save(result)
            result["db_id"] = saved_id
        except Exception as db_err:
            result["db_warning"] = f"DB save failed: {db_err}"

        return result

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[:600]}


@app.get("/live")
def live_price():
    """
    Binance se real-time BTC/USDT price.
    Har call pe fresh price milega — cache nahi hota.
    Dashboard mein auto-refresh ke liye use karo.
    """
    try:
        from data import fetch_live_price
        data = fetch_live_price()
        return data
    except Exception as e:
        return {"error": str(e)}


@app.get("/historical/{days}")
def historical(days: int = 90):
    """
    Return OHLCV historical data for charting.
    Used by the Streamlit dashboard for candlestick + line charts.
    """
    try:
        from data import fetch_ohlcv
        df = fetch_ohlcv(days, source="binance")
        df["date"] = df["date"].astype(str)
        return {
            "days":   days,
            "count":  len(df),
            "prices": df.to_dict(orient="records")
        }
    except Exception as e:
        # Fallback: close prices only
        try:
            raw   = _fetch(days)
            dates = pd.date_range(end=pd.Timestamp.today(), periods=len(raw), freq="D")
            records = [
                {"date": str(d.date()), "close": round(float(p), 2)}
                for d, p in zip(dates, raw)
            ]
            return {"days": days, "count": len(records), "prices": records}
        except Exception as e2:
            return {"error": str(e2)}


@app.get("/backtest")
def backtest(
    days:      int   = Query(60,               description="Days to backtest"),
    threshold: float = Query(DEFAULT_THRESHOLD, description="Signal threshold USD")
):
    """
    Simulate model trading signals on historical data.
    Measures win rate, total return, signal distribution.
    """
    if not model or not scaler:
        return {"error": "Model not loaded. Run: python train.py"}

    try:
        raw       = _fetch(days + SEQ_LEN + 10)
        portfolio = 10_000.0
        signals   = []
        pnl_curve = []
        wins, losses = 0, 0

        for i in range(SEQ_LEN, len(raw) - 1):
            seq        = raw[i - SEQ_LEN:i].reshape(-1, 1)
            scaled_seq = scaler.transform(seq).reshape(1, SEQ_LEN, 1)
            pred_s     = model.predict(scaled_seq, verbose=0)
            pred_price = float(scaler.inverse_transform(pred_s)[0][0])

            current  = float(raw[i - 1])
            actual   = float(raw[i])
            diff     = pred_price - current
            sig      = _signal(diff, threshold)

            ret = (actual - current) / current
            if sig == "BUY":
                portfolio *= (1 + ret)
                if ret > 0: wins += 1
                else:       losses += 1
            elif sig == "SELL":
                portfolio *= (1 - ret)
                if ret < 0: wins += 1
                else:       losses += 1

            signals.append(sig)
            pnl_curve.append(round(portfolio, 2))

        total_return = round((portfolio - 10_000) / 10_000 * 100, 2)
        total_trades = wins + losses
        win_rate     = round(wins / total_trades * 100, 1) if total_trades > 0 else 0

        result = {
            "days":             days,
            "total_return_%":   total_return,
            "final_portfolio":  round(portfolio, 2),
            "win_rate_%":       win_rate,
            "total_trades":     total_trades,
            "wins":             wins,
            "losses":           losses,
            "signal_counts":    {
                "BUY":  signals.count("BUY"),
                "SELL": signals.count("SELL"),
                "HOLD": signals.count("HOLD")
            },
            "pnl_curve":        pnl_curve[-50:],
            "threshold_used":   threshold,
            "recommendation":   "Profitable strategy" if total_return > 5 else
                                "Break-even" if total_return > 0 else
                                "Needs tuning"
        }

        # ── Save to MongoDB Atlas ──
        try:
            saved_id = backtest_save(result)
            result["db_id"] = saved_id
        except Exception as db_err:
            result["db_warning"] = f"DB save failed: {db_err}"

        return result

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[:600]}


@app.get("/benchmark")
def benchmark(num_runs: int = Query(5, description="Number of prediction runs to benchmark")):
    """Measure inference speed, CPU usage, memory delta."""
    if not model or not scaler:
        return {"error": "Model not loaded. Run: python train.py"}

    try:
        raw     = _fetch(SEQ_LEN + 20)
        process = psutil.Process()
        times   = []

        for _ in range(num_runs):
            t0 = time.perf_counter()
            _predict_next(raw)
            times.append((time.perf_counter() - t0) * 1000)

        result = {
            "runs":                    num_runs,
            "avg_inference_ms":        round(float(np.mean(times)), 2),
            "min_inference_ms":        round(float(np.min(times)), 2),
            "max_inference_ms":        round(float(np.max(times)), 2),
            "std_inference_ms":        round(float(np.std(times)), 2),
            "cpu_%":                   psutil.cpu_percent(interval=0.5),
            "ram_%":                   psutil.virtual_memory().percent,
            "ram_used_mb":             round(process.memory_info().rss / 1024**2, 1),
            "grade":                   "Excellent" if np.mean(times) < 50 else
                                       "Good"      if np.mean(times) < 200 else
                                       "Needs GPU"
        }

        # ── Save to MongoDB Atlas ──
        try:
            saved_id = benchmark_save(result)
            result["db_id"] = saved_id
        except Exception as db_err:
            result["db_warning"] = f"DB save failed: {db_err}"

        return result
    except Exception as e:
        return {"error": str(e)}


# ────────────────────────────────────────────────────────────
# MONGODB HISTORY ENDPOINTS
# ────────────────────────────────────────────────────────────

@app.get("/db/status")
def db_status():
    """MongoDB connection status + collection counts."""
    return {
        "mongodb_connected": is_connected(),
        "stats": get_dashboard_stats()
    }


@app.get("/db/predictions")
def get_predictions(limit: int = Query(50, description="Max records to return")):
    """Return saved prediction history from MongoDB."""
    return {"predictions": prediction_history(limit)}


@app.get("/db/predictions/signals")
def get_signal_stats():
    """Aggregate BUY/SELL/HOLD counts from MongoDB."""
    return {"signal_stats": prediction_signal_stats()}


@app.get("/db/predictions/trend")
def get_pred_trend(limit: int = 30):
    """Return accuracy trend data for charts."""
    return {"trend": prediction_accuracy_trend(limit)}


@app.get("/db/backtests")
def get_backtests(limit: int = Query(20, description="Max records to return")):
    """Return saved backtest history from MongoDB."""
    return {
        "backtests": backtest_history(limit),
        "best":      backtest_best()
    }


@app.get("/db/benchmarks")
def get_benchmarks(limit: int = Query(10, description="Max records to return")):
    """Return saved benchmark history from MongoDB."""
    return {"benchmarks": benchmark_history(limit)}


@app.get("/db/tickets")
def get_tickets():
    """Get all tickets from MongoDB."""
    from database import ticket_get_all
    return {"tickets": ticket_get_all()}


@app.post("/db/tickets")
def create_ticket(body: dict):
    """Create a ticket in MongoDB."""
    from database import ticket_create
    doc = ticket_create(
        title=body.get("title",""),
        description=body.get("description",""),
        category=body.get("category","Other"),
        priority=body.get("priority","Medium")
    )
    return doc


@app.patch("/db/tickets/{ticket_id}")
def update_ticket(ticket_id: int, body: dict):
    """Update ticket status/priority in MongoDB."""
    from database import ticket_update_status
    ok = ticket_update_status(ticket_id, body.get("status","Open"), body.get("priority"))
    return {"updated": ok}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
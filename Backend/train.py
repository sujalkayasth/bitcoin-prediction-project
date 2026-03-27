# ============================================================
# train.py — BTC NextGen | Training Pipeline
# Run this ONCE to train and save model + scaler
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import joblib
import os
import json
from datetime import datetime

from model import build_model
from data import fetch_data

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "transformer.keras")  # .keras format (recommended)
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")

SEQ_LEN     = 60       # 60 days of history → predict next day
EPOCHS      = 50
BATCH_SIZE  = 32
TRAIN_SPLIT = 0.85
DATA_DAYS   = 730      # ~2 years of daily data

# Model hyperparams
D_MODEL    = 64
NUM_HEADS  = 4
FF_DIM     = 256
NUM_LAYERS = 3
DROPOUT    = 0.1


# ────────────────────────────────────────────────────────────
# Sequence creation
# ────────────────────────────────────────────────────────────
def create_sequences(data: np.ndarray, seq_len: int):
    """Create (X, y) sliding window sequences."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ────────────────────────────────────────────────────────────
# Learning Rate Scheduler
# ────────────────────────────────────────────────────────────
def cosine_lr_scheduler(epoch, lr):
    """Cosine annealing learning rate schedule."""
    import math
    min_lr  = 1e-6
    max_lr  = 1e-3
    T       = EPOCHS
    new_lr  = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / T))
    return float(new_lr)


# ────────────────────────────────────────────────────────────
# Plot training history
# ────────────────────────────────────────────────────────────
def plot_training(history, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    # Loss
    axes[0].plot(history.history["loss"],     color="#f78166", linewidth=2, label="Train Loss")
    axes[0].plot(history.history["val_loss"], color="#79c0ff", linewidth=2, label="Val Loss",   linestyle="--")
    axes[0].set_title("Training Loss (Huber)", color="white", fontsize=13)
    axes[0].set_xlabel("Epoch", color="#8b949e")
    axes[0].legend(facecolor="#21262d", labelcolor="white")

    # MAE
    axes[1].plot(history.history["mae"],     color="#7ee787", linewidth=2, label="Train MAE")
    axes[1].plot(history.history["val_mae"], color="#d2a8ff", linewidth=2, label="Val MAE",    linestyle="--")
    axes[1].set_title("Training MAE (USD)", color="white", fontsize=13)
    axes[1].set_xlabel("Epoch", color="#8b949e")
    axes[1].legend(facecolor="#21262d", labelcolor="white")

    fig.suptitle("BTC NextGen — Training History", color="white", fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ────────────────────────────────────────────────────────────
# Plot predictions vs actuals
# ────────────────────────────────────────────────────────────
def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")

    x = range(len(y_true))
    ax.plot(x, y_true, color="#79c0ff", linewidth=1.5, label="Actual BTC Price", alpha=0.9)
    ax.plot(x, y_pred, color="#f78166", linewidth=1.5, label="Predicted Price",  alpha=0.9, linestyle="--")

    # Shade error
    ax.fill_between(x, y_true, y_pred, alpha=0.15, color="#d2a8ff")

    ax.set_title("BTC Price — Actual vs Predicted (Test Set)", color="white", fontsize=14)
    ax.set_xlabel("Days", color="#8b949e")
    ax.set_ylabel("Price (USD)", color="#8b949e")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(facecolor="#21262d", labelcolor="white")

    plt.tight_layout()
    path = os.path.join(save_dir, "predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ────────────────────────────────────────────────────────────
def main():
    import tensorflow as tf

    print("=" * 60)
    print("  BTC NextGen — Transformer Training Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Fetch data
    print("\n[1/6] Fetching data...")
    raw_data = fetch_data(DATA_DAYS, source="yfinance")   # change source if needed
    print(f"  Fetched {len(raw_data)} days | Latest: ${raw_data[-1]:,.2f}")

    if len(raw_data) < SEQ_LEN + 10:
        raise ValueError(f"Not enough data: {len(raw_data)} rows, need ≥ {SEQ_LEN + 10}")

    # 2. Scale data to [0, 1]
    print("\n[2/6] Preprocessing...")
    scaler      = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(raw_data.reshape(-1, 1))  # (N, 1)

    # 3. Create sequences
    X, y = create_sequences(scaled_data.flatten(), SEQ_LEN)
    X    = X.reshape(X.shape[0], SEQ_LEN, 1)

    split   = int(TRAIN_SPLIT * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Seq len: {SEQ_LEN}")

    # 4. Build model
    print("\n[3/6] Building model...")
    model = build_model(
        seq_len=SEQ_LEN, d_model=D_MODEL, num_heads=NUM_HEADS,
        ff_dim=FF_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT
    )
    model.summary()

    # 5. Train
    print(f"\n[4/6] Training for {EPOCHS} epochs...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(cosine_lr_scheduler, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-7, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Evaluate
    print("\n[5/6] Evaluating...")
    pred_scaled = model.predict(X_test, verbose=0)
    predicted   = scaler.inverse_transform(pred_scaled).flatten()
    y_true      = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_true, predicted)
    rmse = math.sqrt(mean_squared_error(y_true, predicted))
    mape = float(np.mean(np.abs((y_true - predicted) / y_true)) * 100)
    r2   = float(1 - np.sum((y_true - predicted)**2) / np.sum((y_true - np.mean(y_true))**2))

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  MAE   : ${mae:>10,.2f} USD         │")
    print(f"  │  RMSE  : ${rmse:>10,.2f} USD         │")
    print(f"  │  MAPE  : {mape:>9.2f}%             │")
    print(f"  │  R²    : {r2:>9.4f}               │")
    print(f"  └─────────────────────────────────┘")

    # 7. Save everything
    print("\n[6/6] Saving model & artifacts...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Model  saved: {MODEL_PATH}")
    print(f"  Scaler saved: {SCALER_PATH}")

    # Save metrics
    metrics = {
        "mae": round(mae, 4), "rmse": round(rmse, 4),
        "mape": round(mape, 4), "r2": round(r2, 4),
        "trained_at": datetime.now().isoformat(),
        "epochs_run": len(history.history["loss"]),
        "train_samples": len(X_train),
        "seq_len": SEQ_LEN,
        "data_source": "yfinance"
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {METRICS_PATH}")

    # Plots
    plot_training(history, PLOTS_DIR)
    plot_predictions(y_true, predicted, PLOTS_DIR)

    print("\n✅ Training complete!")
    print(f"   Run the dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()

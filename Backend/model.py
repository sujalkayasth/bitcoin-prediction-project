# ============================================================
# model.py — BTC NextGen | Transformer Architecture
# Modern Encoder + Multi-Head Attention + Positional Encoding
# ============================================================

import tensorflow as tf
from tensorflow.keras import Model, Input, layers
import numpy as np


# ────────────────────────────────────────────────────────────
# Positional Encoding (Sinusoidal — "Attention is All You Need")
# ────────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="BTCNextGen")
class PositionalEncoding(layers.Layer):
    """
    Adds sinusoidal positional encodings to token embeddings.
    Helps the model understand temporal order of price data.
    """
    def __init__(self, max_len=5000, d_model=64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        position  = np.arange(self.max_len)[:, np.newaxis]           # (max_len, 1)
        div_term  = np.exp(
            np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model)
        )                                                              # (d_model/2,)
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)  # (1, max_len, d_model)
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg


# ────────────────────────────────────────────────────────────
# Transformer Encoder Block
# ────────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="BTCNextGen")
class TransformerEncoderBlock(layers.Layer):
    """
    Single Transformer encoder block:
      - Multi-Head Self-Attention
      - Residual connection + Layer Norm
      - Feed-Forward Network (FFN)
      - Residual connection + Layer Norm
      - Dropout for regularization
    """
    def __init__(self, d_model=64, num_heads=4, ff_dim=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.ff_dim     = ff_dim
        self.dropout    = dropout

        self.attention  = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),   # GELU > ReLU for transformers
            layers.Dense(d_model),
        ])
        self.norm1    = layers.LayerNormalization(epsilon=1e-6)
        self.norm2    = layers.LayerNormalization(epsilon=1e-6)
        self.drop1    = layers.Dropout(dropout)
        self.drop2    = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Self-attention + residual
        attn  = self.attention(x, x, training=training)
        attn  = self.drop1(attn, training=training)
        x     = self.norm1(x + attn)

        # Feed-forward + residual
        ff    = self.ffn(x)
        ff    = self.drop2(ff, training=training)
        return self.norm2(x + ff)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model":   self.d_model,
            "num_heads": self.num_heads,
            "ff_dim":    self.ff_dim,
            "dropout":   self.dropout,
        })
        return cfg


# ────────────────────────────────────────────────────────────
# Full Model Builder
# ────────────────────────────────────────────────────────────
def build_model(
    seq_len:    int   = 60,
    d_model:    int   = 64,
    num_heads:  int   = 4,
    ff_dim:     int   = 256,
    num_layers: int   = 3,
    dropout:    float = 0.1,
) -> Model:
    """
    Builds a Transformer encoder model for BTC price forecasting.

    Architecture:
        Input (seq_len, 1)
        → Dense projection → d_model
        → Positional Encoding
        → N × TransformerEncoderBlock
        → GlobalAveragePooling1D
        → Dense(64, GELU) + Dropout
        → Dense(1) — predicted next-day price (scaled)

    Args:
        seq_len:    Input sequence length (e.g. 60 = 60 days of history)
        d_model:    Embedding/model dimension
        num_heads:  Number of attention heads (d_model must be divisible by num_heads)
        ff_dim:     Feed-forward hidden dimension
        num_layers: Number of stacked encoder blocks
        dropout:    Dropout rate

    Returns:
        Compiled tf.keras.Model
    """
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    inputs = Input(shape=(seq_len, 1), name="price_sequence")

    # Project scalar price → d_model embedding
    x = layers.Dense(d_model, name="input_projection")(inputs)

    # Positional encoding
    x = PositionalEncoding(max_len=seq_len, d_model=d_model, name="pos_enc")(x)

    # Stacked Transformer encoder blocks
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"transformer_block_{i+1}"
        )(x)

    # Aggregate sequence → fixed-size vector
    x = layers.GlobalAveragePooling1D(name="global_pool")(x)

    # Output head
    x = layers.Dense(64, activation="gelu", name="head_dense")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(1, name="price_prediction")(x)

    model = Model(inputs=inputs, outputs=outputs, name="btc_transformer_v2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="huber",       # Huber loss = robust to outliers (better than MSE for prices)
        metrics=["mae"]
    )

    return model


# ────────────────────────────────────────────────────────────
# Quick test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_model(seq_len=60)
    model.summary()
    print("\nTest forward pass...")
    dummy = np.random.randn(4, 60, 1).astype(np.float32)
    out   = model(dummy, training=False)
    print(f"Output shape: {out.shape}  (expected: (4, 1))")

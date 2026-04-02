# ============================================================
# app.py — BTC NextGen Dashboard v3.0
# Live Binance price + 30s auto-refresh + live clock
# ============================================================

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json, os, time
from datetime import datetime, timedelta

st.set_page_config(
    page_title="BTC NextGen | AI Forecast",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND      = BACKEND = "https://bitcoin-prediction-project--sujalkayasth111.replit.app"
TICKETS_FILE = "tickets.json"
REFRESH_SEC  = 30   # har 30 second mein dashboard update

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[data-testid="stAppViewContainer"]{background:#0a0e1a!important;color:#e0e6f0!important;font-family:'Space Grotesk',sans-serif!important}
[data-testid="stSidebar"]{background:#0d1220!important;border-right:1px solid #1e2a3a!important}
[data-testid="metric-container"]{background:#111827!important;border:1px solid #1e293b!important;border-radius:12px!important;padding:16px!important}
[data-testid="stMetricLabel"]{color:#64748b!important;font-size:11px!important;text-transform:uppercase!important;letter-spacing:.08em!important}
[data-testid="stMetricValue"]{color:#f1f5f9!important;font-family:'JetBrains Mono',monospace!important;font-size:20px!important;font-weight:600!important}
[data-testid="stMetricDelta"] svg{display:none}
[data-testid="stTabs"]>div:first-child{border-bottom:2px solid #1e293b!important}
button[data-baseweb="tab"]{color:#64748b!important;font-weight:500!important;padding:10px 20px!important;background:transparent!important;border-bottom:2px solid transparent!important;margin-bottom:-2px!important}
button[data-baseweb="tab"][aria-selected="true"]{color:#f7931a!important;border-bottom:2px solid #f7931a!important}
[data-testid="stButton"] button{background:linear-gradient(135deg,#f7931a,#e6820a)!important;color:#000!important;font-weight:700!important;border:none!important;border-radius:8px!important;padding:10px 24px!important}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{background:#111827!important;border:1px solid #1e293b!important;color:#e0e6f0!important;border-radius:8px!important}
hr{border-color:#1e293b!important}
.signal-buy{background:#064e3b;color:#34d399;border:1px solid #059669;padding:12px 28px;border-radius:10px;font-size:26px;font-weight:700;font-family:'JetBrains Mono',monospace;display:inline-block;text-align:center}
.signal-sell{background:#450a0a;color:#f87171;border:1px solid #dc2626;padding:12px 28px;border-radius:10px;font-size:26px;font-weight:700;font-family:'JetBrains Mono',monospace;display:inline-block;text-align:center}
.signal-hold{background:#1c1917;color:#fbbf24;border:1px solid #d97706;padding:12px 28px;border-radius:10px;font-size:26px;font-weight:700;font-family:'JetBrains Mono',monospace;display:inline-block;text-align:center}
.info-card{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:16px 20px;margin:6px 0}
.card-label{color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px}
.card-value{color:#f1f5f9;font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:600}
.card-sub{color:#475569;font-size:12px;margin-top:3px}
.header-title{font-size:30px;font-weight:700;background:linear-gradient(135deg,#f7931a,#ffcc44);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header-sub{color:#475569;font-size:13px;margin-top:-4px;letter-spacing:.05em}
.status-ok{color:#34d399;font-weight:600}
.status-fail{color:#f87171;font-weight:600}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:#0a0e1a}
::-webkit-scrollbar-thumb{background:#1e293b;border-radius:3px}
</style>
""", unsafe_allow_html=True)

# ── Chart theme ─────────────────────────────────────────────
CHART_BG   = "#0a0e1a"
GRID_COLOR = "#1e293b"
TEXT_COLOR = "#64748b"
BTC_ORANGE = "#f7931a"
GREEN      = "#34d399"
RED        = "#f87171"
PURPLE     = "#a78bfa"
BLUE       = "#60a5fa"

def chart_layout(title="", height=420):
    return dict(
        title=dict(text=title, font=dict(color="#94a3b8", size=13), x=0.01),
        paper_bgcolor=CHART_BG, plot_bgcolor="#0d1220",
        font=dict(family="Space Grotesk", color=TEXT_COLOR, size=12),
        xaxis=dict(gridcolor=GRID_COLOR, color=TEXT_COLOR, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, color=TEXT_COLOR, showgrid=True, zeroline=False, tickformat="$,.0f"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height, hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e293b", font=dict(color="white")),
    )


# ── API helpers ─────────────────────────────────────────────

def api_live_price():
    """Binance se seedha live price — NO cache."""
    try:
        r = requests.get(f"{BACKEND}/live", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=30)
def api_predict(threshold=150.0):
    try:
        r = requests.get(f"{BACKEND}/predict?threshold={threshold}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=120)
def api_historical(days=90):
    try:
        r = requests.get(f"{BACKEND}/historical/{days}", timeout=20)
        r.raise_for_status()
        return r.json().get("prices", [])
    except:
        return []

@st.cache_data(ttl=300)
def api_backtest(days=60, threshold=150.0):
    try:
        r = requests.get(f"{BACKEND}/backtest?days={days}&threshold={threshold}", timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_benchmark(runs=5):
    try:
        r = requests.get(f"{BACKEND}/benchmark?num_runs={runs}", timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_health():
    try:
        r = requests.get(f"{BACKEND}/health", timeout=5)
        return r.json()
    except:
        return {"status": "unreachable"}

def api_metrics():
    try:
        r = requests.get(f"{BACKEND}/metrics", timeout=5)
        return r.json()
    except:
        return {}

def api_db_status():
    try:
        r = requests.get(f"{BACKEND}/db/status", timeout=5)
        return r.json()
    except:
        return {"mongodb_connected": False}

@st.cache_data(ttl=30)
def api_prediction_history(limit=50):
    try:
        r = requests.get(f"{BACKEND}/db/predictions?limit={limit}", timeout=10)
        return r.json().get("predictions", [])
    except:
        return []

@st.cache_data(ttl=30)
def api_signal_stats():
    try:
        r = requests.get(f"{BACKEND}/db/predictions/signals", timeout=5)
        return r.json().get("signal_stats", {})
    except:
        return {}

@st.cache_data(ttl=60)
def api_backtest_history():
    try:
        r = requests.get(f"{BACKEND}/db/backtests", timeout=10)
        return r.json()
    except:
        return {}

@st.cache_data(ttl=60)
def api_benchmark_history():
    try:
        r = requests.get(f"{BACKEND}/db/benchmarks", timeout=10)
        return r.json().get("benchmarks", [])
    except:
        return []

def load_tickets():
    if os.path.exists(TICKETS_FILE):
        with open(TICKETS_FILE) as f:
            return json.load(f)
    return []

def save_tickets(tickets):
    with open(TICKETS_FILE, "w") as f:
        json.dump(tickets, f, indent=2)


# ════════════════════════════════════════════════════════════
# SESSION STATE — refresh timer
# ════════════════════════════════════════════════════════════
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()


# ── 30 second auto-refresh check ────────────────────────────
elapsed = time.time() - st.session_state.last_refresh
if elapsed >= REFRESH_SEC:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()

remaining = int(REFRESH_SEC - elapsed)


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
      <div style='font-size:36px'>₿</div>
      <div style='font-size:17px;font-weight:700;color:#f7931a'>BTC NextGen</div>
      <div style='font-size:10px;color:#475569;letter-spacing:.1em'>AI PRICE FORECASTING</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    health   = api_health()
    db_info  = api_db_status()
    model_ok = health.get("model_loaded", False)
    mongo_ok = db_info.get("mongodb_connected", False)

    st.markdown(f"""
    <div style='padding:12px;background:#111827;border-radius:8px;border:1px solid #1e293b;margin-bottom:12px'>
      <div style='font-size:10px;color:#475569;margin-bottom:6px;text-transform:uppercase;letter-spacing:.08em'>System Status</div>
      <div class='{"status-ok" if model_ok else "status-fail"}'>{"● Model Ready" if model_ok else "● Model Offline"}</div>
      <div class='{"status-ok" if mongo_ok else "status-fail"}' style='margin-top:4px'>{"● MongoDB Connected" if mongo_ok else "● MongoDB Offline"}</div>
      <div style='font-size:11px;color:#475569;margin-top:6px'>CPU: {health.get("cpu_%","--")}% | RAM: {health.get("ram_%","--")}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Refresh countdown bar
    pct = int((remaining / REFRESH_SEC) * 100)
    st.markdown(f"""
    <div style='margin-bottom:12px'>
      <div style='font-size:10px;color:#475569;margin-bottom:4px'>
        🔄 Auto-refresh in <span style='color:#f7931a;font-weight:700'>{remaining}s</span>
      </div>
      <div style='background:#1e293b;border-radius:4px;height:4px'>
        <div style='background:#f7931a;height:4px;border-radius:4px;width:{pct}%;transition:width 1s'></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.error("Run `python train.py` first!")
    if not mongo_ok:
        st.warning("Set MONGO_URI in .env")

    if mongo_ok:
        stats = db_info.get("stats", {})
        st.markdown("### 🍃 MongoDB")
        s1, s2 = st.columns(2)
        s1.metric("Predictions", stats.get("total_predictions", 0))
        s2.metric("Backtests",   stats.get("total_backtests", 0))
        s3, s4 = st.columns(2)
        s3.metric("Benchmarks",  stats.get("total_benchmarks", 0))
        s4.metric("Tickets",     stats.get("total_tickets", 0))
        st.divider()

    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Signal Threshold ($)", 50, 1000, 150, 25)
    hist_days = st.selectbox("Chart Period", [30, 60, 90, 180, 365], index=2)

    st.divider()
    m = api_metrics()
    if m and "mae" in m:
        st.markdown("### 📊 Model Metrics")
        st.markdown(f"""
        <div class='info-card'><div class='card-label'>MAE</div><div class='card-value'>${m.get('mae',0):,.2f}</div></div>
        <div class='info-card'><div class='card-label'>MAPE</div><div class='card-value'>{m.get('mape',0):.2f}%</div></div>
        <div class='info-card'><div class='card-label'>R² Score</div><div class='card-value'>{m.get('r2',0):.4f}</div></div>
        """, unsafe_allow_html=True)
        st.caption(f"Trained: {m.get('trained_at','')[:10]}")

    st.divider()
    if st.button("🔄 Refresh Now"):
        st.session_state.last_refresh = 0
        st.cache_data.clear()
        st.rerun()


# ════════════════════════════════════════════════════════════
# HEADER — live clock JavaScript se
# ════════════════════════════════════════════════════════════
now = datetime.now()
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <div class='header-title'>Bitcoin Price Forecasting</div>
    <div class='header-sub'>USING MODERN TRANSFORMER ARCHITECTURE · AI-POWERED SIGNALS</div>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style='text-align:right;padding-top:6px'>
      <div style='font-size:11px;color:#64748b'>{now.strftime("%A, %d %B %Y")}</div>
      <div id='live-clock' style='font-family:JetBrains Mono,monospace;color:#f7931a;font-size:22px;font-weight:700'>
        {now.strftime("%H:%M")}
      </div>
    </div>
    <script>
    (function(){{
      function tick(){{
        var n=new Date();
        var el=document.getElementById('live-clock');
        if(el) el.textContent=
          String(n.getHours()).padStart(2,'0')+':'+
          String(n.getMinutes()).padStart(2,'0');
      }}
      setInterval(tick,1000); tick();
    }})();
    </script>
    """, unsafe_allow_html=True)

st.divider()


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Live Forecast",
    "🕯️  Price Charts",
    "⚡  Backtesting",
    "🖥️  Benchmarking",
    "🍃  DB History",
    "🎫  Service Desk"
])


# ══════════════════════════════════════════════
# TAB 1 — Live Forecast
# ══════════════════════════════════════════════
with tab1:

    # ── STEP 1: Live price from Binance (no cache) ──
    live     = api_live_price()
    live_ok  = live and "error" not in live
    lp       = float(live.get("price", 0))          if live_ok else 0.0
    lc       = float(live.get("change_24h", 0) or 0) if live_ok else 0.0
    lcp      = float(live.get("change_pct_24h", 0) or 0) if live_ok else 0.0
    lh       = float(live.get("high_24h", 0) or 0)  if live_ok else 0.0
    ll_price = float(live.get("low_24h", 0) or 0)   if live_ok else 0.0
    lv       = float(live.get("volume_24h", 0) or 0) if live_ok else 0.0
    lsrc     = live.get("source", "Binance")          if live_ok else "offline"
    clr      = GREEN if lcp >= 0 else RED
    arrow    = "▲" if lcp >= 0 else "▼"

    # ── Live Price Banner — st.components se (WebSocket properly kaam karta hai) ──
    import streamlit.components.v1 as components

    clr_init   = "#34d399" if lcp >= 0 else "#f87171"
    arrow_init = "▲" if lcp >= 0 else "▼"
    border_clr = "#064e3b" if lcp >= 0 else "#450a0a"
    sign_init  = "+" if lcp >= 0 else ""

    components.html(f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      body {{ background:transparent; font-family:'Space Grotesk','JetBrains Mono',monospace; }}
      #banner {{
        background:#0d1220;
        border:1px solid {border_clr};
        border-radius:14px;
        padding:20px 28px;
        display:flex;
        align-items:center;
        justify-content:space-between;
        flex-wrap:wrap;
        gap:16px;
        transition: border-color 0.3s;
      }}
      .label {{ font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.12em; margin-bottom:8px; }}
      #ws-price {{
        font-size:40px; font-weight:700; color:#f1f5f9;
        font-family:'JetBrains Mono',monospace;
        transition: color 0.15s;
        display:inline;
      }}
      #ws-pct {{ font-size:20px; font-weight:700; margin-left:12px; display:inline; color:{clr_init}; }}
      #ws-abs {{ font-size:12px; color:{clr_init}; font-weight:600; margin-top:6px; }}
      .stat-box {{ text-align:center; }}
      .stat-label {{ font-size:10px; color:#64748b; text-transform:uppercase; margin-bottom:4px; }}
      .stat-val {{ font-family:'JetBrains Mono',monospace; font-size:17px; font-weight:700; }}
      #ws-dot {{ color:#fbbf24; }}
      .right-stats {{ display:flex; gap:32px; flex-wrap:wrap; }}
    </style>
    </head>
    <body>
    <div id="banner">
      <div>
        <div class="label"><span id="ws-dot">●</span> LIVE BTC PRICE</div>
        <div>
          <span id="ws-price">${lp:,.2f}</span>
          <span id="ws-pct">{arrow_init} {sign_init}{lcp:.2f}%</span>
        </div>
        <div style="margin-top:6px;font-size:12px;color:#475569;">
          24h Change: <span id="ws-abs" style="color:{clr_init};font-weight:600">${lc:+,.2f}</span>
        </div>
      </div>
      <div class="right-stats">
        <div class="stat-box">
          <div class="stat-label">24H HIGH</div>
          <div class="stat-val" id="ws-high" style="color:#34d399">${lh:,.2f}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">24H LOW</div>
          <div class="stat-val" id="ws-low" style="color:#f87171">${ll_price:,.2f}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">VOLUME</div>
          <div class="stat-val" id="ws-vol" style="color:#94a3b8">{lv:,.0f} BTC</div>
        </div>
      </div>
    </div>

    <script>
    var lastPrice = {lp};
    var ws;

    function fmt(n, dec) {{
      return n.toLocaleString('en-US', {{minimumFractionDigits:dec, maximumFractionDigits:dec}});
    }}

    function connect() {{
      ws = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@ticker');

      ws.onopen = function() {{
        document.getElementById('ws-dot').style.color = '#34d399';
      }};

      ws.onmessage = function(e) {{
        var d     = JSON.parse(e.data);
        var price = parseFloat(d.c);
        var open  = parseFloat(d.o);
        var high  = parseFloat(d.h);
        var low   = parseFloat(d.l);
        var vol   = parseFloat(d.v);
        var pct   = parseFloat(d.P);
        var abs   = price - open;
        var up    = price >= lastPrice;
        var clr   = up ? '#34d399' : '#f87171';
        var arrow = pct >= 0 ? '▲' : '▼';
        var sign  = pct >= 0 ? '+' : '';

        // Price flash
        var pe = document.getElementById('ws-price');
        pe.style.color = clr;
        pe.textContent = '$' + fmt(price, 2);
        setTimeout(function(){{ pe.style.color = '#f1f5f9'; }}, 250);

        // % change
        var pp = document.getElementById('ws-pct');
        pp.style.color = clr;
        pp.textContent = arrow + ' ' + sign + pct.toFixed(2) + '%';

        // Abs change
        var pa = document.getElementById('ws-abs');
        pa.style.color = clr;
        pa.textContent = '$' + (abs >= 0 ? '+' : '') + fmt(abs, 2);

        // High / Low / Vol
        document.getElementById('ws-high').textContent = '$' + fmt(high, 2);
        document.getElementById('ws-low').textContent  = '$' + fmt(low, 2);
        document.getElementById('ws-vol').textContent  = fmt(vol, 0) + ' BTC';

        // Border color
        document.getElementById('banner').style.borderColor = pct >= 0 ? '#064e3b' : '#450a0a';

        lastPrice = price;
      }};

      ws.onerror = function() {{
        document.getElementById('ws-dot').style.color = '#f87171';
      }};

      ws.onclose = function() {{
        document.getElementById('ws-dot').style.color = '#fbbf24';
        setTimeout(connect, 2000);
      }};
    }}

    connect();
    </script>
    </body>
    </html>
    """, height=130, scrolling=False)

    # ── STEP 2: Model prediction ──
    data = api_predict(threshold)

    if "error" in data:
        st.error(f"Backend error: {data['error']}")
        st.info("Run: python main.py")
    else:
        # current_price = LIVE Binance price (same as banner above)
        current   = lp if live_ok and lp > 0 else data.get("current_price", 0)
        predicted = data.get("predicted_price", 0)
        diff      = predicted - current
        signal    = "BUY" if diff > threshold else "SELL" if diff < -threshold else "HOLD"
        conf      = round(min(abs(diff) / threshold * 100, 99), 1)
        rsi       = data.get("rsi_14", 0)
        trend_30  = data.get("trend_30d_%", 0)
        ma        = data.get("moving_averages", {})
        risk      = data.get("risk_metrics", {})

        # ── Metric cards ──
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Current Price (Live)", f"${current:,.2f}",
                      delta=f"{lcp:+.2f}% 24h" if live_ok else None)
            # id inject karo — WebSocket isse update karega
            st.markdown("""
            <script>
            (function(){
              var tries = 0;
              var iv = setInterval(function(){
                tries++;
                var els = document.querySelectorAll('[data-testid="stMetricValue"]');
                if(els.length > 0){
                  els[0].id = 'metric-current-price';
                  clearInterval(iv);
                }
                if(tries > 30) clearInterval(iv);
              }, 200);
            })();
            </script>
            """, unsafe_allow_html=True)
        with c2:
            st.metric("Predicted (Next Day)", f"${predicted:,.2f}",
                      delta=f"{diff:+,.2f}")
        with c3:
            st.metric("30d Trend", f"{trend_30:+.2f}%")
        with c4:
            rsi_label = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric("RSI (14)", f"{rsi:.1f}", delta=rsi_label)
        with c5:
            st.metric("Confidence", f"{conf:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        sig_col, risk_col, ma_col = st.columns([1, 1, 2])

        with sig_col:
            st.markdown("#### Trading Signal")
            sig_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(signal, "signal-hold")
            sig_desc  = {"BUY": "Price rise expected", "SELL": "Price fall expected", "HOLD": "Uncertain — stay neutral"}.get(signal, "")
            st.markdown(f"""
            <div style='text-align:center'>
              <div class='{sig_class}'>{arrow} {signal}</div>
              <div style='color:#475569;font-size:12px;margin-top:10px'>{sig_desc}</div>
              <div style='color:#64748b;font-size:11px;margin-top:4px'>Threshold: ±${threshold:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with risk_col:
            st.markdown("#### Risk Metrics")
            vol    = risk.get("volatility_annual", 0)
            sharpe = risk.get("sharpe_ratio", 0)
            max_dd = risk.get("max_drawdown", 0)
            st.markdown(f"""
            <div class='info-card'><div class='card-label'>Volatility (Annual)</div><div class='card-value'>{vol:.1%}</div></div>
            <div class='info-card'><div class='card-label'>Sharpe Ratio</div><div class='card-value'>{sharpe:.3f}</div><div class='card-sub'>{'Good' if sharpe>1 else 'Average' if sharpe>0 else 'Poor'}</div></div>
            <div class='info-card'><div class='card-label'>Max Drawdown (90d)</div><div class='card-value' style='color:#f87171'>{max_dd:.1%}</div></div>
            """, unsafe_allow_html=True)

        with ma_col:
            st.markdown("#### Moving Averages vs Live Price")
            ma_data   = {"MA7": ma.get("ma7"), "MA25": ma.get("ma25"), "MA50": ma.get("ma50"), "MA99": ma.get("ma99")}
            clrs_ma   = [GREEN if v and current > v else RED for v in ma_data.values()]
            fig_ma    = go.Figure()
            fig_ma.add_trace(go.Bar(
                x=list(ma_data.keys()), y=[v or 0 for v in ma_data.values()],
                marker_color=clrs_ma,
                text=[f"${v:,.0f}" if v else "N/A" for v in ma_data.values()],
                textposition="outside", textfont=dict(color="white", size=11)
            ))
            fig_ma.add_hline(y=current, line_dash="dash", line_color=BTC_ORANGE,
                             annotation_text=f"Live ${current:,.0f}",
                             annotation_font=dict(color=BTC_ORANGE))
            fig_ma.update_layout(**chart_layout("Moving Averages", height=270))
            fig_ma.update_layout(yaxis_tickformat="$,.0f", showlegend=False)
            st.plotly_chart(fig_ma, use_container_width=True)

        # ── Confidence Gauge ──
        st.markdown("#### Prediction Confidence")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=conf,
            number=dict(font=dict(color="white", size=36), suffix="%"),
            delta=dict(reference=50, increasing=dict(color=GREEN), decreasing=dict(color=RED)),
            gauge=dict(
                axis=dict(range=[0,100], tickfont=dict(color="#64748b")),
                bar=dict(color=BTC_ORANGE, thickness=0.25),
                bgcolor="#111827", borderwidth=0,
                steps=[dict(range=[0,33],color="#0f1f2e"),dict(range=[33,66],color="#1a2a1a"),dict(range=[66,100],color="#1a1f0e")],
                threshold=dict(line=dict(color=BTC_ORANGE,width=3),thickness=0.8,value=conf)
            )
        ))
        fig_g.update_layout(paper_bgcolor=CHART_BG, font=dict(color="white"), height=230, margin=dict(l=20,r=20,t=20,b=10))
        st.plotly_chart(fig_g, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — Price Charts
# ══════════════════════════════════════════════
with tab2:
    prices_raw = api_historical(hist_days)

    if not prices_raw:
        st.warning("No historical data. Check backend.")
    else:
        df = pd.DataFrame(prices_raw)
        df["date"] = pd.to_datetime(df["date"])

        chart_type = st.radio("Chart Type", ["Line", "Candlestick", "Area + Volume"], horizontal=True)

        if chart_type == "Line" or ("open" not in df.columns):
            close_col = "close" if "close" in df.columns else df.columns[1]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[close_col], mode="lines",
                name="BTC Price", line=dict(color=BTC_ORANGE, width=2),
                fill="tozeroy", fillcolor="rgba(247,147,26,0.07)"
            ))
            pred_data = api_predict(threshold)
            if "predicted_price" in pred_data:
                next_date = df["date"].iloc[-1] + timedelta(days=1)
                fig.add_trace(go.Scatter(
                    x=[df["date"].iloc[-1], next_date],
                    y=[float(df[close_col].iloc[-1]), pred_data["predicted_price"]],
                    mode="lines+markers", name="Forecast",
                    line=dict(color=PURPLE, width=2, dash="dot"),
                    marker=dict(size=10, color=PURPLE, symbol="star")
                ))
            fig.update_layout(**chart_layout(f"BTC/USD — {hist_days} Day History", height=460))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Candlestick" and "open" in df.columns:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(
                x=df["date"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"],
                increasing_line_color=GREEN, decreasing_line_color=RED, name="OHLC"
            ), row=1, col=1)
            if "volume" in df.columns:
                clrs_v = [GREEN if c >= o else RED for c, o in zip(df["close"], df["open"])]
                fig.add_trace(go.Bar(x=df["date"], y=df["volume"], marker_color=clrs_v, name="Volume", opacity=0.6), row=2, col=1)
            fig.update_layout(**chart_layout("BTC/USD Candlestick", height=500))
            fig.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        else:
            close_col = "close" if "close" in df.columns else df.columns[1]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
            fig.add_trace(go.Scatter(x=df["date"], y=df[close_col], fill="tozeroy", fillcolor="rgba(247,147,26,0.1)", line=dict(color=BTC_ORANGE,width=2), name="BTC Price"), row=1, col=1)
            if "volume" in df.columns:
                fig.add_trace(go.Bar(x=df["date"], y=df["volume"], marker_color=BLUE, opacity=0.5, name="Volume"), row=2, col=1)
            fig.update_layout(**chart_layout("BTC/USD Area + Volume", height=480))
            st.plotly_chart(fig, use_container_width=True)

        # Monthly return heatmap
        st.markdown("#### Monthly Return Heatmap")
        close_col = "close" if "close" in df.columns else df.columns[1]
        df["month"]     = df["date"].dt.to_period("M").astype(str)
        df["daily_ret"] = df[close_col].pct_change() * 100
        monthly = df.groupby("month")["daily_ret"].mean().reset_index()
        monthly.columns = ["month", "avg_%"]
        if len(monthly) > 1:
            fig_h = go.Figure(go.Bar(
                x=monthly["month"], y=monthly["avg_%"],
                marker_color=[GREEN if v >= 0 else RED for v in monthly["avg_%"]],
                text=[f"{v:+.2f}%" for v in monthly["avg_%"]],
                textposition="outside", textfont=dict(color="white", size=10)
            ))
            fig_h.add_hline(y=0, line_color="#475569", line_width=1)
            fig_h.update_layout(**chart_layout("Avg Daily Return by Month", height=270))
            fig_h.update_layout(yaxis_ticksuffix="%", yaxis_tickformat=".2f")
            st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — Backtesting
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### Strategy Backtesting")
    bt_col1, bt_col2 = st.columns([1, 3])
    with bt_col1:
        bt_days  = st.slider("Backtest Period", 20, 120, 60)
        bt_thr   = st.slider("Threshold ($)", 50, 1000, 150, 25)
        run_bt   = st.button("▶ Run Backtest", use_container_width=True)

    if run_bt:
        with st.spinner("Running backtest..."):
            bt = api_backtest(bt_days, bt_thr)
        if "error" in bt:
            st.error(bt["error"])
        else:
            with bt_col2:
                ret = bt.get("total_return_%", 0)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Return",    f"{ret:+.2f}%")
                k2.metric("Win Rate",        f"{bt.get('win_rate_%',0):.1f}%")
                k3.metric("Final Portfolio", f"${bt.get('final_portfolio',10000):,.0f}")
                k4.metric("Total Trades",    bt.get("total_trades", 0))

            pnl = bt.get("pnl_curve", [])
            if pnl:
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(y=pnl, mode="lines", line=dict(color=BTC_ORANGE,width=2), fill="tozeroy", fillcolor="rgba(247,147,26,0.08)", name="Portfolio"))
                fig_pnl.add_hline(y=10000, line_dash="dash", line_color="#475569", annotation_text="Start $10,000")
                fig_pnl.update_layout(**chart_layout("Portfolio Value Over Time", height=300))
                st.plotly_chart(fig_pnl, use_container_width=True)

            sc = bt.get("signal_counts", {})
            if sc:
                fig_sc = go.Figure(go.Pie(
                    labels=list(sc.keys()), values=list(sc.values()), hole=0.5,
                    marker_colors=[GREEN, RED, "#fbbf24"], textfont=dict(color="white")
                ))
                fig_sc.update_layout(**chart_layout("Signal Distribution", height=250), showlegend=True)
                st.plotly_chart(fig_sc, use_container_width=True)

            st.info(f"Recommendation: {bt.get('recommendation','N/A')}")
    else:
        st.info("Click '▶ Run Backtest' to simulate.")


# ══════════════════════════════════════════════
# TAB 4 — Benchmarking
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### Inference Benchmark")
    b_col1, b_col2 = st.columns([1, 3])
    with b_col1:
        b_runs    = st.slider("Runs", 3, 20, 5)
        run_bench = st.button("🚀 Run Benchmark", use_container_width=True)

    if run_bench:
        with st.spinner("Benchmarking..."):
            bench = api_benchmark(b_runs)
        if "error" in bench:
            st.error(bench["error"])
        else:
            with b_col2:
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Avg Inference", f"{bench.get('avg_inference_ms',0):.1f} ms")
                g2.metric("Min / Max",     f"{bench.get('min_inference_ms',0):.0f} / {bench.get('max_inference_ms',0):.0f} ms")
                g3.metric("CPU",           f"{bench.get('cpu_%',0):.1f}%")
                g4.metric("RAM Used",      f"{bench.get('ram_used_mb',0):.0f} MB")

            grade = bench.get("grade", "N/A")
            clr_g = GREEN if grade == "Excellent" else BTC_ORANGE if grade == "Good" else RED
            st.markdown(f"<div style='text-align:center;margin:16px 0'><span style='color:{clr_g};font-size:28px;font-weight:700;font-family:JetBrains Mono;border:1px solid {clr_g};padding:8px 24px;border-radius:8px'>{grade}</span></div>", unsafe_allow_html=True)

            gc1, gc2 = st.columns(2)
            for col, lbl, val, clr_g2 in [(gc1,"CPU %",bench.get("cpu_%",0),BLUE),(gc2,"RAM %",bench.get("ram_%",0),PURPLE)]:
                with col:
                    fig_gg = go.Figure(go.Indicator(
                        mode="gauge+number", value=val,
                        number=dict(suffix="%", font=dict(color="white")),
                        gauge=dict(axis=dict(range=[0,100]), bar=dict(color=clr_g2,thickness=0.3), bgcolor="#111827", borderwidth=0, steps=[dict(range=[0,100],color="#0d1220")])
                    ))
                    fig_gg.update_layout(title=dict(text=lbl,font=dict(color="#64748b",size=13)), paper_bgcolor=CHART_BG, font=dict(color="white"), height=190, margin=dict(l=20,r=20,t=40,b=10))
                    st.plotly_chart(fig_gg, use_container_width=True)

            with st.expander("Full Report"):
                st.json(bench)
    else:
        st.info("Click '🚀 Run Benchmark'.")


# ══════════════════════════════════════════════
# TAB 5 — DB History
# ══════════════════════════════════════════════
with tab5:
    st.markdown("### 🍃 MongoDB Atlas — Saved History")
    mongo_ok2 = api_db_status().get("mongodb_connected", False)

    if not mongo_ok2:
        st.error("MongoDB not connected. Set MONGO_URI in .env")
    else:
        dbt1, dbt2, dbt3 = st.tabs(["📊 Predictions", "⚡ Backtests", "🖥️ Benchmarks"])

        with dbt1:
            pred_hist = api_prediction_history(50)
            sig_stats = api_signal_stats()
            if pred_hist:
                if sig_stats:
                    sc1, sc2 = st.columns([1,2])
                    with sc1:
                        fig_sp = go.Figure(go.Pie(
                            labels=list(sig_stats.keys()), values=list(sig_stats.values()), hole=0.55,
                            marker_colors=[GREEN if k=="BUY" else RED if k=="SELL" else "#fbbf24" for k in sig_stats],
                            textfont=dict(color="white",size=12)
                        ))
                        fig_sp.update_layout(**chart_layout("Signal Distribution", height=250), showlegend=True)
                        st.plotly_chart(fig_sp, use_container_width=True)
                    with sc2:
                        df_ph = pd.DataFrame(pred_hist[:30])
                        if "timestamp" in df_ph.columns:
                            df_ph["timestamp"] = pd.to_datetime(df_ph["timestamp"])
                            fig_pp = go.Figure()
                            fig_pp.add_trace(go.Scatter(x=df_ph["timestamp"], y=df_ph["current_price"], name="Current", line=dict(color=BLUE,width=2)))
                            fig_pp.add_trace(go.Scatter(x=df_ph["timestamp"], y=df_ph["predicted_price"], name="Predicted", line=dict(color=BTC_ORANGE,width=2,dash="dot")))
                            fig_pp.update_layout(**chart_layout("Actual vs Predicted", height=250))
                            st.plotly_chart(fig_pp, use_container_width=True)
                df_show = pd.DataFrame(pred_hist)
                show_cols = [c for c in ["timestamp","current_price","predicted_price","signal","confidence","rsi_14"] if c in df_show.columns]
                st.dataframe(df_show[show_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No predictions yet. Go to Tab 1.")

        with dbt2:
            bt_data = api_backtest_history()
            bt_list = bt_data.get("backtests", [])
            bt_best = bt_data.get("best", {})
            if bt_list:
                if bt_best:
                    bb1, bb2, bb3 = st.columns(3)
                    bb1.metric("Best Return",    f"{bt_best.get('total_return_%',0):+.2f}%")
                    bb2.metric("Win Rate",       f"{bt_best.get('win_rate_%',0):.1f}%")
                    bb3.metric("Final Portfolio",f"${bt_best.get('final_portfolio',10000):,.0f}")
                df_bt = pd.DataFrame(bt_list)
                show_bt = [c for c in ["run_at","days","total_return_%","win_rate_%","total_trades","recommendation"] if c in df_bt.columns]
                st.dataframe(df_bt[show_bt], use_container_width=True, hide_index=True)
            else:
                st.info("No backtests yet. Go to Tab 3.")

        with dbt3:
            bench_list = api_benchmark_history()
            if bench_list:
                b1,b2,b3 = st.columns(3)
                b1.metric("Best Inference", f"{min(b.get('avg_inference_ms',999) for b in bench_list):.1f} ms")
                b2.metric("Latest Grade",   bench_list[0].get("grade","N/A"))
                b3.metric("Total Runs",     len(bench_list))
                df_bh = pd.DataFrame(bench_list)
                show_bh = [c for c in ["run_at","runs","avg_inference_ms","cpu_%","ram_%","grade"] if c in df_bh.columns]
                st.dataframe(df_bh[show_bh], use_container_width=True, hide_index=True)
            else:
                st.info("No benchmarks yet. Go to Tab 4.")


# ══════════════════════════════════════════════
# TAB 6 — Service Desk
# ══════════════════════════════════════════════
with tab6:
    st.markdown("### 🎫 Service Desk")
    mongo_ok3 = api_db_status().get("mongodb_connected", False)
    st.success("🍃 MongoDB Atlas" if mongo_ok3 else "⚠️ Local JSON fallback", icon="✅" if mongo_ok3 else "⚠️")

    def _local_save(t,d,c,p):
        tix=load_tickets(); tix.append({"id":len(tix)+1,"title":t,"description":d,"category":c,"priority":p,"status":"Open","date":datetime.now().strftime("%d %b %Y, %H:%M")}); save_tickets(tix)

    fc, lc = st.columns([1,2])
    with fc:
        st.markdown("#### New Ticket")
        title    = st.text_input("Title *")
        desc     = st.text_area("Description *", height=100)
        category = st.selectbox("Category", ["Prediction Error","Model Accuracy","UI Issue","Performance","Feature Request","Other"])
        priority = st.selectbox("Priority", ["Low","Medium","High","Critical"])
        if st.button("Submit Ticket", use_container_width=True):
            if title.strip() and desc.strip():
                if mongo_ok3:
                    try:
                        requests.post(f"{BACKEND}/db/tickets", json={"title":title,"description":desc,"category":category,"priority":priority}, timeout=10)
                        st.success("✅ Saved to MongoDB!")
                    except:
                        _local_save(title,desc,category,priority); st.success("✅ Saved locally!")
                else:
                    _local_save(title,desc,category,priority); st.success("✅ Ticket submitted!")
                st.rerun()
            else:
                st.error("Title and Description required.")

    with lc:
        st.markdown("#### All Tickets")
        if mongo_ok3:
            try:
                r = requests.get(f"{BACKEND}/db/tickets", timeout=10)
                tickets = r.json().get("tickets", [])
            except:
                tickets = load_tickets()
        else:
            tickets = load_tickets()

        if tickets:
            op = sum(1 for t in tickets if t.get("status")=="Open")
            pr = sum(1 for t in tickets if t.get("status")=="In Progress")
            rs = sum(1 for t in tickets if t.get("status")=="Resolved")
            ts1,ts2,ts3 = st.columns(3)
            ts1.metric("Open",op); ts2.metric("In Progress",pr); ts3.metric("Resolved",rs)
            df_t   = pd.DataFrame(tickets)
            sc_all = [c for c in ["ticket_id","id","title","category","priority","status","created_at","date"] if c in df_t.columns][:6]
            edited = st.data_editor(df_t[sc_all], column_config={
                "status":   st.column_config.SelectboxColumn("Status",   options=["Open","In Progress","Resolved"], required=True),
                "priority": st.column_config.SelectboxColumn("Priority", options=["Low","Medium","High","Critical"], required=True)
            }, use_container_width=True, hide_index=True, num_rows="fixed")
            if st.button("💾 Save Changes"):
                if not mongo_ok3:
                    for i,row in edited.iterrows():
                        if i < len(tickets): tickets[i]["status"] = row.get("status","Open")
                    save_tickets(tickets)
                st.success("✅ Saved!"); st.rerun()
        else:
            st.info("No tickets yet.")
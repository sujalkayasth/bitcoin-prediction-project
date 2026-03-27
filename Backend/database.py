# ============================================================
# database.py — BTC NextGen | MongoDB Atlas + .env support
# ============================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("dotenv loaded")
except ImportError:
    print("pip install python-dotenv")

from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime
from bson import ObjectId
import os, json

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME   = os.getenv("DB_NAME", "btc_nextgen")

if not MONGO_URI or "YOUR_USERNAME" in MONGO_URI:
    print("MONGO_URI not set in .env")

COL_TICKETS="tickets"; COL_PREDICTIONS="predictions"; COL_BACKTESTS="backtests"; COL_BENCHMARKS="benchmarks"
_client=None; _db=None

def get_db():
    global _client, _db
    if _db is not None: return _db
    if not MONGO_URI or "YOUR_USERNAME" in MONGO_URI: return None
    try:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        _db = _client[DB_NAME]
        print(f"MongoDB connected: {DB_NAME}")
        _ensure_indexes()
        return _db
    except Exception as e:
        print(f"MongoDB failed: {e}")
        return None

def _ensure_indexes():
    try:
        _db[COL_TICKETS].create_index([("created_at", DESCENDING)])
        _db[COL_TICKETS].create_index([("status", 1)])
        _db[COL_PREDICTIONS].create_index([("timestamp", DESCENDING)])
        _db[COL_BACKTESTS].create_index([("run_at", DESCENDING)])
        _db[COL_BENCHMARKS].create_index([("run_at", DESCENDING)])
    except: pass

def is_connected():
    try:
        db = get_db()
        if db is None: return False
        db.client.admin.command("ping")
        return True
    except: return False

def _serialize(doc):
    if doc is None: return {}
    doc = dict(doc)
    if "_id" in doc: doc["_id"] = str(doc["_id"])
    for k,v in doc.items():
        if isinstance(v, datetime): doc[k] = v.strftime("%d %b %Y, %H:%M")
    return doc

def _serialize_list(docs): return [_serialize(d) for d in docs]

# TICKETS
def ticket_create(title, description, category, priority="Medium"):
    db = get_db()
    if db is None: return _fallback_ticket_create(title, description, category, priority)
    count = db[COL_TICKETS].count_documents({})
    doc = {"ticket_id": count+1, "title": title.strip(), "description": description.strip(),
           "category": category, "priority": priority, "status": "Open",
           "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()}
    result = db[COL_TICKETS].insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    return _serialize(doc)

def ticket_get_all(status_filter=None):
    db = get_db()
    if db is None: return _fallback_load_tickets()
    query = {"status": status_filter} if status_filter else {}
    return _serialize_list(db[COL_TICKETS].find(query).sort("created_at", DESCENDING))

def ticket_update_status(ticket_id, status, priority=None):
    db = get_db()
    if db is None: return False
    fields = {"status": status, "updated_at": datetime.utcnow()}
    if priority: fields["priority"] = priority
    result = db[COL_TICKETS].update_one({"ticket_id": ticket_id}, {"$set": fields})
    return result.modified_count > 0

def ticket_delete(ticket_id):
    db = get_db()
    if db is None: return False
    return db[COL_TICKETS].delete_one({"ticket_id": ticket_id}).deleted_count > 0

def ticket_stats():
    db = get_db()
    if db is None:
        from collections import Counter
        return dict(Counter(t.get("status","Open") for t in _fallback_load_tickets()))
    return {r["_id"]: r["count"] for r in db[COL_TICKETS].aggregate([{"$group":{"_id":"$status","count":{"$sum":1}}}])}

# PREDICTIONS
def prediction_save(data):
    db = get_db()
    if db is None: return ""
    doc = {"timestamp": datetime.utcnow(), "current_price": data.get("current_price"),
           "predicted_price": data.get("predicted_price"), "difference": data.get("difference"),
           "signal": data.get("signal"), "confidence": data.get("confidence"),
           "rsi_14": data.get("rsi_14"), "trend_30d": data.get("trend_30d_%"),
           "moving_averages": data.get("moving_averages",{}), "risk_metrics": data.get("risk_metrics",{}),
           "threshold_used": data.get("threshold_used")}
    return str(db[COL_PREDICTIONS].insert_one(doc).inserted_id)

def prediction_history(limit=50):
    db = get_db()
    if db is None: return []
    return _serialize_list(db[COL_PREDICTIONS].find({}).sort("timestamp", DESCENDING).limit(limit))

def prediction_signal_stats():
    db = get_db()
    if db is None: return {}
    return {r["_id"]: r["count"] for r in db[COL_PREDICTIONS].aggregate([{"$group":{"_id":"$signal","count":{"$sum":1}}}])}

def prediction_accuracy_trend(limit=30):
    db = get_db()
    if db is None: return []
    return _serialize_list(db[COL_PREDICTIONS].find({},{"timestamp":1,"current_price":1,"predicted_price":1,"signal":1}).sort("timestamp",DESCENDING).limit(limit))

# BACKTESTS
def backtest_save(data):
    db = get_db()
    if db is None: return ""
    doc = {"run_at": datetime.utcnow(), "days": data.get("days"),
           "total_return_%": data.get("total_return_%"), "final_portfolio": data.get("final_portfolio"),
           "win_rate_%": data.get("win_rate_%"), "total_trades": data.get("total_trades"),
           "wins": data.get("wins"), "losses": data.get("losses"),
           "signal_counts": data.get("signal_counts",{}), "recommendation": data.get("recommendation"),
           "threshold_used": data.get("threshold_used")}
    return str(db[COL_BACKTESTS].insert_one(doc).inserted_id)

def backtest_history(limit=20):
    db = get_db()
    if db is None: return []
    return _serialize_list(db[COL_BACKTESTS].find({}).sort("run_at", DESCENDING).limit(limit))

def backtest_best():
    db = get_db()
    if db is None: return {}
    return _serialize(db[COL_BACKTESTS].find_one(sort=[("total_return_%", DESCENDING)]))

# BENCHMARKS
def benchmark_save(data):
    db = get_db()
    if db is None: return ""
    doc = {"run_at": datetime.utcnow(), "runs": data.get("runs"),
           "avg_inference_ms": data.get("avg_inference_ms"), "min_inference_ms": data.get("min_inference_ms"),
           "max_inference_ms": data.get("max_inference_ms"), "std_inference_ms": data.get("std_inference_ms"),
           "cpu_%": data.get("cpu_%"), "ram_%": data.get("ram_%"),
           "ram_used_mb": data.get("ram_used_mb"), "grade": data.get("grade")}
    return str(db[COL_BENCHMARKS].insert_one(doc).inserted_id)

def benchmark_history(limit=10):
    db = get_db()
    if db is None: return []
    return _serialize_list(db[COL_BENCHMARKS].find({}).sort("run_at", DESCENDING).limit(limit))

# DASHBOARD STATS
def get_dashboard_stats():
    db = get_db()
    if db is None: return {"connected": False}
    try:
        return {"connected": True,
                "total_predictions": db[COL_PREDICTIONS].count_documents({}),
                "total_backtests":   db[COL_BACKTESTS].count_documents({}),
                "total_benchmarks":  db[COL_BENCHMARKS].count_documents({}),
                "total_tickets":     db[COL_TICKETS].count_documents({}),
                "open_tickets":      db[COL_TICKETS].count_documents({"status":"Open"}),
                "signal_stats":      prediction_signal_stats(),
                "ticket_stats":      ticket_stats()}
    except Exception as e: return {"connected": True, "error": str(e)}

# FALLBACK
FALLBACK_FILE = "tickets_fallback.json"
def _fallback_load_tickets():
    if os.path.exists(FALLBACK_FILE):
        with open(FALLBACK_FILE) as f: return json.load(f)
    return []
def _fallback_save_tickets(tickets):
    with open(FALLBACK_FILE, "w") as f: json.dump(tickets, f, indent=2, default=str)
def _fallback_ticket_create(title, description, category, priority):
    tickets = _fallback_load_tickets()
    doc = {"ticket_id": len(tickets)+1, "title": title, "description": description,
           "category": category, "priority": priority, "status": "Open",
           "created_at": datetime.now().strftime("%d %b %Y, %H:%M")}
    tickets.append(doc)
    _fallback_save_tickets(tickets)
    return doc

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  BTC NextGen — MongoDB Connection Test")
    print("="*50)
    print(f"MONGO_URI: {'SET' if MONGO_URI and 'YOUR_USERNAME' not in MONGO_URI else 'NOT SET'}")
    print(f"DB_NAME  : {DB_NAME}")
    if is_connected():
        s = get_dashboard_stats()
        print("SUCCESS!")
        print(f"  predictions={s.get('total_predictions',0)}, backtests={s.get('total_backtests',0)}, tickets={s.get('total_tickets',0)}")
    else:
        print("FAILED — .env check karo")
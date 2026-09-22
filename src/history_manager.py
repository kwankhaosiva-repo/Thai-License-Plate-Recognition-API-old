"""
src/history_manager.py

Thread-safe SQLite Recognition History & Data Collection Manager.
Stores recognition events, plate crops (thumbnails), latency waterfalls, and validation statuses.
Provides filtering, search, pagination, summary stats, and CSV export.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TZ_BANGKOK = timezone(timedelta(hours=7), name="Asia/Bangkok")

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "lpr_history.db"

_lock = threading.Lock()

# Active history backend ("firestore" | "sqlite"), resolved lazily by _get_history_store()
_history_backend: Optional[str] = None


def get_db_connection() -> sqlite3.Connection:
    """Returns a thread-safe connection to the SQLite history database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema and indexes."""
    with _lock:
        conn = get_db_connection()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS recognition_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        record_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        plate_text TEXT NOT NULL,
                        country TEXT NOT NULL,
                        province TEXT NOT NULL,
                        province_prob REAL DEFAULT 0.0,
                        pattern TEXT NOT NULL,
                        is_valid INTEGER NOT NULL DEFAULT 1,
                        char_box_text TEXT DEFAULT '',
                        ctc_text TEXT DEFAULT '',
                        char_box_status TEXT DEFAULT 'complete',
                        char_box_note TEXT DEFAULT '',
                        total_latency_ms INTEGER DEFAULT 0,
                        model_latencies TEXT DEFAULT '{}',
                        thumbnail TEXT DEFAULT '',
                        raw_image TEXT DEFAULT ''
                    )
                """)
                # Migration: add raw_image, user_id, user_email columns if older DB schema exists
                try:
                    conn.execute("ALTER TABLE recognition_history ADD COLUMN raw_image TEXT DEFAULT ''")
                except Exception:
                    pass
                try:
                    conn.execute("ALTER TABLE recognition_history ADD COLUMN user_id TEXT DEFAULT 'guest'")
                except Exception:
                    pass
                try:
                    conn.execute("ALTER TABLE recognition_history ADD COLUMN user_email TEXT DEFAULT ''")
                except Exception:
                    pass

                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON recognition_history(timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_country ON recognition_history(country)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_plate ON recognition_history(plate_text)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_is_valid ON recognition_history(is_valid)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON recognition_history(user_id)")
        finally:
            conn.close()


def image_to_base64_thumbnail(image_bgr: Optional[np.ndarray], max_w: int = 360, max_h: int = 140, quality: int = 85) -> str:
    """Converts a BGR plate crop to a clear base64 JPEG data URL for display and zoom."""
    if image_bgr is None or image_bgr.size == 0:
        return ""
    try:
        h, w = image_bgr.shape[:2]
        scale = min(max_w / float(max(1, w)), max_h / float(max(1, h)), 1.0)
        new_w, new_h = max(24, int(w * scale)), max(16, int(h * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def image_to_base64_full(image_bgr: Optional[np.ndarray], max_dim: int = 960, quality: int = 75) -> str:
    """Converts a full BGR frame to a compressed base64 JPEG data URL for lightbox zoom inspection."""
    if image_bgr is None or image_bgr.size == 0:
        return ""
    try:
        h, w = image_bgr.shape[:2]
        long_side = max(h, w)
        scale = min(max_dim / float(max(1, long_side)), 1.0)
        new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


_recent_saves_lock = threading.Lock()
_recent_saves: Dict[str, float] = {}


def save_recognition(
    data: Dict[str, Any],
    thumbnail_bgr: Optional[np.ndarray] = None,
    raw_bgr: Optional[np.ndarray] = None,
    user: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Saves a recognized plate result to the history database and syncs to Google Cloud (Firestore & BigQuery).
    Robustly maps timing, confidence, crops, full-resolution images, and authenticated user identity.
    Returns the unique record_id.
    """
    plate_text = str(data.get("plate_text") or "").strip()
    clean_key = re.sub(r"[\s-]", "", plate_text).strip()
    dedup_window_sec = float(data.get("dedup_window_sec", 5.0))

    # 5-Second Deduplication Guard:
    # Groups real-time video stream frames of the same vehicle together into 1 record.
    # Suppresses duplicate database writes of the same plate text within 5.0 seconds.
    if clean_key and dedup_window_sec > 0:
        with _recent_saves_lock:
            now_ts = time.time()
            # Clean up old keys (> 60s)
            expired = [k for k, t in _recent_saves.items() if (now_ts - t) > 60.0]
            for k in expired:
                del _recent_saves[k]

            last_saved = _recent_saves.get(clean_key, 0.0)
            if (now_ts - last_saved) < dedup_window_sec:
                logger.info(f"[History Dedup] Skipped duplicate save for '{plate_text}' within {dedup_window_sec}s window")
                return ""
            _recent_saves[clean_key] = now_ts

    record_id = f"lpr_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    now_dt = datetime.now(TZ_BANGKOK)
    timestamp_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    created_at = time.time()

    country = str(data.get("country") or "Thai").strip()
    province = str(data.get("province") or "").strip()

    # Robust province probability extraction (percentage 0-100)
    conf_val = data.get("confidence")
    prov_conf = conf_val.get("province_classification") if isinstance(conf_val, dict) else 0.0
    prov_prob_raw = (
        data.get("province_prob")
        or prov_conf
        or 0.0
    )
    try:
        province_prob = float(prov_prob_raw)
        if 0.0 < province_prob <= 1.0:
            province_prob = round(province_prob * 100.0, 1)
    except (ValueError, TypeError):
        province_prob = 0.0

    pattern = str(
        data.get("pattern_name")
        or data.get("pattern")
        or "Standard Private Car"
    ).strip()
    is_valid = 1 if data.get("is_valid", True) else 0

    char_box_text = str(data.get("char_box_text") or "").strip()
    ctc_text = str(data.get("ctc_text") or data.get("raw_plate_text") or "").strip()
    char_box_status = str(data.get("char_box_status") or "complete").strip()
    char_box_note = str(data.get("char_box_note") or "").strip()

    # Timing / Latency resolution
    timing = data.get("timing") or {}
    total_latency_ms = int(
        data.get("total_latency_ms")
        or data.get("latency_ms")
        or timing.get("total_ms")
        or 0
    )
    # Save waterfall breakdown
    model_latencies = json.dumps(data.get("model_latencies") or timing or {})

    # Plate crop thumbnail
    thumbnail_b64 = ""
    if thumbnail_bgr is not None and thumbnail_bgr.size > 0:
        thumbnail_b64 = image_to_base64_thumbnail(thumbnail_bgr)
    elif "crops" in data and data["crops"].get("plate_rectified"):
        thumbnail_b64 = data["crops"]["plate_rectified"]

    # Full frame / vehicle scene image
    raw_b64 = ""
    if raw_bgr is not None and raw_bgr.size > 0:
        raw_b64 = image_to_base64_full(raw_bgr)
    elif "crops" in data and data["crops"].get("raw"):
        raw_b64 = data["crops"]["raw"]

    # Resolve authenticated user identity
    user_info = user or data.get("user") or {}
    user_id = str(user_info.get("uid") or user_info.get("user_id") or "guest")
    user_email = str(user_info.get("email") or user_info.get("user_email") or "")

    with _lock:
        conn = get_db_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO recognition_history (
                        record_id, timestamp, created_at, plate_text, country,
                        province, province_prob, pattern, is_valid,
                        char_box_text, ctc_text, char_box_status, char_box_note,
                        total_latency_ms, model_latencies, thumbnail, raw_image,
                        user_id, user_email
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id, timestamp_str, created_at, plate_text, country,
                        province, province_prob, pattern, is_valid,
                        char_box_text, ctc_text, char_box_status, char_box_note,
                        total_latency_ms, model_latencies, thumbnail_b64, raw_b64,
                        user_id, user_email,
                    ),
                )
                # Automatic Rolling Cap (FIFO):
                # Enforces a 3,000-record limit so the database file never swells indefinitely in container RAM/disk
                conn.execute(
                    """
                    DELETE FROM recognition_history
                    WHERE id IN (
                        SELECT id FROM recognition_history
                        ORDER BY id DESC
                        LIMIT -1 OFFSET 3000
                    )
                    """
                )
        finally:
            conn.close()

    # Asynchronous Google Cloud Platform Sync (Firestore 'lpr-db' & BigQuery 'lpr_query.lpr_history')
    try:
        from src.cloud_storage_manager import cloud_storage_manager
        cloud_payload = dict(data)
        cloud_payload["record_id"] = record_id
        cloud_payload["timestamp"] = now_dt.isoformat()
        cloud_payload["thumbnail"] = thumbnail_b64
        cloud_payload["raw_image"] = raw_b64
        cloud_payload["total_latency_ms"] = total_latency_ms
        cloud_payload["province_prob"] = province_prob
        cloud_payload["pattern"] = pattern
        cloud_payload["char_box_text"] = char_box_text
        cloud_payload["ctc_text"] = ctc_text
        cloud_payload["char_box_status"] = char_box_status
        cloud_payload["user_id"] = user_id
        cloud_payload["user_email"] = user_email
        cloud_storage_manager.sync_recognition(cloud_payload, user_profile=user_info)
    except Exception as e:
        logger.warning(f"[CloudSync Dispatch Error] {e}")

    return record_id


def _coerce_timestamp(value: Any) -> str:
    """Normalizes Firestore timestamps (datetime/Z-ISO/local-naive) to 'YYYY-MM-DD HH:MM:SS' local strings."""
    if isinstance(value, datetime):
        return value.astimezone(TZ_BANGKOK).strftime("%Y-%m-%d %H:%M:%S")
    s = str(value or "")
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(TZ_BANGKOK)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value or "")


# ---------------------------------------------------------------------------
# Firestore-first history layer
#
# Cloud Run containers are ephemeral (per-instance SQLite dies on redeploy and
# diverges across instances), so history APIs read the durable Firestore store
# first and fall back to local SQLite ONLY when Firestore is unavailable
# (e.g. running locally without GCP credentials).
# ---------------------------------------------------------------------------


def _firestore_available() -> bool:
    """True if the cloud sync manager can reach Firestore (checks cached client + ping)."""
    try:
        from src.cloud_storage_manager import cloud_storage_manager as csm
        status = csm.get_status()
        return bool(status.get("firestore", {}).get("connected"))
    except Exception:
        return False


def _normalize_firestore_record(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Maps a Firestore document to the SQLite record shape the dashboard expects."""
    return {
        "record_id": doc.get("record_id") or doc.get("id") or "",
        "timestamp": _coerce_timestamp(doc.get("timestamp")),
        "plate_text": doc.get("plate_text", ""),
        "country": doc.get("country", "Thai"),
        "province": doc.get("province", ""),
        "province_prob": float(doc.get("province_prob") or 0.0),
        "pattern": doc.get("pattern", ""),
        "is_valid": 1 if doc.get("is_valid", True) else 0,
        "char_box_text": doc.get("char_box_text", ""),
        "ctc_text": doc.get("ctc_text", ""),
        "char_box_status": doc.get("char_box_status", "complete"),
        "char_box_note": "",
        "total_latency_ms": int(doc.get("total_latency_ms") or doc.get("latency_ms") or 0),
        "model_latencies": {},
        # UI reads `thumbnail` / `raw_image` as data URLs
        "thumbnail": doc.get("thumbnail", ""),
        "raw_image": doc.get("raw_image", ""),
        "user_id": doc.get("user_id", "guest"),
        "user_email": doc.get("user_email", ""),
        "user_name": doc.get("user_name", ""),
    }


def _get_history_store() -> str:
    """Decides the active history backend once per process (cached)."""
    global _history_backend
    if _history_backend is None:
        import os
        if os.environ.get("K_SERVICE") or os.environ.get("HISTORY_BACKEND") == "firestore":
            _history_backend = "firestore"
        elif os.environ.get("HISTORY_BACKEND") == "sqlite":
            _history_backend = "sqlite"
        else:
            # Local default: probe Firestore, use it if reachable, else SQLite
            _history_backend = "firestore" if _firestore_available() else "sqlite"
        logger.info(f"[History] Active backend: {_history_backend}")
    return _history_backend


def query_history(
    page: int = 1,
    page_size: int = 50,
    date_from: str = "",
    date_to: str = "",
    country: str = "",
    pattern: str = "",
    status: str = "",
    search: str = "",
    sort_by: str = "timestamp",
    sort_order: str = "desc",
) -> Dict[str, Any]:
    """Queries paginated historical records (Firestore-first, SQLite fallback)."""
    if _get_history_store() == "firestore":
        try:
            from src.cloud_storage_manager import cloud_storage_manager as csm
            docs = csm.query_firestore_filtered(
                limit=page_size,
                date_from=date_from,
                date_to=date_to,
                country=country,
                pattern=pattern,
                status=status,
                search=search,
            )
            records = [_normalize_firestore_record(d) for d in docs]
            reverse = sort_order.lower() != "asc"
            key_map = {
                "timestamp": lambda r: r["timestamp"],
                "plate_text": lambda r: r["plate_text"],
                "country": lambda r: r["country"],
                "province": lambda r: r["province"],
                "total_latency_ms": lambda r: r["total_latency_ms"],
                "is_valid": lambda r: r["is_valid"],
            }
            records.sort(key=key_map.get(sort_by, key_map["timestamp"]), reverse=reverse)
            total = csm.count_firestore_records()
            total_pages = max(1, (total + page_size - 1) // page_size)
            return {
                "records": records,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "source": "firestore",
            }
        except Exception as e:
            logger.warning(f"[History] Firestore query failed, falling back to SQLite: {e}")

    return _query_history_sqlite(
        page=page, page_size=page_size, date_from=date_from, date_to=date_to,
        country=country, pattern=pattern, status=status, search=search,
        sort_by=sort_by, sort_order=sort_order,
    )


def _query_history_sqlite(
    page: int = 1,
    page_size: int = 50,
    date_from: str = "",
    date_to: str = "",
    country: str = "",
    pattern: str = "",
    status: str = "",
    search: str = "",
    sort_by: str = "timestamp",
    sort_order: str = "desc",
) -> Dict[str, Any]:
    """Queries paginated historical records with multi-criteria filtering."""
    page = max(1, page)
    page_size = max(1, min(page_size, 200))
    offset = (page - 1) * page_size

    query_parts = ["1=1"]
    params: List[Any] = []

    if date_from:
        query_parts.append("timestamp >= ?")
        params.append(f"{date_from} 00:00:00")
    if date_to:
        query_parts.append("timestamp <= ?")
        params.append(f"{date_to} 23:59:59")

    if country and country.upper() != "ALL":
        query_parts.append("country = ?")
        params.append(country)

    if pattern and pattern.upper() != "ALL":
        query_parts.append("pattern LIKE ?")
        params.append(f"%{pattern}%")

    if status and status.upper() != "ALL":
        if status.upper() in ("VALID", "1"):
            query_parts.append("is_valid = 1")
        elif status.upper() in ("INVALID", "0"):
            query_parts.append("is_valid = 0")

    if search:
        s_clean = f"%{search.strip()}%"
        query_parts.append("(plate_text LIKE ? OR province LIKE ? OR pattern LIKE ? OR record_id LIKE ?)")
        params.extend([s_clean, s_clean, s_clean, s_clean])

    where_clause = " AND ".join(query_parts)

    allowed_sorts = {
        "timestamp": "timestamp",
        "plate_text": "plate_text",
        "country": "country",
        "province": "province",
        "total_latency_ms": "total_latency_ms",
        "is_valid": "is_valid",
    }
    order_col = allowed_sorts.get(sort_by, "timestamp")
    order_dir = "ASC" if sort_order.lower() == "asc" else "DESC"

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        count_sql = f"SELECT COUNT(*) FROM recognition_history WHERE {where_clause}"
        cur.execute(count_sql, params)
        total_count = cur.fetchone()[0]

        data_sql = f"""
            SELECT * FROM recognition_history
            WHERE {where_clause}
            ORDER BY {order_col} {order_dir}
            LIMIT ? OFFSET ?
        """
        cur.execute(data_sql, params + [page_size, offset])
        rows = cur.fetchall()

        records = []
        for r in rows:
            rec = dict(r)
            try:
                rec["model_latencies"] = json.loads(rec.get("model_latencies") or "{}")
            except Exception:
                rec["model_latencies"] = {}
            records.append(rec)

        total_pages = max(1, (total_count + page_size - 1) // page_size)

        return {
            "records": records,
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "source": "sqlite",
        }
    finally:
        conn.close()


def get_history_stats(days: int = 7) -> Dict[str, Any]:
    """Computes summary KPI stats over the given window (Firestore-first)."""
    if _get_history_store() == "firestore":
        try:
            from src.cloud_storage_manager import cloud_storage_manager as csm
            cutoff = (datetime.now(TZ_BANGKOK) - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")
            docs = csm.query_firestore_filtered(limit=500)
            recent = [d for d in docs if str(d.get("timestamp", "")) >= cutoff]
            total = len(recent)
            thai = sum(1 for d in recent if d.get("country") == "Thai")
            lao = sum(1 for d in recent if d.get("country") == "Laos")
            valid = sum(1 for d in recent if d.get("is_valid", True))
            lats = [float(d.get("total_latency_ms") or 0) for d in recent if d.get("total_latency_ms")]
            latest = max((str(d.get("timestamp", "")) for d in recent), default="None")
            return {
                "window_days": days,
                "total_detections": total,
                "thai_count": thai,
                "lao_count": lao,
                "valid_count": valid,
                "valid_pct": round(valid / total * 100.0, 1) if total else 100.0,
                "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0.0,
                "latest_detection": latest,
                "source": "firestore",
            }
        except Exception as e:
            logger.warning(f"[History] Firestore stats failed, falling back to SQLite: {e}")
    return _get_history_stats_sqlite(days=days)


def _get_history_stats_sqlite(days: int = 7) -> Dict[str, Any]:
    """Computes summary KPI stats over the given window (SQLite)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")

        cur.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN country = 'Thai' THEN 1 ELSE 0 END) as thai_count,
                SUM(CASE WHEN country = 'Laos' THEN 1 ELSE 0 END) as lao_count,
                SUM(CASE WHEN is_valid = 1 THEN 1 ELSE 0 END) as valid_count,
                AVG(total_latency_ms) as avg_latency
            FROM recognition_history
            WHERE timestamp >= ?
            """,
            (cutoff_date,),
        )
        row = cur.fetchone()

        total = row["total"] or 0
        thai_count = row["thai_count"] or 0
        lao_count = row["lao_count"] or 0
        valid_count = row["valid_count"] or 0
        avg_latency = round(float(row["avg_latency"] or 0.0), 1)
        valid_pct = round((valid_count / float(total) * 100.0), 1) if total > 0 else 100.0

        cur.execute("SELECT timestamp, plate_text FROM recognition_history ORDER BY id DESC LIMIT 1")
        latest = cur.fetchone()
        latest_str = f"{latest['plate_text']} ({latest['timestamp']})" if latest else "None"

        return {
            "window_days": days,
            "total_detections": total,
            "thai_count": thai_count,
            "lao_count": lao_count,
            "valid_count": valid_count,
            "valid_pct": valid_pct,
            "avg_latency_ms": avg_latency,
            "latest_detection": latest_str,
        }
    finally:
        conn.close()


def export_history_csv(
    date_from: str = "",
    date_to: str = "",
    country: str = "",
    search: str = "",
) -> str:
    """Exports historical events as a CSV string."""
    res = query_history(
        page=1,
        page_size=10000,
        date_from=date_from,
        date_to=date_to,
        country=country,
        search=search,
        sort_by="timestamp",
        sort_order="desc",
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Record ID", "Timestamp", "Country", "License Plate",
        "Province", "Prov Prob (%)", "Pattern", "Valid Format",
        "Char Box Output", "CTC Output", "Total Latency (ms)"
    ])

    for r in res["records"]:
        writer.writerow([
            r["record_id"],
            r["timestamp"],
            r["country"],
            r["plate_text"],
            r["province"],
            r["province_prob"],
            r["pattern"],
            "VALID" if r["is_valid"] else "INVALID",
            r["char_box_text"],
            r["ctc_text"],
            r["total_latency_ms"],
        ])

    return output.getvalue()


def clear_history(older_than_days: Optional[int] = None) -> int:
    """Deletes historical logs (Firestore-first; SQLite fallback, mirrored when primary)."""
    deleted = 0
    store = _get_history_store()
    if store == "firestore":
        try:
            from src.cloud_storage_manager import cloud_storage_manager as csm
            deleted = csm.delete_firestore_records(older_than_days=older_than_days)
        except Exception as e:
            logger.warning(f"[History] Firestore clear failed, falling back to SQLite: {e}")
            store = "sqlite"
            deleted = 0
    if store == "sqlite":
        with _lock:
            conn = get_db_connection()
            try:
                with conn:
                    if older_than_days is not None:
                        cutoff = (datetime.now() - timedelta(days=older_than_days)).strftime("%Y-%m-%d 00:00:00")
                        cur = conn.execute("DELETE FROM recognition_history WHERE timestamp < ?", (cutoff,))
                    else:
                        cur = conn.execute("DELETE FROM recognition_history")
                    deleted = cur.rowcount
            finally:
                conn.close()
    with _recent_saves_lock:
        _recent_saves.clear()
    return deleted


# Auto-initialize database tables on module import
init_db()

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
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "lpr_history.db"

_lock = threading.Lock()


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
                # Migration: add raw_image column if older DB schema exists
                try:
                    conn.execute("ALTER TABLE recognition_history ADD COLUMN raw_image TEXT DEFAULT ''")
                except Exception:
                    pass

                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON recognition_history(timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_country ON recognition_history(country)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_plate ON recognition_history(plate_text)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_is_valid ON recognition_history(is_valid)")
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


def save_recognition(
    data: Dict[str, Any],
    thumbnail_bgr: Optional[np.ndarray] = None,
    raw_bgr: Optional[np.ndarray] = None,
) -> str:
    """
    Saves a recognized plate result to the history database.
    Robustly maps timing, confidence, crops, and full-resolution images.
    Returns the unique record_id.
    """
    record_id = f"lpr_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    now_dt = datetime.now()
    timestamp_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    created_at = time.time()

    plate_text = str(data.get("plate_text") or "").strip()
    country = str(data.get("country") or "Thai").strip()
    province = str(data.get("province") or "").strip()

    # Robust province probability extraction (percentage 0-100)
    prov_prob_raw = (
        data.get("province_prob")
        or (data.get("confidence") or {}).get("province_classification")
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
                        total_latency_ms, model_latencies, thumbnail, raw_image
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id, timestamp_str, created_at, plate_text, country,
                        province, province_prob, pattern, is_valid,
                        char_box_text, ctc_text, char_box_status, char_box_note,
                        total_latency_ms, model_latencies, thumbnail_b64, raw_b64,
                    ),
                )
        finally:
            conn.close()

    return record_id


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
        }
    finally:
        conn.close()


def get_history_stats(days: int = 7) -> Dict[str, Any]:
    """Computes summary KPI stats over the given window."""
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
    """Deletes historical logs (either older than N days or completely)."""
    with _lock:
        conn = get_db_connection()
        try:
            with conn:
                if older_than_days is not None:
                    cutoff = (datetime.now() - timedelta(days=older_than_days)).strftime("%Y-%m-%d 00:00:00")
                    cur = conn.execute("DELETE FROM recognition_history WHERE timestamp < ?", (cutoff,))
                else:
                    cur = conn.execute("DELETE FROM recognition_history")
                return cur.rowcount
        finally:
            conn.close()


# Auto-initialize database tables on module import
init_db()

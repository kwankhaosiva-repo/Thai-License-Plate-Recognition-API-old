"""
src/cloud_storage_manager.py

Unified Google Cloud Platform integration for:
1. Google Cloud Firestore: Database 'lpr-db', Collection 'recognition_history'
   Stores rich documents including plate metadata, validation status, thumbnails, and logged-in user profiles.
2. Google Cloud BigQuery: Dataset 'lpr_query', Table 'lpr_history'
   Inserts analytical rows (plate_number, province, confidence, timestamp) for high-performance querying and reporting.

Executes writes asynchronously via a background thread pool to ensure zero impact on inference latency.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import cfg

logger = logging.getLogger(__name__)


TZ_BANGKOK = datetime.timezone(datetime.timedelta(hours=7), name="Asia/Bangkok")


class CloudStorageManager:
    """Thread-safe Google Cloud Platform manager for Firestore & BigQuery."""

    _instance: Optional[CloudStorageManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> CloudStorageManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CloudStorageManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.project_id: str = getattr(cfg, "GCP_PROJECT_ID", "lpr-car-plate")
        self.firestore_db_id: str = getattr(cfg, "FIRESTORE_DATABASE_ID", "lpr-db")
        self.firestore_collection: str = getattr(cfg, "FIRESTORE_COLLECTION", "recognition_history")
        self.bigquery_dataset: str = getattr(cfg, "BIGQUERY_DATASET", "lpr_query")
        self.bigquery_table: str = getattr(cfg, "BIGQUERY_TABLE", "lpr_history")
        self.enabled: bool = getattr(cfg, "GCP_CLOUD_SYNC_ENABLED", True)

        self._fs_client = None
        self._bq_client = None
        self._credentials = None
        self._auth_resolved = False

        # Non-blocking async background executor
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gcp_cloud_sync")
        self._initialized = True
        self._resolve_credentials()

    def _resolve_credentials(self) -> None:
        """Resolves service account credentials from keyfile or environment."""
        if not self.enabled:
            return

        try:
            from google.oauth2 import service_account

            key_filename = getattr(cfg, "GCP_KEY_FILENAME", "lpr-car-plate-e4c1f71338f3.json")
            project_root = getattr(cfg, "PROJECT_ROOT", Path(__file__).resolve().parent.parent)
            key_path = Path(key_filename) if Path(key_filename).is_absolute() else (project_root / key_filename)

            if key_path.exists():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)
                self._credentials = service_account.Credentials.from_service_account_file(str(key_path))
                self._auth_resolved = True
                logger.info(f"[GCP CloudSync] Loaded service account credentials from {key_path.name}")
            elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                env_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
                if Path(env_path).exists():
                    self._credentials = service_account.Credentials.from_service_account_file(env_path)
                    self._auth_resolved = True
                    logger.info(f"[GCP CloudSync] Loaded credentials from env GOOGLE_APPLICATION_CREDENTIALS={env_path}")
            else:
                # Cloud Run default runtime service account
                import google.auth
                self._credentials, _ = google.auth.default()
                self._auth_resolved = True
                logger.info("[GCP CloudSync] Using Google Application Default Credentials")
        except Exception as e:
            logger.warning(f"[GCP CloudSync] Failed to resolve GCP credentials: {e}")
            self._auth_resolved = False

    def _get_firestore(self):
        """Returns lazy-initialized Firestore Client."""
        if not self.enabled:
            return None
        if self._fs_client is None:
            try:
                from google.cloud import firestore
                self._fs_client = firestore.Client(
                    project=self.project_id,
                    credentials=self._credentials,
                    database=self.firestore_db_id,
                )
            except Exception as e:
                logger.error(f"[GCP Firestore] Client initialization error: {e}")
                self._fs_client = None
        return self._fs_client

    def _get_bigquery(self):
        """Returns lazy-initialized BigQuery Client."""
        if not self.enabled:
            return None
        if self._bq_client is None:
            try:
                from google.cloud import bigquery
                self._bq_client = bigquery.Client(
                    project=self.project_id,
                    credentials=self._credentials,
                )
            except Exception as e:
                logger.error(f"[GCP BigQuery] Client initialization error: {e}")
                self._bq_client = None
        return self._bq_client

    def _ensure_bigquery_table(self, bq) -> None:
        """Checks if the BigQuery table exists; if not, automatically creates it with proper schema and 90-day partitioning."""
        if bq is None:
            return
        full_table_id = f"{self.project_id}.{self.bigquery_dataset}.{self.bigquery_table}"
        try:
            bq.get_table(full_table_id)
        except Exception:
            try:
                from google.cloud import bigquery
                schema = [
                    bigquery.SchemaField("record_id", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
                    bigquery.SchemaField("plate_number", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("province", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("country", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("confidence", "FLOAT", mode="NULLABLE"),
                    bigquery.SchemaField("is_valid", "BOOLEAN", mode="NULLABLE"),
                    bigquery.SchemaField("pattern", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("user_email", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("latency_ms", "INTEGER", mode="NULLABLE"),
                    bigquery.SchemaField("source_mode", "STRING", mode="NULLABLE"),
                ]
                table = bigquery.Table(full_table_id, schema=schema)
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.HOUR,
                    expiration_ms=90 * 24 * 3600 * 1000,  # 90 days
                )
                bq.create_table(table, exists_ok=True)
                logger.info(f"[BigQuery Auto-Create] Successfully created table {full_table_id} with 11 columns and 90-day expiration!")
            except Exception as ce:
                logger.error(f"[BigQuery Auto-Create Error] Failed to create table: {ce}")

    def sync_recognition(
        self,
        record_dict: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Asynchronously syncs recognition record to Firestore (lpr-db) and BigQuery (lpr_query.lpr_history).
        Dispatched to a background worker to avoid blocking API response time.
        """
        if not self.enabled or not record_dict:
            return

        # Deep copy essential primitive fields before background execution
        safe_copy = {
            "record_id": record_dict.get("record_id") or record_dict.get("id") or "",
            "plate_text": record_dict.get("plate_text", ""),
            "country": record_dict.get("country", "Thai"),
            "province": record_dict.get("province", ""),
            "province_prob": float(record_dict.get("province_prob") or 0.0),
            "confidence": float(
                record_dict.get("confidence", {}).get("plate_detection", 0.0)
                if isinstance(record_dict.get("confidence"), dict)
                else (record_dict.get("confidence") or 0.0)
            ),
            "is_valid": bool(record_dict.get("is_valid", True)),
            "pattern": record_dict.get("pattern", "UNKNOWN"),
            "total_latency_ms": int(record_dict.get("total_latency_ms") or record_dict.get("timing", {}).get("total_ms", 0)),
            "timestamp": record_dict.get("timestamp") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "thumbnail": record_dict.get("thumbnail") or record_dict.get("crops", {}).get("plate_rectified", ""),
            "source_mode": record_dict.get("source_mode") or ("rtsp_stream" if record_dict.get("stream_id") else "image_upload"),
        }

        user_info = {
            "user_id": user_profile.get("uid", "guest") if user_profile else "guest",
            "user_email": user_profile.get("email", "") if user_profile else "",
            "user_name": user_profile.get("name", "Guest Visitor") if user_profile else "Guest Visitor",
            "user_role": user_profile.get("role", "viewer") if user_profile else "viewer",
        }

        self._executor.submit(self._execute_sync, safe_copy, user_info)

    def _execute_sync(self, record: Dict[str, Any], user: Dict[str, Any]) -> None:
        """Background worker executing Firestore and BigQuery sync operations."""
        plate_text = record.get("plate_text", "").strip()
        if not plate_text:
            return

        # 1. Sync to Google Cloud Firestore (lpr-db)
        try:
            fs = self._get_firestore()
            if fs is not None:
                doc_id = record.get("record_id") or f"rec_{int(datetime.datetime.now().timestamp() * 1000)}"
                doc_payload = {
                    "record_id": doc_id,
                    "plate_text": plate_text,
                    "country": record.get("country", "Thai"),
                    "province": record.get("province", ""),
                    "province_prob": record.get("province_prob", 0.0),
                    "confidence": record.get("confidence", 0.0),
                    "is_valid": record.get("is_valid", True),
                    "pattern": record.get("pattern", "UNKNOWN"),
                    "total_latency_ms": record.get("total_latency_ms", 0),
                    "timestamp": record.get("timestamp"),
                    "thumbnail": record.get("thumbnail", ""),
                    "source_mode": record.get("source_mode", "image_upload"),
                    "user_id": user.get("user_id", "guest"),
                    "user_email": user.get("user_email", ""),
                    "user_name": user.get("user_name", ""),
                    "user_role": user.get("user_role", ""),
                    "synced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                fs.collection(self.firestore_collection).document(doc_id).set(doc_payload)
                logger.info(f"[Firestore Sync] Saved {plate_text} (User: {user.get('user_email') or 'guest'}) to {self.firestore_collection}/{doc_id}")
        except Exception as e:
            logger.warning(f"[Firestore Sync Error] Failed to write record: {e}")

        # 2. Sync to Google Cloud BigQuery (lpr_query.lpr_history)
        try:
            bq = self._get_bigquery()
            if bq is not None:
                full_table_id = f"{self.project_id}.{self.bigquery_dataset}.{self.bigquery_table}"
                self._ensure_bigquery_table(bq)

                # Parse timestamp for BigQuery TIMESTAMP column
                raw_ts = record.get("timestamp")
                if isinstance(raw_ts, str):
                    try:
                        if " " in raw_ts and "T" not in raw_ts:
                            dt_local = datetime.datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S").astimezone()
                            clean_ts = dt_local.astimezone(datetime.timezone.utc).isoformat()
                        elif not raw_ts.endswith("Z") and "+" not in raw_ts:
                            clean_ts = raw_ts.replace(" ", "T") + "Z"
                        else:
                            clean_ts = raw_ts.replace(" ", "T")
                    except Exception:
                        clean_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                else:
                    clean_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

                bq_row = {
                    "record_id": record.get("record_id") or "",
                    "timestamp": clean_ts,
                    "plate_number": plate_text,
                    "province": record.get("province", "") or "Unknown",
                    "country": record.get("country", "Thai") or "Thai",
                    "confidence": float(record.get("confidence", 0.0)),
                    "is_valid": bool(record.get("is_valid", True)),
                    "pattern": record.get("pattern", "Standard Private Car"),
                    "user_email": user.get("user_email", ""),
                    "latency_ms": int(record.get("total_latency_ms", 0)),
                    "source_mode": record.get("source_mode", "image_upload"),
                }

                errors = bq.insert_rows_json(full_table_id, [bq_row])
                if errors:
                    logger.warning(f"[BigQuery Sync Error] Insert errors: {errors}")
                else:
                    logger.info(f"[BigQuery Sync] Inserted row {plate_text} ({bq_row['province']} | {bq_row['user_email']}) into {full_table_id}")
        except Exception as e:
            logger.warning(f"[BigQuery Sync Error] Failed to insert row: {e}")

    def query_bigquery_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Queries the latest recognition records directly from BigQuery table."""
        bq = self._get_bigquery()
        if bq is None:
            return []

        try:
            self._ensure_bigquery_table(bq)
            full_table_id = f"`{self.project_id}.{self.bigquery_dataset}.{self.bigquery_table}`"
            sql = f"""
                SELECT record_id, timestamp, plate_number, province, country, confidence, is_valid, pattern, user_email, latency_ms, source_mode
                FROM {full_table_id}
                ORDER BY timestamp DESC
                LIMIT {min(max(1, limit), 500)}
            """
            query_job = bq.query(sql)
            records = []
            for row in query_job:
                ts_val = row.get("timestamp")
                if hasattr(ts_val, "astimezone"):
                    ts_str = ts_val.astimezone(TZ_BANGKOK).strftime("%Y-%m-%d %H:%M:%S")
                elif hasattr(ts_val, "isoformat"):
                    ts_str = ts_val.isoformat()
                else:
                    ts_str = str(ts_val)
                records.append({
                    "record_id": row.get("record_id") or "",
                    "timestamp": ts_str,
                    "plate_number": row.get("plate_number", ""),
                    "province": row.get("province", ""),
                    "country": row.get("country", "Thai"),
                    "confidence": round(float(row.get("confidence", 0.0)), 4),
                    "is_valid": bool(row.get("is_valid", True)),
                    "pattern": row.get("pattern", ""),
                    "user_email": row.get("user_email", ""),
                    "latency_ms": int(row.get("latency_ms") or 0),
                    "source_mode": row.get("source_mode", "image_upload"),
                })
            return records
        except Exception as e:
            logger.error(f"[BigQuery Query Error] {e}")
            return []

    def query_firestore_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Queries the latest recognition documents from Firestore collection."""
        fs = self._get_firestore()
        if fs is None:
            return []

        try:
            from google.cloud import firestore
            docs = (
                fs.collection(self.firestore_collection)
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(min(max(1, limit), 200))
                .stream()
            )
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"[Firestore Query Error] {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Returns live GCP connection health and collection metadata."""
        fs_ok = False
        bq_ok = False
        bq_rows = 0

        try:
            fs = self._get_firestore()
            if fs is not None:
                # Test ping by listing collections
                _ = list(fs.collections())
                fs_ok = True
        except Exception as e:
            logger.debug(f"Firestore health check failed: {e}")

        try:
            bq = self._get_bigquery()
            if bq is not None:
                self._ensure_bigquery_table(bq)
                full_table_id = f"{self.project_id}.{self.bigquery_dataset}.{self.bigquery_table}"
                tbl = bq.get_table(full_table_id)
                bq_rows = getattr(tbl, "num_rows", 0)
                bq_ok = True
        except Exception as e:
            logger.debug(f"BigQuery health check failed: {e}")

        return {
            "enabled": self.enabled,
            "project_id": self.project_id,
            "is_cloud_run": bool(os.environ.get("K_SERVICE") or os.environ.get("CLOUD_RUN_JOB")),
            "credentials_loaded": self._auth_resolved,
            "firestore": {
                "database_id": self.firestore_db_id,
                "collection": self.firestore_collection,
                "connected": fs_ok,
            },
            "bigquery": {
                "dataset": self.bigquery_dataset,
                "table": self.bigquery_table,
                "connected": bq_ok,
                "total_rows": bq_rows,
            },
        }

    def sync_firestore_to_bigquery(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Idempotent bulk ETL synchronization:
        Reads records from Firestore collection and inserts missing records into BigQuery table.
        Safe for periodic Cloud Scheduler triggering or on-demand execution.
        """
        fs = self._get_firestore()
        bq = self._get_bigquery()

        if not fs or not bq:
            return {
                "status": "error",
                "message": "GCP Firestore or BigQuery client unavailable",
                "newly_synced_rows": 0,
            }

        full_table_id = f"{self.project_id}.{self.bigquery_dataset}.{self.bigquery_table}"
        self._ensure_bigquery_table(bq)
        bq_table_ref = f"`{full_table_id}`"

        # 1. Fetch existing record_ids in BigQuery to prevent duplicate insertions
        existing_ids = set()
        try:
            check_sql = f"SELECT DISTINCT record_id FROM {bq_table_ref} WHERE record_id IS NOT NULL AND record_id != ''"
            query_job = bq.query(check_sql)
            for row in query_job:
                rid = row.get("record_id")
                if rid:
                    existing_ids.add(rid)
        except Exception as e:
            logger.warning(f"[CloudSync] Could not pre-query existing BigQuery IDs: {e}")

        # 2. Fetch documents from Firestore
        docs = list(fs.collection(self.firestore_collection).limit(limit).stream())
        new_rows = []

        for doc in docs:
            d = doc.to_dict()
            doc_id = d.get("record_id") or doc.id
            if doc_id in existing_ids:
                continue

            plate_text = d.get("plate_text", "").strip()
            if not plate_text:
                continue

            # Format UTC timestamp for BigQuery
            raw_ts = d.get("timestamp")
            if isinstance(raw_ts, str):
                try:
                    if " " in raw_ts and "T" not in raw_ts:
                        dt_bkk = datetime.datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ_BANGKOK)
                        clean_ts = dt_bkk.isoformat()
                    elif not raw_ts.endswith("Z") and "+" not in raw_ts:
                        clean_ts = raw_ts.replace(" ", "T") + "+07:00"
                    else:
                        clean_ts = raw_ts.replace(" ", "T")
                except Exception:
                    clean_ts = datetime.datetime.now(TZ_BANGKOK).isoformat()
            else:
                clean_ts = datetime.datetime.now(TZ_BANGKOK).isoformat()

            new_rows.append({
                "record_id": doc_id,
                "timestamp": clean_ts,
                "plate_number": plate_text,
                "province": d.get("province", "") or "Unknown",
                "country": d.get("country", "Thai") or "Thai",
                "confidence": float(d.get("confidence", 0.0)),
                "is_valid": bool(d.get("is_valid", True)),
                "pattern": d.get("pattern", "Standard Private Car"),
                "user_email": d.get("user_email", ""),
                "latency_ms": int(d.get("total_latency_ms", 0)),
                "source_mode": d.get("source_mode", "image_upload"),
            })

        # 3. Batch insert missing rows into BigQuery
        inserted_count = 0
        if new_rows:
            errors = bq.insert_rows_json(full_table_id, new_rows)
            if errors:
                logger.error(f"[CloudSync] Batch insert errors into BigQuery: {errors}")
                return {
                    "status": "partial_error",
                    "errors": errors,
                    "attempted": len(new_rows),
                    "already_synced": len(existing_ids),
                }
            inserted_count = len(new_rows)
            logger.info(f"[CloudSync] Successfully synced {inserted_count} records from Firestore to BigQuery")

        return {
            "status": "success",
            "firestore_total_docs": len(docs),
            "already_in_bigquery": len(existing_ids),
            "newly_synced_rows": inserted_count,
            "target_table": full_table_id,
        }


# Singleton instance accessor
cloud_storage_manager = CloudStorageManager()

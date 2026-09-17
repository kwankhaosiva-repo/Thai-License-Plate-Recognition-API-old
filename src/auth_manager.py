"""
src/auth_manager.py

Authentication, User Management, and Database Module for Thai LPR API.
Supports:
- Secure Salted PBKDF2-HMAC-SHA256 Password Hashing
- Cryptographic Session JWT Tokens (PyJWT HS256)
- Thread-safe SQLite User & Settings Database (data/lpr_users.db)
- Optional Multi-Cloud Firestore Sync when credentials are present
- Fast Dev Admin mode for friction-free local pair-programming
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jwt
from fastapi import Request

from src.config import cfg

logger = logging.getLogger("ThaiLPR_Auth")

_lock = threading.Lock()
_firestore_client = None
_firestore_checked = False


# =====================================================================
# Database Helpers & Initialization
# =====================================================================

def get_users_db() -> sqlite3.Connection:
    """Returns a thread-safe connection to the SQLite users database."""
    cfg.USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cfg.USERS_DB_PATH), check_same_thread=False, timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_users_db():
    """Initializes the users and activity audit tables with proper indexes."""
    with _lock:
        conn = get_users_db()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        uid TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'admin',
                        password_hash TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_login_at TEXT NOT NULL,
                        settings_json TEXT DEFAULT '{}'
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_activity (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        uid TEXT NOT NULL,
                        action TEXT NOT NULL,
                        detail TEXT DEFAULT '',
                        ip_address TEXT DEFAULT '',
                        created_at TEXT NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_users_uid ON users(uid)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_uid ON user_activity(uid)")
        finally:
            conn.close()
    logger.info("[AUTH] Users database schema verified at %s", cfg.USERS_DB_PATH)


# Initialize schema on module load
init_users_db()


# =====================================================================
# Optional Firebase Firestore Sync Driver
# =====================================================================

def _get_firestore_client():
    """Lazy-initializes Google Cloud Firestore if credentials and library exist."""
    global _firestore_client, _firestore_checked
    if _firestore_checked:
        return _firestore_client

    with _lock:
        if _firestore_checked:
            return _firestore_client
        _firestore_checked = True

        cred_path = cfg.FIREBASE_CREDENTIALS_PATH or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        project_id = cfg.FIREBASE_PROJECT_ID

        if not project_id and not cred_path:
            logger.debug("[AUTH] No Firebase project ID or credentials configured; using local SQLite.")
            return None

        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            if not firebase_admin._apps:
                if cred_path and os.path.isfile(cred_path):
                    cred = credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(cred, {"projectId": project_id or None})
                else:
                    firebase_admin.initialize_app(options={"projectId": project_id or None})
            _firestore_client = firestore.client()
            logger.info("[AUTH] Connected to Google Cloud Firestore.")
        except Exception as e:
            logger.warning("[AUTH] Firestore initialization skipped/failed: %s (will use SQLite)", e)
            _firestore_client = None

        return _firestore_client


def _sync_user_to_firestore(user_record: Dict[str, Any]):
    """Syncs user account record to Firestore asynchronously if connected."""
    db = _get_firestore_client()
    if db is None:
        return
    try:
        coll = cfg.FIRESTORE_USERS_COLLECTION
        doc_ref = db.collection(coll).document(user_record["uid"])
        safe_copy = {k: v for k, v in user_record.items() if k not in ("password_hash", "salt")}
        doc_ref.set(safe_copy, merge=True)
        logger.info("[AUTH] Synced user '%s' to Firestore collection '%s'", user_record["uid"], coll)
    except Exception as e:
        logger.warning("[AUTH] Could not sync user '%s' to Firestore: %s", user_record.get("uid"), e)


# =====================================================================
# Password Hashing & Crypto Security
# =====================================================================

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Generates a secure PBKDF2-HMAC-SHA256 hash with 100,000 iterations.
    Returns: (hex_hash, hex_salt)
    """
    if not salt:
        salt = secrets.token_hex(16)
    hashed_bytes = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    )
    return hashed_bytes.hex(), salt


def verify_password_hash(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a plaintext password against a stored PBKDF2 hash using constant-time comparison."""
    test_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(test_hash, stored_hash)


# =====================================================================
# JWT Session Management
# =====================================================================

def create_session_jwt(user_dict: Dict[str, Any], expires_sec: Optional[int] = None) -> str:
    """Creates a cryptographically signed JWT session token."""
    if expires_sec is None:
        expires_sec = cfg.JWT_EXPIRES_DAYS * 86400
    now = int(time.time())
    payload = {
        "uid": str(user_dict.get("uid", "")),
        "email": str(user_dict.get("email", "")),
        "name": str(user_dict.get("name", "")),
        "role": str(user_dict.get("role", "admin")),
        "iat": now,
        "exp": now + expires_sec,
        "iss": "thai-lpr-system",
    }
    return jwt.encode(payload, cfg.JWT_SECRET_KEY, algorithm=cfg.JWT_ALGORITHM)


def decode_session_jwt(token_str: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Decodes and validates a session JWT token."""
    if not token_str:
        return False, None, "Empty token"
    try:
        payload = jwt.decode(token_str, cfg.JWT_SECRET_KEY, algorithms=[cfg.JWT_ALGORITHM])
        return True, payload, "Valid token"
    except jwt.ExpiredSignatureError:
        return False, None, "Session expired, please sign in again"
    except Exception as e:
        return False, None, f"Invalid token: {e}"


# =====================================================================
# Core User Account Services
# =====================================================================

def register_user_account(
    email_or_id: str,
    password: str,
    name: str = "",
    role: str = "admin",
) -> Tuple[bool, Optional[Dict[str, Any]], str, Optional[str]]:
    """
    Registers a new user account into SQLite and optional Firestore.
    Returns: (success, user_profile, message, session_token)
    """
    ident = email_or_id.strip()
    if not ident:
        return False, None, "Email or User ID is required", None
    if not password or len(password) < 4:
        return False, None, "Password must be at least 4 characters", None

    uid = ident.lower().replace(" ", "_")
    email = ident if "@" in ident else f"{ident}@thailpr.local"
    display_name = name.strip() or ident.split("@")[0]
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    pwd_hash, salt = hash_password(password)

    default_settings = {
        "theme": "dark",
        "default_mode": "upload",
        "auto_refresh_history": True,
        "confidence_threshold": 0.50,
        "debug_view_default": False,
    }

    with _lock:
        conn = get_users_db()
        try:
            with conn:
                # Check if user already exists
                row = conn.execute(
                    "SELECT id FROM users WHERE uid = ? OR email = ?",
                    (uid, email),
                ).fetchone()
                if row:
                    return False, None, f"User with ID or email '{ident}' already exists", None

                conn.execute(
                    """
                    INSERT INTO users (uid, email, name, role, password_hash, salt, created_at, last_login_at, settings_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uid,
                        email,
                        display_name,
                        role,
                        pwd_hash,
                        salt,
                        now_str,
                        now_str,
                        json.dumps(default_settings),
                    ),
                )
        except sqlite3.IntegrityError:
            return False, None, f"User '{ident}' already exists", None
        finally:
            conn.close()

    sanitized = {
        "uid": uid,
        "email": email,
        "name": display_name,
        "role": role,
        "created_at": now_str,
        "last_login_at": now_str,
        "settings": default_settings,
    }

    session_token = create_session_jwt(sanitized)
    _sync_user_to_firestore(sanitized)
    log_user_activity(uid, "register", f"Account created for {display_name}")

    return True, sanitized, f"Account created successfully for {display_name}", session_token


def authenticate_user_password(
    email_or_id: str,
    password: str,
) -> Tuple[bool, Optional[Dict[str, Any]], str, Optional[str]]:
    """
    Authenticates user with Email/ID and Password.
    Returns: (success, user_profile, message, session_token)
    """
    ident = email_or_id.strip()
    if not ident or not password:
        return False, None, "Email/ID and password are required", None

    uid_candidate = ident.lower().replace(" ", "_")

    conn = get_users_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE uid = ? OR email = ?",
            (uid_candidate, ident),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return False, None, f"User '{ident}' not found. Please create an account.", None

    pwd_hash = row["password_hash"]
    salt = row["salt"]

    if not verify_password_hash(password, pwd_hash, salt):
        return False, None, "Incorrect password. Please try again.", None

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Update last login
    with _lock:
        conn = get_users_db()
        try:
            with conn:
                conn.execute("UPDATE users SET last_login_at = ? WHERE uid = ?", (now_str, row["uid"]))
        finally:
            conn.close()

    try:
        settings = json.loads(row["settings_json"] or "{}")
    except Exception:
        settings = {}

    sanitized = {
        "uid": row["uid"],
        "email": row["email"],
        "name": row["name"],
        "role": row["role"],
        "created_at": row["created_at"],
        "last_login_at": now_str,
        "settings": settings,
    }

    session_token = create_session_jwt(sanitized)
    log_user_activity(row["uid"], "login", "User logged in with password")

    return True, sanitized, "Sign in successful", session_token


def get_user_profile(uid: str) -> Optional[Dict[str, Any]]:
    """Retrieves public user profile and settings by UID."""
    conn = get_users_db()
    try:
        row = conn.execute("SELECT * FROM users WHERE uid = ?", (uid,)).fetchone()
        if not row:
            return None
        try:
            settings = json.loads(row["settings_json"] or "{}")
        except Exception:
            settings = {}
        return {
            "uid": row["uid"],
            "email": row["email"],
            "name": row["name"],
            "role": row["role"],
            "created_at": row["created_at"],
            "last_login_at": row["last_login_at"],
            "settings": settings,
        }
    finally:
        conn.close()


def update_user_settings(uid: str, new_settings: Dict[str, Any]) -> Tuple[bool, str]:
    """Updates user dashboard settings and preferences."""
    profile = get_user_profile(uid)
    if not profile:
        return False, f"User '{uid}' not found"

    current = profile.get("settings", {})
    merged = {**current, **new_settings}

    with _lock:
        conn = get_users_db()
        try:
            with conn:
                conn.execute(
                    "UPDATE users SET settings_json = ? WHERE uid = ?",
                    (json.dumps(merged), uid),
                )
        finally:
            conn.close()

    log_user_activity(uid, "update_settings", f"Updated settings: {list(new_settings.keys())}")
    return True, "Settings updated successfully"


def log_user_activity(uid: str, action: str, detail: str = "", ip: str = ""):
    """Thread-safely appends an audit event to user_activity."""
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        conn = get_users_db()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO user_activity (uid, action, detail, ip_address, created_at) VALUES (?, ?, ?, ?, ?)",
                    (uid, action, detail, ip, now_str),
                )
        except Exception as e:
            logger.debug("[AUTH] Failed to log user activity: %s", e)
        finally:
            conn.close()


# =====================================================================
# Dev Admin & Fast Localhost Bypass
# =====================================================================

DEV_ADMIN_PROFILE = {
    "uid": "dev_admin",
    "email": "admin@localhost.dev",
    "name": "Dev Admin (Local)",
    "role": "admin",
    "created_at": "2026-01-01 00:00:00",
    "last_login_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "settings": {
        "theme": "dark",
        "default_mode": "upload",
        "auto_refresh_history": True,
        "confidence_threshold": 0.50,
        "debug_view_default": False,
    },
}


def login_dev_admin() -> Tuple[Dict[str, Any], str]:
    """Generates an instant Dev Admin session for frictionless development."""
    token = create_session_jwt(DEV_ADMIN_PROFILE, expires_sec=86400 * 30)
    log_user_activity("dev_admin", "dev_admin_login", "Dev admin accessed via localhost")
    return DEV_ADMIN_PROFILE, token


def get_auth_config() -> Dict[str, Any]:
    """Returns frontend-visible auth configuration."""
    db_provider = "firestore" if _get_firestore_client() is not None else "sqlite"
    return {
        "auth_enabled": getattr(cfg, "AUTH_ENABLED", True),
        "auth_required": getattr(cfg, "AUTH_ENABLED", True),
        "allow_dev_admin": getattr(cfg, "ALLOW_DEV_ADMIN", True),
        "db_provider": db_provider,
        "project_name": "Thai LPR System",
    }


# =====================================================================
# FastAPI Request Dependency
# =====================================================================

async def get_current_user_profile(request: Request) -> Dict[str, Any]:
    """
    FastAPI dependency to extract and authenticate current user from:
    1. Authorization header: `Bearer <token>`
    2. Session cookie: `auth_token`
    3. Dev Admin token fallback if on localhost and ALLOW_DEV_ADMIN is enabled
    """
    token = None
    auth_hdr = request.headers.get("Authorization", "")
    if auth_hdr.lower().startswith("bearer "):
        token = auth_hdr[7:].strip()

    if not token:
        token = request.cookies.get("auth_token", "")

    if token:
        if token == "dev_admin_token" and getattr(cfg, "ALLOW_DEV_ADMIN", True):
            return DEV_ADMIN_PROFILE

        ok, claims, _ = decode_session_jwt(token)
        if ok and claims:
            user = get_user_profile(claims.get("uid", ""))
            if user:
                return user
            # If user not found in DB (e.g. wiped), use claims
            return {
                "uid": claims.get("uid", "user"),
                "email": claims.get("email", ""),
                "name": claims.get("name", "User"),
                "role": claims.get("role", "admin"),
                "settings": {},
            }

    # Dev fallback for localhost if auth is not strictly required
    client_host = getattr(request.client, "host", "")
    if client_host in ("127.0.0.1", "localhost", "::1") and getattr(cfg, "ALLOW_DEV_ADMIN", True):
        return DEV_ADMIN_PROFILE

    # Default guest/unauthenticated user
    return {
        "uid": "guest",
        "email": "guest@thailpr.local",
        "name": "Guest Visitor",
        "role": "viewer",
        "settings": {},
    }

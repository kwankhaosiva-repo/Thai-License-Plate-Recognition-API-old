"""
tests/test_auth.py
Unit tests for Authentication, Password Hashing, JWT Tokens, and SQLite User DB.
"""
import os
import unittest
from pathlib import Path

from src.auth_manager import (
    authenticate_user_password,
    create_session_jwt,
    decode_session_jwt,
    get_auth_config,
    get_user_profile,
    hash_password,
    init_users_db,
    login_dev_admin,
    register_user_account,
    update_user_settings,
    verify_password_hash,
)
from src.config import cfg


class TestAuthManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure clean test DB
        init_users_db()

    def test_password_hashing(self):
        pwd = "SecretPassword123"
        pwd_hash, salt = hash_password(pwd)
        self.assertNotEqual(pwd, pwd_hash)
        self.assertTrue(verify_password_hash(pwd, pwd_hash, salt))
        self.assertFalse(verify_password_hash("WrongPassword", pwd_hash, salt))

    def test_jwt_creation_and_decoding(self):
        user = {
            "uid": "test_user_jwt",
            "email": "test@jwt.local",
            "name": "Test JWT",
            "role": "admin",
        }
        token = create_session_jwt(user, expires_sec=3600)
        self.assertIsInstance(token, str)
        self.assertTrue(len(token) > 20)

        ok, claims, msg = decode_session_jwt(token)
        self.assertTrue(ok)
        self.assertEqual(claims["uid"], "test_user_jwt")
        self.assertEqual(claims["email"], "test@jwt.local")
        self.assertEqual(claims["role"], "admin")

    def test_user_registration_and_login(self):
        uid = "test_driver_99"
        email = "driver99@test.com"
        pwd = "MySecurePassword2026"
        name = "Driver 99"

        # Register
        ok, user, msg, token = register_user_account(email, pwd, name=name, role="operator")
        if not ok and "already exists" in msg:
            # User was created in a previous run, continue to login test
            pass
        else:
            self.assertTrue(ok)
            self.assertEqual(user["name"], name)
            self.assertEqual(user["role"], "operator")
            self.assertTrue(token is not None)

        # Login with correct password
        ok_login, user_login, msg_login, token_login = authenticate_user_password(email, pwd)
        self.assertTrue(ok_login)
        self.assertEqual(user_login["name"], name)
        self.assertTrue(token_login is not None)

        # Login with wrong password
        ok_fail, _, msg_fail, _ = authenticate_user_password(email, "WrongPassword")
        self.assertFalse(ok_fail)
        self.assertIn("Incorrect password", msg_fail)

    def test_user_settings_update(self):
        uid = "settings_user_01"
        register_user_account(uid, "Pass1234", name="Settings User")

        profile_before = get_user_profile(uid)
        self.assertIsNotNone(profile_before)

        # Update settings
        ok, msg = update_user_settings(uid, {"theme": "cyberpunk", "confidence_threshold": 0.75})
        self.assertTrue(ok)

        profile_after = get_user_profile(uid)
        self.assertEqual(profile_after["settings"]["theme"], "cyberpunk")
        self.assertEqual(profile_after["settings"]["confidence_threshold"], 0.75)

    def test_dev_admin_login(self):
        profile, token = login_dev_admin()
        self.assertEqual(profile["uid"], "dev_admin")
        self.assertEqual(profile["role"], "admin")
        ok, claims, _ = decode_session_jwt(token)
        self.assertTrue(ok)
        self.assertEqual(claims["uid"], "dev_admin")

    def test_auth_config(self):
        config = get_auth_config()
        self.assertIn("auth_enabled", config)
        self.assertIn("db_provider", config)
        self.assertIn(config["db_provider"], ("sqlite", "firestore"))


if __name__ == "__main__":
    unittest.main()

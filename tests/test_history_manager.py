"""
tests/test_history_manager.py

Unit tests for SQLite Recognition History & Data Collection Manager.
"""

import unittest
import numpy as np
from src.history_manager import (
    save_recognition,
    query_history,
    get_history_stats,
    export_history_csv,
    clear_history,
)


class TestHistoryManager(unittest.TestCase):
    def setUp(self):
        clear_history()

    def test_save_and_query_record(self):
        mock_thumb = np.zeros((40, 100, 3), dtype=np.uint8)
        rid = save_recognition(
            {
                "plate_text": "83-0015",
                "country": "Thai",
                "province": "ราชบุรี",
                "province_prob": 99.9,
                "pattern": "NN-NNNN (Truck/Transport)",
                "is_valid": True,
                "char_box_text": "83005",
                "ctc_text": "83-0015",
                "total_latency_ms": 380,
                "model_latencies": {"m1": 120, "m2": 110, "m3": 150},
            },
            thumbnail_bgr=mock_thumb,
        )
        self.assertTrue(rid.startswith("lpr_"))

        res = query_history(page=1, page_size=10)
        self.assertEqual(res["total"], 1)
        rec = res["records"][0]
        self.assertEqual(rec["plate_text"], "83-0015")
        self.assertEqual(rec["province"], "ราชบุรี")
        self.assertTrue(rec["thumbnail"].startswith("data:image/jpeg;base64,"))

    def test_filtering_and_stats(self):
        save_recognition({"plate_text": "1กข 1234", "country": "Thai", "is_valid": True, "total_latency_ms": 200})
        save_recognition({"plate_text": "ກກ 0083", "country": "Laos", "is_valid": True, "total_latency_ms": 300})
        save_recognition({"plate_text": "XXXXX", "country": "Thai", "is_valid": False, "total_latency_ms": 250})

        thai_res = query_history(country="Thai")
        self.assertEqual(thai_res["total"], 2)

        lao_res = query_history(country="Laos")
        self.assertEqual(lao_res["total"], 1)

        valid_res = query_history(status="VALID")
        self.assertEqual(valid_res["total"], 2)

        stats = get_history_stats(days=7)
        self.assertEqual(stats["total_detections"], 3)
        self.assertEqual(stats["thai_count"], 2)
        self.assertEqual(stats["lao_count"], 1)
        self.assertEqual(stats["valid_count"], 2)

    def test_export_csv(self):
        save_recognition({"plate_text": "70-1234", "country": "Thai", "province": "เชียงใหม่", "is_valid": True})
        csv_text = export_history_csv()
        self.assertIn("License Plate", csv_text)
        self.assertIn("70-1234", csv_text)
        self.assertIn("เชียงใหม่", csv_text)


if __name__ == "__main__":
    unittest.main()

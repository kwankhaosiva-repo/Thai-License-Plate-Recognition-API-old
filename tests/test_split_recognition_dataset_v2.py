"""
tests/test_split_recognition_dataset_v2.py

Unit tests for the leak-free v2 splitter. These catch the two failure modes
that were measured in the datasets currently on disk:

  1. a source image contributing to more than one split (leakage), and
  2. classes left with zero training crops after the split.
"""

import random
import unittest
from collections import Counter, defaultdict

from src.split_recognition_dataset_v2 import (
    group_by_class_and_key,
    stratified_assign,
)


def make_manifest(n_plates: int, chars_per_plate: int = 6, n_classes: int = 50,
                  seed: int = 7):
    """Synthetic manifest shaped like the real one: one plate -> N char crops
    whose classes come from a small per-plate alphabet."""
    rng = random.Random(seed)
    alphabet = [f"c{i}" for i in range(n_classes)]
    rows = []
    for p in range(n_plates):
        key = f"plate{p:05d}_0001_plate_jpg"
        plate_chars = [rng.choice(alphabet) for _ in range(chars_per_plate)]
        for i, ch in enumerate(plate_chars):
            rows.append({
                "task": "char",
                "key": key,
                "label": ch,
                "rel_path": f"char_crops_v2/all/{ch}/{key}_b{i}.jpg",
            })
    return rows


class TestLeakFreeSplit(unittest.TestCase):
    def test_no_source_image_spans_both_splits(self):
        rows = make_manifest(500)
        by_class = group_by_class_and_key(rows)
        assignment, _ = stratified_assign(by_class, 0.2, 42)

        # every key must be assigned exactly once
        self.assertEqual(len(assignment), len({r["key"] for r in rows}))
        self.assertTrue(set(assignment.values()) <= {"train", "valid"})

        # a key is a single split value, so the leakage check is structural —
        # assert it explicitly anyway to catch a future regression
        per_key = defaultdict(set)
        for (cls, key) in ((c, k) for c, ks in by_class.items() for k in ks):
            per_key[key].add(assignment[key])
        self.assertTrue(all(len(v) == 1 for v in per_key.values()))

    def test_valid_ratio_is_close_to_target(self):
        rows = make_manifest(500)
        by_class = group_by_class_and_key(rows)
        assignment, stats = stratified_assign(by_class, 0.2, 42)
        total = sum(s["crops"] for s in stats.values())
        valid = sum(s["valid_crops"] for s in stats.values())
        self.assertGreater(valid / total, 0.10)
        self.assertLess(valid / total, 0.30)

    def test_every_class_keeps_training_crops(self):
        rows = make_manifest(500)
        by_class = group_by_class_and_key(rows)
        _, stats = stratified_assign(by_class, 0.2, 42)
        empty = [c for c, s in stats.items() if s["train_crops"] == 0]
        self.assertEqual(empty, [], f"classes lost all training crops: {empty}")

    def test_deterministic_for_a_fixed_seed(self):
        rows = make_manifest(200)
        by_class = group_by_class_and_key(rows)
        a1, _ = stratified_assign(by_class, 0.2, 123)
        a2, _ = stratified_assign(by_class, 0.2, 123)
        self.assertEqual(a1, a2)

    def test_single_plate_class_stays_in_train(self):
        """A class backed by one source image cannot be validated — it must stay
        in train rather than silently emptying the training set."""
        rows = [
            {"task": "char", "key": "only_plate", "label": "rare",
             "rel_path": "char_crops_v2/all/rare/only_plate_b0.jpg"},
            {"task": "char", "key": "common1", "label": "common",
             "rel_path": "char_crops_v2/all/common/common1_b0.jpg"},
            {"task": "char", "key": "common2", "label": "common",
             "rel_path": "char_crops_v2/all/common/common2_b0.jpg"},
        ]
        by_class = group_by_class_and_key(rows)
        assignment, stats = stratified_assign(by_class, 0.5, 1)
        self.assertEqual(assignment["only_plate"], "train")
        self.assertGreater(stats["rare"]["train_crops"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

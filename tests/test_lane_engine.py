import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lane_engine import (
    Point2D,
    RoadCalibration,
    GroundContact,
    estimate_vehicle_ground_contact,
    assess_lane_straddling,
    TemporalLaneTracker,
    interpolate_x_on_line,
    canonicalize_road_quad,
)


class TestLaneEngine(unittest.TestCase):

    def setUp(self):
        # Synthetic camera view of a road lane
        # Ground plane quad: [TL, TR, BR, BL]
        self.ground_quad = [
            Point2D(x=300.0, y=200.0),  # TL (far left)
            Point2D(x=500.0, y=200.0),  # TR (far right)
            Point2D(x=700.0, y=600.0),  # BR (near right)
            Point2D(x=100.0, y=600.0),  # BL (near left)
        ]
        # Lanes: aligned with road edges
        self.lanes = {
            "left": [Point2D(x=300.0, y=200.0), Point2D(x=100.0, y=600.0)],
            "right": [Point2D(x=500.0, y=200.0), Point2D(x=700.0, y=600.0)],
        }
        self.calib = RoadCalibration(
            ground_quad=self.ground_quad,
            lanes=self.lanes,
            bev_width=400,
            bev_height=800,
            world_width_m=3.5,
            world_length_m=15.0,
        )

    def test_canonicalize_road_quad_all_permutations(self):
        import itertools
        p_fl = Point2D(x=300.0, y=200.0)
        p_fr = Point2D(x=500.0, y=200.0)
        p_nr = Point2D(x=700.0, y=600.0)
        p_nl = Point2D(x=100.0, y=600.0)
        expected = [p_fl, p_fr, p_nr, p_nl]

        for perm in itertools.permutations([p_fl, p_fr, p_nr, p_nl]):
            canonical = canonicalize_road_quad(list(perm))
            self.assertEqual(canonical, expected)

    def test_homography_forward_and_backward_projection(self):
        # 1. Forward projection of corners must hit BEV corners
        bev_corners = self.calib.project_points_to_bev(self.ground_quad)
        self.assertAlmostEqual(bev_corners[0].x, 0.0, delta=1.0)
        self.assertAlmostEqual(bev_corners[0].y, 0.0, delta=1.0)
        self.assertAlmostEqual(bev_corners[1].x, 399.0, delta=1.0)
        self.assertAlmostEqual(bev_corners[1].y, 0.0, delta=1.0)
        self.assertAlmostEqual(bev_corners[2].x, 399.0, delta=1.0)
        self.assertAlmostEqual(bev_corners[2].y, 799.0, delta=1.0)
        self.assertAlmostEqual(bev_corners[3].x, 0.0, delta=1.0)
        self.assertAlmostEqual(bev_corners[3].y, 799.0, delta=1.0)

        # 2. Backward projection of BEV corners must return original quad
        unprojected = self.calib.project_points_from_bev(bev_corners)
        for orig, unproj in zip(self.ground_quad, unprojected):
            self.assertAlmostEqual(orig.x, unproj.x, delta=0.5)
            self.assertAlmostEqual(orig.y, unproj.y, delta=0.5)

    def test_interpolate_x_on_line(self):
        p1 = Point2D(x=100.0, y=0.0)
        p2 = Point2D(x=300.0, y=1000.0)
        # Midpoint at y=500 should be x=200
        x_mid = interpolate_x_on_line(p1, p2, 500.0)
        self.assertIsNotNone(x_mid)
        self.assertAlmostEqual(x_mid, 200.0, delta=0.1)

    def test_vehicle_in_lane_assessment(self):
        # Vehicle centered at near road:
        # Near lane is from x=100 to x=700 at y=600
        # Put vehicle bbox from x=300 to x=500 at y=600
        bbox = [300.0, 450.0, 500.0, 600.0]
        contact = estimate_vehicle_ground_contact(bbox, self.calib)
        assessment = assess_lane_straddling(contact, self.calib)

        self.assertFalse(assessment.is_straddling)
        self.assertEqual(assessment.status_label, "IN_LANE")
        self.assertGreater(assessment.left_wheel_dist_to_lane_cm, 0.0)
        self.assertGreater(assessment.right_wheel_dist_to_lane_cm, 0.0)

    def test_vehicle_straddling_left(self):
        # Near left lane boundary is at x=100 at y=600
        # Place vehicle overlapping left lane: bbox xmin=50, xmax=250 at y=600
        bbox = [50.0, 450.0, 250.0, 600.0]
        contact = estimate_vehicle_ground_contact(bbox, self.calib)
        assessment = assess_lane_straddling(contact, self.calib)

        self.assertTrue(assessment.is_straddling)
        self.assertEqual(assessment.status_label, "STRADDLE_LEFT")
        self.assertEqual(assessment.violation_side, "left")
        self.assertGreater(assessment.straddle_depth_cm, 0.0)

    def test_vehicle_straddling_right(self):
        # Near right lane boundary is at x=700 at y=600
        # Place vehicle overlapping right lane: bbox xmin=600, xmax=780 at y=600
        bbox = [600.0, 450.0, 780.0, 600.0]
        contact = estimate_vehicle_ground_contact(bbox, self.calib)
        assessment = assess_lane_straddling(contact, self.calib)

        self.assertTrue(assessment.is_straddling)
        self.assertEqual(assessment.status_label, "STRADDLE_RIGHT")
        self.assertEqual(assessment.violation_side, "right")
        self.assertGreater(assessment.straddle_depth_cm, 0.0)

    def test_temporal_tracker_persistence(self):
        tracker = TemporalLaneTracker(min_straddle_frames=3)
        from src.lane_engine import VehicleLaneResult, StraddleAssessment

        def make_res(is_straddle: bool):
            contact = estimate_vehicle_ground_contact([50.0, 450.0, 250.0, 600.0], self.calib)
            assessment = StraddleAssessment(
                is_straddling=is_straddle,
                status_label="STRADDLE_LEFT" if is_straddle else "IN_LANE",
                violation_side="left" if is_straddle else None,
                left_wheel_dist_to_lane_cm=-15.0 if is_straddle else 30.0,
                right_wheel_dist_to_lane_cm=50.0,
                straddle_depth_cm=15.0 if is_straddle else 0.0,
                details="test",
            )
            return VehicleLaneResult(
                bbox=[50.0, 450.0, 250.0, 600.0],
                confidence=0.85,
                class_id=7,
                class_name="truck",
                ground_contact=contact,
                assessment=assessment,
            )

        # Frame 1: Straddle -> streak 1 (not persisted)
        res1 = tracker.update([make_res(True)])
        self.assertFalse(res1[0][2])

        # Frame 2: Straddle -> streak 2 (not persisted)
        res2 = tracker.update([make_res(True)])
        self.assertFalse(res2[0][2])

        # Frame 3: Straddle -> streak 3 (>= 3 -> PERSISTED!)
        res3 = tracker.update([make_res(True)])
        self.assertTrue(res3[0][2])


if __name__ == "__main__":
    unittest.main()

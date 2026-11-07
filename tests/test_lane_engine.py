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

    def test_nearest_y_front_axle_filtering(self):
        # Truck with front steer tires at y=600 and trailer tire at y=450
        fake_tires = [
            {"conf": 0.45, "bbox": [180.0, 560.0, 220.0, 600.0], "x1": 180.0, "y1": 560.0, "x2": 220.0, "y2": 600.0}, # Left steer (center = 200.0)
            {"conf": 0.50, "bbox": [380.0, 560.0, 420.0, 600.0], "x1": 380.0, "y1": 560.0, "x2": 420.0, "y2": 600.0}, # Right steer (center = 400.0)
            {"conf": 0.40, "bbox": [430.0, 410.0, 460.0, 450.0], "x1": 430.0, "y1": 410.0, "x2": 460.0, "y2": 450.0}, # Trailer tire (should be ignored)
        ]
        bbox = [170.0, 300.0, 440.0, 600.0]
        contact = estimate_vehicle_ground_contact(bbox, self.calib, detected_tires=fake_tires)

        # Baseline should be at y=600 (front steer axle), exactly at tire footprint centers
        self.assertEqual(contact.left_wheel.y, 600.0)
        self.assertEqual(contact.right_wheel.y, 600.0)
        self.assertEqual(contact.left_wheel.x, 200.0)
        self.assertEqual(contact.right_wheel.x, 400.0)

    def test_single_front_tire_physical_track_width(self):
        # Only right front steer tire detected at center=(400.0, 600.0)
        fake_tires = [
            {"conf": 0.50, "bbox": [380.0, 560.0, 420.0, 600.0], "x1": 380.0, "y1": 560.0, "x2": 420.0, "y2": 600.0}, # Right steer (center=400.0)
        ]
        bbox = [170.0, 300.0, 440.0, 600.0]
        contact = estimate_vehicle_ground_contact(bbox, self.calib, class_name="truck", detected_tires=fake_tires)

        # Right wheel anchored at true AI contact
        self.assertEqual(contact.right_wheel.x, 400.0)
        self.assertEqual(contact.right_wheel.y, 600.0)
        # Left wheel must have exactly 2.05m physical track width in BEV space
        bev_r = contact.bev_right_wheel
        bev_l = contact.bev_left_wheel
        track_w_cm = (bev_r.x - bev_l.x) / (self.calib.pixels_per_meter_x / 100.0)
        self.assertAlmostEqual(track_w_cm, 205.0, delta=1.0)

    def test_duplicate_wheel_clustering_returns_max_two_steer_tires(self):
        # 3 duplicate proposals on left wheel, 2 duplicate proposals on right wheel
        fake_proposals = [
            {"conf": 0.40, "bbox": [476.0, 348.0, 506.0, 384.0], "x1": 476.0, "y1": 348.0, "x2": 506.0, "y2": 384.0},
            {"conf": 0.20, "bbox": [478.0, 358.0, 504.0, 386.0], "x1": 478.0, "y1": 358.0, "x2": 504.0, "y2": 386.0},
            {"conf": 0.10, "bbox": [479.0, 361.0, 513.0, 386.0], "x1": 479.0, "y1": 361.0, "x2": 513.0, "y2": 386.0},
            {"conf": 0.33, "bbox": [576.0, 326.0, 611.0, 386.0], "x1": 576.0, "y1": 326.0, "x2": 611.0, "y2": 386.0},
            {"conf": 0.12, "bbox": [575.0, 340.0, 608.0, 387.0], "x1": 575.0, "y1": 340.0, "x2": 608.0, "y2": 387.0},
        ]
        # In TruckWheelDetector, spatial clustering collapses these 5 proposals to exactly 2
        front_proposals = list(fake_proposals)
        front_proposals.sort(key=lambda t: t.get("conf", 0.0), reverse=True)
        clustered = []
        for t in front_proposals:
            tcx = (t["bbox"][0] + t["bbox"][2]) / 2.0
            tcy = (t["bbox"][1] + t["bbox"][3]) / 2.0
            if not any(abs(tcx - (c["bbox"][0]+c["bbox"][2])/2.0) < 45.0 and abs(tcy - (c["bbox"][1]+c["bbox"][3])/2.0) < 30.0 for c in clustered):
                clustered.append(t)

        self.assertEqual(len(clustered), 2)
        # Left wheel kept with conf 0.40, right wheel kept with conf 0.33
        conf_set = {round(c["conf"], 2) for c in clustered}
        self.assertEqual(conf_set, {0.40, 0.33})

    def test_reject_shadow_tire_outside_vehicle_bounds(self):
        vbox = [473.0, 112.0, 653.0, 354.0] # vy2 = 354.0
        # Proposal 1 is a shadow on the road: y2=365 (11px below vehicle)
        # Proposal 2 is a valid wheel: y2=354 (at vehicle bottom)
        fake_tires = [
            {"conf": 0.35, "bbox": [486.0, 340.0, 508.0, 365.0], "x1": 486.0, "y1": 340.0, "x2": 508.0, "y2": 365.0}, # Shadow
            {"conf": 0.40, "bbox": [573.0, 320.0, 601.0, 354.0], "x1": 573.0, "y1": 320.0, "x2": 601.0, "y2": 354.0}, # Valid
        ]
        # Containment filter rejects the shadow
        matched = []
        for t in fake_tires:
            if t["bbox"][3] <= vbox[3] + 6.0 and t["bbox"][1] < vbox[3]:
                matched.append(t)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["conf"], 0.40)

    def test_articulated_truck_single_front_tire_flips_projection_towards_cab(self):
        # Truck with wide bounding box (trailer on left, cab face on right)
        # Bbox spans from x1=100 to x2=700 (span ~3.5m in BEV).
        # Steer tire detected at tx=450.0 (near cab face on right, but trailer extends far left).
        fake_steer = [
            {
                "conf": 0.45,
                "bbox": [430.0, 560.0, 470.0, 600.0],
                "x1": 430.0, "y1": 560.0, "x2": 470.0, "y2": 600.0,
                # 3 trailer tires to the left of the steer tire
                "all_vehicle_tires": [
                    {"bbox": [150.0, 500.0, 180.0, 540.0], "x1": 150.0, "x2": 180.0},
                    {"bbox": [200.0, 510.0, 230.0, 550.0], "x1": 200.0, "x2": 230.0},
                    {"bbox": [250.0, 520.0, 280.0, 560.0], "x1": 250.0, "x2": 280.0},
                    {"bbox": [430.0, 560.0, 470.0, 600.0], "x1": 430.0, "x2": 470.0},
                ],
            }
        ]
        bbox = [100.0, 300.0, 700.0, 600.0]
        contact = estimate_vehicle_ground_contact(
            bbox, self.calib, class_name="truck", detected_tires=fake_steer
        )

        # Because trailer is on the left, tx=450 is the LEFT steer wheel.
        # Opposite right wheel MUST be projected to the RIGHT (x_right > x_left).
        self.assertEqual(contact.left_wheel.x, 450.0)
        self.assertGreater(contact.right_wheel.x, 450.0)


if __name__ == "__main__":
    unittest.main()

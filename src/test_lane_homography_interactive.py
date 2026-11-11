#!/usr/bin/env python3
"""
Interactive Lane Straddling / Departure Detection Test Tool via Homography (BEV).
Supports:
  - Step-by-step image evaluation across folders
  - Interactive mouse calibration: Click 4 ground points + 2 lane lines
  - Real-world Bird's Eye View (BEV) perspective rectification & distance measurement (cm)
  - Free License Detectors: LibreDFINE Nano/Small (MIT / Apache 2.0)
  - Future-proof: Video file / RTSP stream mode with temporal persistence tracking

Usage:
  python src/test_lane_homography_interactive.py "datasets/lane_test_samples"
  python src/test_lane_homography_interactive.py "datasets/lane_test_samples/truck_day.png"
  python src/test_lane_homography_interactive.py "path/to/video.mp4"

Controls:
  [SPACE] / [ENTER]: Next image / Resume video
  [B]              : Previous image
  [C]              : Enter Interactive Calibration Mode (draw ground plane & lanes)
  [R]              : Reset calibration in progress
  [S]              : Save snapshot of current dashboard
  [M]              : Toggle detector model (LibreDFINE Nano <-> Small)
  [Q] / [ESC]      : Quit
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.lane_engine import (
    Point2D,
    RoadCalibration,
    GroundContact,
    StraddleAssessment,
    VehicleLaneResult,
    FreeLicenseVehicleDetector,
    TemporalLaneTracker,
    estimate_vehicle_ground_contact,
    assess_lane_straddling,
    canonicalize_road_quad,
    interpolate_x_on_line,
)


class InteractiveLaneTester:
    def __init__(
        self,
        weights_dir: Path,
        calib_path: Optional[Path] = None,
        model_size: str = "nano",
        conf_thresh: float = 0.35,
        device: str = "cpu",
        target_nearest: bool = True,
    ):
        self.weights_dir = weights_dir
        self.calib_path = calib_path or Path("lane_calibration.json")
        self.device = device
        self.conf_thresh = conf_thresh
        self.model_size = model_size.lower()
        self.target_nearest = target_nearest

        self.detector = self._load_detector(self.model_size)
        self.tracker = TemporalLaneTracker(min_straddle_frames=5)
        self.calib: Optional[RoadCalibration] = self._load_calibration()
        self.current_img_calib_path: Optional[Path] = None

        # Stage 2: Free-License AI Wheel / Front Cab Detector (v6 / MIT License)
        wheel_weights = self.weights_dir / "front_wheel_detector_v6.pt"
        if not wheel_weights.exists():
            wheel_weights = self.weights_dir / "truck_wheel_detector_mit.pt"
        if wheel_weights.exists():
            try:
                from src.lane_engine import TruckWheelDetector
                self.tire_detector = TruckWheelDetector(wheel_weights, conf_threshold=0.10, device=self.device)
                print(f"🛞 Loaded Stage 2 Wheel & Cab Detector: {wheel_weights.name}")
            except Exception as e:
                print(f"⚠️ Could not load wheel detector: {e}")
                self.tire_detector = None
        else:
            self.tire_detector = None

        # Viewport scale tracking for exact mouse coordinate mapping
        self.header_height = 50
        self.last_cam_w = 1
        self.last_cam_h = 1
        self.last_cam_scale_x = 1.0
        self.last_cam_scale_y = 1.0
        self.mouse_cursor_orig: Optional[Tuple[float, float]] = None

        # Calibration state machine for GUI (Unified 4-Point Corridor Calibration)
        self.calib_mode = False
        self.just_calibrated = False
        self.temp_points: List[Point2D] = []
        self.dragging_corner_idx: Optional[int] = None
        self.mouse_cursor_pos: Optional[Tuple[int, int]] = None

        # Display Mode: 0=ALL BBoxes, 1=FINAL ONLY (2 Wheel Dots + Lanes), 2=Cab Face + Wheels
        self.display_mode: int = 1

    def _load_detector(self, size: str) -> FreeLicenseVehicleDetector:
        target_name = "LibreDFINEn.pt" if size == "nano" else "LibreDFINEs.pt"
        target_path = self.weights_dir / target_name
        if not target_path.exists():
            target_path = self.weights_dir / "LibreDFINEn.pt"
        print(f"📦 Loading Free-License Detector: {target_path.name} on {self.device}...")
        return FreeLicenseVehicleDetector(target_path, conf_threshold=self.conf_thresh, device=self.device)

    def toggle_detector(self) -> str:
        self.model_size = "small" if self.model_size == "nano" else "nano"
        self.detector = self._load_detector(self.model_size)
        return self.model_size

    def _load_calibration(self, resolution: Optional[Tuple[int, int]] = None) -> Optional[RoadCalibration]:
        if self.calib_path and self.calib_path.exists():
            try:
                calib = RoadCalibration.load_json(self.calib_path, resolution=resolution)
                print(f"📐 Loaded lane calibration ({calib.ref_width}x{calib.ref_height}) from: {self.calib_path.name}")
                return calib
            except Exception as e:
                print(f"⚠️ Could not load calibration {self.calib_path}: {e}")
        return None

    def load_calibration_for_image(self, img_path_str: str) -> None:
        # Auto-switch calibration profile based on image resolution (e.g. Cam01 1280x720 vs Cam02 640x480)
        img = cv2.imread(img_path_str)
        if img is None:
            return
        h, w = img.shape[:2]
        self.last_orig_w = w
        self.last_orig_h = h

        target_key = f"{w}x{h}"
        if self.calib is not None and f"{self.calib.ref_width}x{self.calib.ref_height}" == target_key:
            return

        if self.calib_path and self.calib_path.exists():
            self.calib = self._load_calibration(resolution=(w, h))
        if self.calib is None:
            self.apply_camera_default_calibration()

    def save_current_calibration(self) -> None:
        if self.calib and self.calib_path:
            profile_key = f"{self.calib.ref_width}x{self.calib.ref_height}"
            self.calib.save_json(self.calib_path, profile_key=profile_key)
            print(f"💾 Saved lane calibration profile [{profile_key}] to: {self.calib_path.name}")

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        self.mouse_cursor_pos = (x, y)
        header_h = getattr(self, "header_height", 50)
        cam_w = getattr(self, "last_cam_w", 1)
        cam_h = getattr(self, "last_cam_h", 1)
        scale_x = getattr(self, "last_cam_scale_x", 1.0)
        scale_y = getattr(self, "last_cam_scale_y", 1.0)

        # Map window (x, y) to original camera image coordinates
        if header_h <= y < header_h + cam_h and 0 <= x < cam_w:
            orig_x = float(x) * scale_x
            orig_y = float(y - header_h) * scale_y
            self.mouse_cursor_orig = (orig_x, orig_y)
        else:
            self.mouse_cursor_orig = None

        if not self.calib_mode or self.mouse_cursor_orig is None:
            return

        cur_pt = Point2D(x=self.mouse_cursor_orig[0], y=self.mouse_cursor_orig[1])

        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if clicking near an existing corner to drag
                if self.calib and self.calib.ground_quad and len(self.calib.ground_quad) == 4 and len(self.temp_points) == 0:
                    for idx, pt in enumerate(self.calib.ground_quad):
                        dx = pt.x - cur_pt.x
                        dy = pt.y - cur_pt.y
                        if (dx * dx + dy * dy) ** 0.5 < 35.0:
                            self.dragging_corner_idx = idx
                            print(f"  🖐️ Dragging Corner #{idx+1} ({pt.x:.1f}, {pt.y:.1f})")
                            return

                # Otherwise, record point for 4-point corridor setup
                if len(self.temp_points) < 4:
                    self.temp_points.append(cur_pt)
                    print(f"  [Corner {len(self.temp_points)}/4] Set at ({cur_pt.x:.1f}, {cur_pt.y:.1f})")
                    if len(self.temp_points) == 4:
                        self._finalize_4pt_calibration(self.temp_points)

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging_corner_idx is not None and self.calib:
                    new_quad = list(self.calib.ground_quad)
                    new_quad[self.dragging_corner_idx] = cur_pt
                    # Update live without canonicalizing during drag to prevent corner swap
                    self.calib.ground_quad = new_quad
                    self.calib.lanes = {
                        "left": [new_quad[0], new_quad[3]],
                        "right": [new_quad[1], new_quad[2]],
                    }
                    self.calib.compute_homography()
                    self.just_calibrated = True

            elif event == cv2.EVENT_LBUTTONUP:
                if self.dragging_corner_idx is not None:
                    print(f"  ✅ Corner #{self.dragging_corner_idx + 1} adjusted to ({cur_pt.x:.1f}, {cur_pt.y:.1f})")
                    self.dragging_corner_idx = None
                    if self.calib and len(self.calib.ground_quad) == 4:
                        # Auto-square immediately so Far-pair and Near-pair stay perfectly horizontal
                        self.auto_square_calibration()

        except Exception as e:
            print(f"❌ Error during calibration mouse interaction: {e}")
            import traceback
            traceback.print_exc()

    def start_calibration(self) -> None:
        self.calib_mode = True
        self.just_calibrated = False
        self.temp_points = []
        self.dragging_corner_idx = None
        w = getattr(self, "last_orig_w", 1280)
        h = getattr(self, "last_orig_h", 720)
        if self.calib is not None:
            self.calib = self.calib.rescale_to_image(w, h)
            import copy
            self.calib_backup = copy.deepcopy(self.calib)
        else:
            self.calib_backup = None
        print(f"\n🔧 ENTERING LANE CORRIDOR CALIBRATION ({w}x{h}):")
        print("  - Drag any of the 4 yellow corners to align with road lane lines")
        print("  - Press [S] to auto-square / level depth (removes axle slant & shear distortion)")
        print("  - Press [ENTER] or [C] to save & exit")
        print("  - Press [ESC] to cancel changes (revert)")
        print("  - Press [R] to redraw 4 fresh corners")
        print("  - Press [D] for camera golden defaults (35m corridor)\n")

    def auto_square_calibration(self) -> None:
        """Auto-align transverse depths (G1-G2 far, G4-G3 near) to eliminate homography shear."""
        if not self.calib or not self.calib.ground_quad or len(self.calib.ground_quad) != 4:
            print("⚠️ Calibration corridor has not been configured yet.")
            return

        q = self.calib.ground_quad
        # Level far depth baseline (G1 Far-Left, G2 Far-Right)
        # Level near depth baseline (G4 Near-Left, G3 Near-Right)
        y_far = float((q[0].y + q[1].y) / 2.0)
        y_near = float((q[3].y + q[2].y) / 2.0)

        new_quad = [
            Point2D(x=q[0].x, y=y_far),
            Point2D(x=q[1].x, y=y_far),
            Point2D(x=q[2].x, y=y_near),
            Point2D(x=q[3].x, y=y_near),
        ]
        self.calib.ground_quad = new_quad
        self.calib.lanes = {
            "left": [new_quad[0], new_quad[3]],
            "right": [new_quad[1], new_quad[2]],
        }
        self.calib.compute_homography()
        self.just_calibrated = True
        print(f"📐 Auto-squared lane corridor: Far Y leveled at {y_far:.1f}, Near Y leveled at {y_near:.1f}. Perspective shear removed!")

    def finish_calibration(self) -> None:
        if self.calib_mode:
            # Auto-square / level depths automatically before saving
            self.auto_square_calibration()
            self.calib_mode = False
            self.temp_points = []
            self.dragging_corner_idx = None
            self.save_current_calibration()
            self.just_calibrated = True
            print("💾 Calibration auto-squared & saved! Exited calibration mode.")

    def cancel_calibration(self) -> None:
        if self.calib_mode:
            self.calib_mode = False
            self.temp_points = []
            self.dragging_corner_idx = None
            if hasattr(self, "calib_backup") and self.calib_backup is not None:
                self.calib = self.calib_backup
            self.just_calibrated = True
            print("↩️ Calibration cancelled (changes reverted).")

    def apply_camera_default_calibration(self) -> None:
        ref_w = getattr(self, "last_orig_w", 1280)
        ref_h = getattr(self, "last_orig_h", 720)
        if ref_w == 640 and ref_h == 480:
            quad = [
                Point2D(0.0, 77.0),
                Point2D(42.0, 65.0),
                Point2D(424.0, 395.0),
                Point2D(306.0, 480.0),
            ]
            world_len = 25.0
        else:
            quad = [
                Point2D(674.0, 61.0),
                Point2D(715.0, 79.0),
                Point2D(471.0, 709.0),
                Point2D(175.0, 711.0),
            ]
            world_len = 35.0

        self.calib = RoadCalibration(
            ground_quad=quad,
            lanes={
                "left": [quad[0], quad[3]],
                "right": [quad[1], quad[2]],
            },
            bev_width=400,
            bev_height=800,
            ref_width=ref_w,
            ref_height=ref_h,
            world_width_m=3.5,
            world_length_m=world_len,
        )
        self.save_current_calibration()
        self.calib_mode = False
        self.temp_points = []
        self.just_calibrated = True
        print(f"📐 Restored golden default calibration for Camera ({ref_w}x{ref_h})")

    def reset_calibration_in_progress(self) -> None:
        if self.calib_mode:
            self.temp_points = []
            self.dragging_corner_idx = None
            print("🔄 Cleared points. Click 4 corners (Far-Left -> Far-Right -> Near-Right -> Near-Left).")
        else:
            self.calib = None
            if self.calib_path and self.calib_path.exists():
                self.calib_path.unlink(missing_ok=True)
            print("🗑️ Reset existing calibration. Press [C] to calibrate or [D] for camera defaults.")

    def _finalize_4pt_calibration(self, quad_pts: List[Point2D]) -> None:
        canon_quad = canonicalize_road_quad(quad_pts[:4])
        ref_w = getattr(self, "last_orig_w", 1280)
        ref_h = getattr(self, "last_orig_h", 720)
        world_len = 25.0 if ref_w == 640 and ref_h == 480 else 35.0
        self.calib = RoadCalibration(
            ground_quad=canon_quad,
            lanes={
                "left": [canon_quad[0], canon_quad[3]],
                "right": [canon_quad[1], canon_quad[2]],
            },
            bev_width=400,
            bev_height=800,
            ref_width=ref_w,
            ref_height=ref_h,
            world_width_m=3.5,
            world_length_m=world_len,
        )
        self.save_current_calibration()
        self.calib_mode = False
        self.temp_points = []
        self.dragging_corner_idx = None
        self.just_calibrated = True
        print(f"✅ Road & Lane Calibration saved! (Profile: {ref_w}x{ref_h})\n")
        self.just_calibrated = True
        print("✅ Road & Lane Calibration saved! (Separated Ground Surface Quad + Lane Lines)\n")

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[List[VehicleLaneResult], float]:
        t0 = time.perf_counter()
        raw_detections = self.detector.detect_vehicles(frame_bgr)
        results = []
        h, w = frame_bgr.shape[:2]
        active_calib = self.calib.rescale_to_image(w, h) if self.calib else None

        # Stage 2: Full-frame AI wheel detection (native letterboxed resolution)
        all_frame_tires = []
        if getattr(self, "tire_detector", None) is not None:
            try:
                all_frame_tires = self.tire_detector.detect_all_tires(frame_bgr, conf=0.05)
            except Exception as e:
                all_frame_tires = []

        for det in raw_detections:
            bbox = det["bbox"]
            conf = det["conf"]
            cls_id = det["cls_id"]
            name = det["name"]

            # Filter tires belonging to this vehicle and tag nearest-Y front steer axle
            detected_tires = []
            if getattr(self, "tire_detector", None) is not None:
                try:
                    detected_tires = self.tire_detector.detect_tires_on_vehicle(
                        frame_bgr, bbox, all_tires=all_frame_tires
                    )
                except Exception as e:
                    detected_tires = []

            if active_calib is not None:
                contact = estimate_vehicle_ground_contact(
                    bbox, active_calib, class_name=name, detected_tires=detected_tires
                )
                assessment = assess_lane_straddling(contact, active_calib, tolerance_cm=5.0)
            else:
                p_dummy = Point2D(x=float((bbox[0] + bbox[2]) / 2), y=float(bbox[3]))
                contact = GroundContact(
                    left_wheel=p_dummy, right_wheel=p_dummy, center_bottom=p_dummy,
                    bev_left_wheel=p_dummy, bev_right_wheel=p_dummy, bev_center_bottom=p_dummy,
                )
                assessment = StraddleAssessment(
                    is_straddling=False, status_label="NO_CALIB", violation_side=None,
                    left_wheel_dist_to_lane_cm=0.0, right_wheel_dist_to_lane_cm=0.0,
                    straddle_depth_cm=0.0, details="Press [C] to calibrate road ground & lanes"
                )

            results.append(VehicleLaneResult(
                bbox=bbox,
                confidence=conf,
                class_id=cls_id,
                class_name=name,
                ground_contact=contact,
                assessment=assessment,
                detected_tires=detected_tires,
            ))

        if getattr(self, "target_nearest", True) and len(results) > 1:
            results.sort(key=lambda r: r.bbox[3], reverse=True)
            results = results[:1]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, latency_ms

    def render_dashboard(
        self,
        frame_bgr: np.ndarray,
        results: List[VehicleLaneResult],
        latency_ms: float,
        title_info: str = "",
    ) -> np.ndarray:
        cam_view = frame_bgr.copy()
        h, w = cam_view.shape[:2]

        # 1. Draw Calibration or In-Progress Clicks on Camera View
        if self.calib_mode:
            self._draw_calibration_in_progress(cam_view)
        elif self.calib is not None:
            self._draw_active_calibration(cam_view)

        # 2. Draw Vehicle Detections and Ground Contacts
        for res in results:
            self._draw_vehicle_result(cam_view, res)

        # 3. Create or Render Bird's Eye View (BEV) Panel
        bev_panel = self._render_bev_panel(frame_bgr, results)

        # 4. Combine into Side-by-Side Dashboard
        target_cam_h = 720
        target_cam_w = int(w * (target_cam_h / h))
        self.last_cam_scale_x = w / float(target_cam_w)
        self.last_cam_scale_y = h / float(target_cam_h)
        self.last_cam_w = target_cam_w
        self.last_cam_h = target_cam_h
        self.last_orig_w = w
        self.last_orig_h = h
        self.header_height = 50

        cam_resized = cv2.resize(cam_view, (target_cam_w, target_cam_h))
        bev_resized = cv2.resize(bev_panel, (380, target_cam_h))
        combined = np.hstack([cam_resized, bev_resized])

        # 5. Header Banner
        combined = self._draw_header(combined, results, latency_ms, title_info)
        return combined

    def _draw_active_calibration(self, img: np.ndarray) -> None:
        assert self.calib is not None
        h, w = img.shape[:2]
        active_calib = self.calib.rescale_to_image(w, h)

        # Ground Quad (Hidden in Mode 1: FINAL ONLY)
        if getattr(self, "display_mode", 0) != 1:
            quad_pts = np.int32([p.to_tuple() for p in active_calib.ground_quad])
            cv2.polylines(img, [quad_pts], isClosed=True, color=(255, 200, 0), thickness=2)
            for i, pt in enumerate(active_calib.ground_quad):
                cv2.circle(img, pt.to_tuple(), 5, (255, 255, 0), -1)
                cv2.putText(img, f"G{i+1}", (pt.to_tuple()[0] + 6, pt.to_tuple()[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # Lane Boundaries (Shown in all modes!)
        for lane_name, color in [("left", (0, 0, 255)), ("right", (0, 0, 255))]:
            pts = active_calib.lanes.get(lane_name)
            if pts and len(pts) >= 2:
                p1, p2 = pts[0], pts[1]
                y_bot = float(img.shape[0] - 1)
                x_bot = interpolate_x_on_line(p1, p2, y_bot)
                p_end = (int(round(x_bot)), int(round(y_bot))) if x_bot is not None else p2.to_tuple()
                cv2.line(img, p1.to_tuple(), p_end, color, 3)

    def _draw_calibration_in_progress(self, img: np.ndarray) -> None:
        banner_h = 70
        cv2.rectangle(img, (10, 10), (img.shape[1] - 10, banner_h), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (img.shape[1] - 10, banner_h), (0, 255, 255), 2)

        pts_to_draw = self.temp_points if self.temp_points else (self.calib.ground_quad if self.calib else [])
        n_pts = len(self.temp_points)

        if n_pts == 0 and self.calib and len(self.calib.ground_quad) == 4:
            prompt = "CALIBRATION MODE: Drag yellow corners to align with road lane lines"
            hint = "Tip: [ENTER]/[C] save & exit | [S] Auto-Level (removes axle tilt) | [D] golden defaults | [R] redraw"
        else:
            prompt = f"LANE CORRIDOR SETUP: Click 4 Corners [{n_pts}/4]"
            hint = "Click: 1.Far-Left -> 2.Far-Right -> 3.Near-Right -> 4.Near-Left. [S] to Auto-Level depth."

        cv2.putText(img, prompt, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
        cv2.putText(img, hint, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)

        # Draw Level Alignment Guides when 4 corners exist
        if len(pts_to_draw) == 4:
            y_far = int(round((pts_to_draw[0].y + pts_to_draw[1].y) / 2.0))
            y_near = int(round((pts_to_draw[3].y + pts_to_draw[2].y) / 2.0))
            # Faint depth level guides
            cv2.line(img, (0, y_far), (img.shape[1], y_far), (100, 100, 100), 1, cv2.LINE_AA)
            cv2.line(img, (0, y_near), (img.shape[1], y_near), (100, 100, 100), 1, cv2.LINE_AA)

        # Draw Corridor Polygon if 2 or more points exist (prevent crossing into an X)
        if len(pts_to_draw) > 1:
            if len(pts_to_draw) == 4:
                draw_quad = canonicalize_road_quad(pts_to_draw)
                poly_pts = np.int32([p.to_tuple() for p in draw_quad])
                is_closed = True
            else:
                poly_pts = np.int32([p.to_tuple() for p in pts_to_draw])
                is_closed = False
            cv2.polylines(img, [poly_pts], isClosed=is_closed, color=(0, 200, 255), thickness=2)

        # Draw Handles
        corner_names = ["1.Far-Left", "2.Far-Right", "3.Near-Right", "4.Near-Left"]
        for i, pt in enumerate(pts_to_draw):
            p = pt.to_tuple()
            is_hovered = False
            if self.mouse_cursor_orig is not None:
                dx = pt.x - self.mouse_cursor_orig[0]
                dy = pt.y - self.mouse_cursor_orig[1]
                if (dx * dx + dy * dy) ** 0.5 < 35.0:
                    is_hovered = True

            radius = 12 if is_hovered else 8
            circle_color = (0, 255, 255) if is_hovered else (255, 255, 0)
            cv2.circle(img, p, radius, circle_color, -1)
            cv2.circle(img, p, radius + 2, (0, 0, 0), 2)
            lbl = corner_names[i] if i < len(corner_names) else f"P{i+1}"
            cv2.putText(img, lbl, (p[0] + 12, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        # Precision Crosshair & Rubber-band guide line
        if self.mouse_cursor_orig is not None:
            cur_p = (int(round(self.mouse_cursor_orig[0])), int(round(self.mouse_cursor_orig[1])))
            cv2.drawMarker(img, cur_p, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=26, thickness=2)
            cv2.putText(img, f"({cur_p[0]}, {cur_p[1]})", (cur_p[0] + 10, cur_p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            if 0 < len(self.temp_points) < 4:
                last_p = self.temp_points[-1].to_tuple()
                cv2.line(img, last_p, cur_p, (0, 255, 255), 2, cv2.LINE_AA)

    def _draw_vehicle_result(self, img: np.ndarray, res: VehicleLaneResult) -> None:
        bbox = [int(v) for v in res.bbox]
        is_viol = res.assessment.is_straddling
        box_color = (0, 0, 255) if is_viol else (0, 255, 0)

        # Outer vehicle bounding box (Mode 0: ALL only)
        if self.display_mode == 0:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)

        # Draw front cab face box if available (Modes 0 and 2)
        if self.display_mode in (0, 2) and getattr(res.ground_contact, "cab_face", None) is not None:
            cb = [int(v) for v in res.ground_contact.cab_face]
            cv2.rectangle(img, (cb[0], cb[1]), (cb[2], cb[3]), (0, 255, 128), 2)
            cv2.putText(img, "Cab Face (No Mirror)", (cb[0], max(12, cb[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

        # Draw detected AI front steer tire bounding boxes (Mode 0: ALL only)
        if self.display_mode == 0 and getattr(res, "detected_tires", None):
            for t in res.detected_tires:
                if t.get("class_name") == "cab_face":
                    continue
                tb = [int(v) for v in t["bbox"]]
                t_color = (255, 255, 0)
                cv2.rectangle(img, (tb[0], tb[1]), (tb[2], tb[3]), t_color, 2)
                cv2.circle(img, (int((tb[0] + tb[2]) / 2), tb[3]), 4, (0, 255, 255), -1)
                label = f"Front Tire {t['conf']:.2f}"
                cv2.putText(img, label, (tb[0], max(10, tb[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, t_color, 1)

        # Wheel ground contact baseline and 2 Contact Dots (Shown in ALL modes!)
        contact = res.ground_contact
        p_left = contact.left_wheel.to_tuple()
        p_right = contact.right_wheel.to_tuple()

        cv2.line(img, p_left, p_right, (0, 255, 255), 2)
        dot_color_l = (0, 0, 255) if res.assessment.violation_side == "LEFT" else (0, 255, 255)
        dot_color_r = (0, 0, 255) if res.assessment.violation_side == "RIGHT" else (0, 255, 255)
        cv2.circle(img, p_left, 7, dot_color_l, -1)
        cv2.circle(img, p_right, 7, dot_color_r, -1)
        cv2.circle(img, p_left, 8, (0, 0, 0), 1)
        cv2.circle(img, p_right, 8, (0, 0, 0), 1)

        # Status Badge (Hidden in Mode 1: FINAL ONLY)
        if self.display_mode != 1:
            badge_text = f"{res.class_name} {res.confidence:.2f} | {res.assessment.status_label}"
            if is_viol:
                badge_text += f" ({res.assessment.straddle_depth_cm:.1f}cm)"

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            badge_y = bbox[1] if self.display_mode == 0 else min(p_left[1], p_right[1]) - 16
            by1 = max(0, badge_y - th - 8)
            bx1 = bbox[0] if self.display_mode == 0 else min(p_left[0], p_right[0])
            cv2.rectangle(img, (bx1, by1), (bx1 + tw + 10, by1 + th + 8), box_color, -1)
            cv2.putText(img, badge_text, (bx1 + 5, by1 + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255) if is_viol else (0, 0, 0), 2)

    def _render_bev_panel(self, frame_bgr: np.ndarray, results: List[VehicleLaneResult]) -> np.ndarray:
        bev_w, bev_h = 380, 720
        panel = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
        panel[:] = (24, 26, 30)

        if self.calib is None:
            cv2.putText(panel, "NO CALIBRATION", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(panel, "Press [C] to Calibrate", (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            return panel

        h, w = frame_bgr.shape[:2]
        active_calib = self.calib.rescale_to_image(w, h)

        margin_x = 35
        lane_px_w = bev_w - 2 * margin_x
        sx = lane_px_w / float(active_calib.bev_width)
        sy = bev_h / float(active_calib.bev_height)

        # 1. Clean Asphalt Lane Surface (Dark Asphalt)
        cv2.rectangle(panel, (margin_x, 0), (bev_w - margin_x, bev_h), (34, 37, 42), -1)

        # 2. Left Painted Dashed Lane Line (Upright & Straight)
        y = 0
        dash_len, gap_len = 24, 24
        while y < bev_h:
            cv2.line(panel, (margin_x, y), (margin_x, min(bev_h, y + dash_len)), (220, 220, 220), 3)
            y += dash_len + gap_len

        # 3. Right Painted Solid Lane Line (Upright & Straight)
        cv2.line(panel, (bev_w - margin_x, 0), (bev_w - margin_x, bev_h), (240, 240, 240), 4)

        # 4. Longitudinal Distance Grid Lines (Every 5 meters along corridor)
        for m in range(5, int(active_calib.world_length_m) + 1, 5):
            y_px = int((m / active_calib.world_length_m) * bev_h)
            cv2.line(panel, (margin_x - 10, y_px), (bev_w - margin_x + 10, y_px), (65, 70, 78), 1)
            cv2.putText(panel, f"{m}m", (8, y_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 135, 142), 1)

        # 5. Top-down Vehicles footprint on BEV Map
        approaching_vehicles = []
        for res in results:
            contact = res.ground_contact
            p_left = (int(margin_x + contact.bev_left_wheel.x * sx), int(contact.bev_left_wheel.y * sy))
            p_right = (int(margin_x + contact.bev_right_wheel.x * sx), int(contact.bev_right_wheel.y * sy))
            color = (0, 0, 255) if res.assessment.is_straddling else (0, 255, 0)

            if -20 <= p_left[1] <= bev_h + 30:
                # Top-down vehicle footprint rectangle in BEV
                v_len_m = 9.5 if res.class_name.lower() in ("truck", "bus", "train") else 4.5
                v_len_px = int(v_len_m * (bev_h / float(active_calib.world_length_m)))
                bx1 = min(p_left[0], p_right[0])
                bx2 = max(p_left[0], p_right[0])
                by_front = p_left[1]
                by_rear = max(0, by_front - v_len_px)

                # Semi-transparent vehicle footprint overlay
                footprint_overlay = panel.copy()
                cv2.rectangle(footprint_overlay, (bx1, by_rear), (bx2, by_front), color, -1)
                cv2.addWeighted(footprint_overlay, 0.35, panel, 0.65, 0, panel)
                cv2.rectangle(panel, (bx1, by_rear), (bx2, by_front), color, 2)

                # Wheel axle and contact dots in BEV
                cv2.line(panel, p_left, p_right, (0, 255, 255), 3)
                cv2.circle(panel, p_left, 7, color, -1)
                cv2.circle(panel, p_right, 7, color, -1)

                # Measurement labels
                lbl_y = max(45, min(bev_h - 15, p_left[1] - 12))
                lbl = f"L: {res.assessment.left_wheel_dist_to_lane_cm:+.0f}cm | R: {res.assessment.right_wheel_dist_to_lane_cm:+.0f}cm"
                cv2.putText(panel, lbl, (max(10, min(bev_w - 230, bx1 - 20)), lbl_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            elif p_left[1] < -20:
                approaching_vehicles.append(res)

        if approaching_vehicles:
            first_v = approaching_vehicles[0]
            dist_m = abs(first_v.ground_contact.bev_left_wheel.y) / float(active_calib.pixels_per_meter_y)
            app_text = f"^ {first_v.class_name.upper()} APPROACHING ({dist_m:.0f}m)"
            cv2.putText(panel, app_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        # BEV Panel Title
        cv2.rectangle(panel, (0, 0), (bev_w, 40), (16, 18, 20), -1)
        cv2.putText(panel, "TOP-DOWN BIRD EYE VIEW (BEV)", (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        return panel

    def _draw_header(self, img: np.ndarray, results: List[VehicleLaneResult], latency_ms: float, title_info: str) -> np.ndarray:
        header_h = 50
        header = np.zeros((header_h, img.shape[1], 3), dtype=np.uint8)
        header[:] = (20, 20, 20)

        # Left: Frame & Model info
        left_str = f"{title_info} | LibreDFINE-{self.model_size.upper()} ({latency_ms:.0f}ms)"
        cv2.putText(header, left_str, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        # Center/Dynamic: Confidence, View Mode & Target Badge
        target_mode_str = "NEAREST [T]" if getattr(self, "target_nearest", True) else "ALL [T]"
        tire_ai_str = "Tire-AI: ON" if getattr(self, "tire_detector", None) is not None else "Tire-AI: OFF"
        disp_mode_names = ["ALL [V]", "FINAL ONLY [V]", "CAB+WHEEL [V]"]
        disp_str = disp_mode_names[getattr(self, "display_mode", 0)]
        conf_str = f"Conf: {self.conf_thresh:.2f} [+/-] | {disp_str} | {target_mode_str}"
        (lw, _), _ = cv2.getTextSize(left_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        conf_x = max(lw + 20, img.shape[1] // 2 - 180)
        cv2.putText(header, conf_str, (conf_x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        # Right: Violations count
        viol_count = sum(1 for r in results if r.assessment.is_straddling)
        v_color = (0, 0, 255) if viol_count > 0 else (0, 255, 0)
        v_text = f"VIOLATIONS: {viol_count} VEHICLE(S)" if viol_count > 0 else "ALL IN LANE [CLEAR]"
        cv2.putText(header, v_text, (img.shape[1] - 380, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, v_color, 2)

        return np.vstack([header, img])


def collect_image_paths(input_path_str: str) -> List[str]:
    p = Path(input_path_str)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [str(f) for f in p.iterdir() if f.suffix.lower() in exts]
        files.sort()
        return files
    return []


def is_video_file(path_str: str) -> bool:
    ext = Path(path_str).suffix.lower()
    return ext in {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".m4v"}


def main():
    parser = argparse.ArgumentParser(description="Interactive Lane Straddling Detection via Homography (BEV)")
    parser.add_argument("input_path", type=str, nargs="?", default=None,
                        help="Folder of images, single image, or video file to test")
    parser.add_argument("--input", "-i", dest="input_flag", type=str, default=None,
                        help="Folder of images, single image, or video file to test (alias for input_path)")
    parser.add_argument("--model", type=str, choices=["nano", "small"], default="nano",
                        help="Model size: 'nano' (fastest) or 'small' (higher accuracy)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for vehicle detection (default: 0.35)")
    import torch
    default_device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    parser.add_argument("--device", type=str, default=default_device, help="Device to run on ('cpu' or 'mps')")
    parser.add_argument("--calib", type=str, default="lane_calibration.json", help="Path to lane calibration JSON")
    parser.add_argument("--save-all", action="store_true", help="Non-interactive batch mode: process and save snapshots")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to save output snapshots")
    args = parser.parse_args()

    weights_dir = PROJECT_ROOT / "weights"
    calib_file = Path(args.calib) if Path(args.calib).is_absolute() else (PROJECT_ROOT / args.calib)
    output_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "scratch" / "lane_test_snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    tester = InteractiveLaneTester(
        weights_dir=weights_dir,
        calib_path=calib_file,
        model_size=args.model,
        conf_thresh=args.conf,
        device=args.device,
    )

    input_str = args.input_flag or args.input_path or "datasets/lane_test_samples"
    # Auto-repair paths broken by unquoted spaces in terminal arguments
    if args.input_flag and args.input_path:
        joined_path = f"{args.input_flag} {args.input_path}"
        if Path(joined_path).exists():
            input_str = joined_path

    # --- VIDEO MODE ---
    if is_video_file(input_str):
        print(f"🎬 Video Stream Mode: {input_str}")
        cap = cv2.VideoCapture(input_str)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {input_str}")
            sys.exit(1)

        win_name = "Lane Straddling Detection — Video Stream (Homography BEV)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, tester.on_mouse)

        frame_idx = 0
        is_paused = False

        while cap.isOpened():
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream.")
                    break
                frame_idx += 1
                results, lat_ms = tester.process_frame(frame)
                persisted_tuples = tester.tracker.update(results)
                # Overwrite straddle assessment with temporal persistence
                for track_id, r, persisted in persisted_tuples:
                    if persisted:
                        r.assessment.details += f" [TRACK #{track_id} CONFIRMED]"

            dash = tester.render_dashboard(frame, results, lat_ms, f"Frame #{frame_idx}")
            cv2.imshow(win_name, dash)

            key = cv2.waitKey(1 if not is_paused else 0) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord(" "):
                is_paused = not is_paused
            elif key in (ord("c"), ord("C")):
                tester.start_calibration()
            elif key in (ord("r"), ord("R")):
                tester.reset_calibration_in_progress()
            elif key in (ord("s"), ord("S")):
                out_p = output_dir / f"video_snap_frame_{frame_idx:05d}.png"
                cv2.imwrite(str(out_p), dash)
                print(f"📸 Saved snapshot: {out_p.name}")

        cap.release()
        cv2.destroyAllWindows()
        return

    # --- IMAGE BATCH / INTERACTIVE FOLDER MODE ---
    image_paths = collect_image_paths(input_str)
    if not image_paths:
        print(f"❌ No test images found at: {input_str}")
        sys.exit(1)

    total = len(image_paths)
    print(f"📂 Found {total} image(s) to test at: {input_str}")

    # Non-interactive batch mode
    if args.save_all:
        print(f"Running batch snapshot export to: {output_dir}")
        for idx, img_p in enumerate(image_paths):
            tester.load_calibration_for_image(img_p)
            bgr = cv2.imread(img_p)
            if bgr is None:
                continue
            results, lat_ms = tester.process_frame(bgr)
            dash = tester.render_dashboard(bgr, results, lat_ms, f"[{idx+1}/{total}] {Path(img_p).name}")
            out_p = output_dir / f"lane_snap_{Path(img_p).stem}.png"
            cv2.imwrite(str(out_p), dash)
            print(f"  📸 Saved [{idx+1}/{total}]: {out_p.name} ({lat_ms:.1f}ms)")
        print("\nAll snapshots saved successfully!")
        return

    # Interactive GUI Window Mode
    win_name = "Lane Straddling Detection — Interactive Test Tool (Homography BEV)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1400, 800)
    cv2.setMouseCallback(win_name, tester.on_mouse)

    cur_idx = 0
    print("\n🎮 Interactive Controls:")
    print("  [SPACE] / [ENTER]: Next image")
    print("  [B]              : Previous image")
    print("  [C]              : Calibrate Lane (Click any 4 road corners in ANY order)")
    print("  [D]              : Apply Camera 01 Default Golden Calibration")
    print("  [R]              : Reset calibration")
    print("  [+] / [-]        : Increase / Decrease Confidence Threshold (0.05 steps)")
    print("  [V]              : Toggle Display Mode (ALL BBoxes -> FINAL ONLY (2 Wheel Dots + Lanes) -> Cab Face)")
    print("  [T]              : Toggle Vehicle Target (NEAREST ONLY <-> ALL VEHICLES)")
    print("  [M]              : Toggle Model (Nano <-> Small)")
    print("  [S]              : Save snapshot")
    print("  [Q] / [ESC]      : Quit\n")

    while 0 <= cur_idx < total:
        img_p = image_paths[cur_idx]
        tester.load_calibration_for_image(img_p)
        bgr = cv2.imread(img_p)
        if bgr is None:
            cur_idx += 1
            continue

        results, lat_ms = tester.process_frame(bgr)
        need_reprocess = False

        viol_count = sum(1 for r in results if r.assessment.is_straddling)
        print(f"\n[{cur_idx+1}/{total}] {Path(img_p).name} | Detections: {len(results)} | Violations: {viol_count} | Latency: {lat_ms:.1f}ms")
        for i, r in enumerate(results):
            print(f"   🚗 #{i+1} {r.class_name} ({r.confidence:.2f}): {r.assessment.status_label} - {r.assessment.details}")

        # Responsive single event loop for this frame
        while True:
            # If calibration was just finalized by mouse clicks or parameters changed, re-process immediately!
            if tester.just_calibrated or need_reprocess:
                tester.just_calibrated = False
                need_reprocess = False
                results, lat_ms = tester.process_frame(bgr)
                viol_count = sum(1 for r in results if r.assessment.is_straddling)
                print(f"🔄 Updated frame evaluation: {len(results)} vehicles | Violations: {viol_count} | Latency: {lat_ms:.1f}ms")
                for i, r in enumerate(results):
                    print(f"   🚗 #{i+1} {r.class_name} ({r.confidence:.2f}): {r.assessment.status_label} - {r.assessment.details}")

            dash = tester.render_dashboard(bgr, results, lat_ms, f"[{cur_idx+1}/{total}] {Path(img_p).name}")
            cv2.imshow(win_name, dash)

            # 25ms tick ensures fluid UI rendering and instant reaction to mouse clicks
            key = cv2.waitKey(25) & 0xFF

            if key == 255:  # No key pressed
                continue

            # Special handling while in calibration mode
            if tester.calib_mode:
                if key in (13, ord("\r"), ord("\n"), ord("c"), ord("C")):
                    tester.finish_calibration()
                    need_reprocess = True
                elif key == 27:
                    tester.cancel_calibration()
                    need_reprocess = True
                elif key in (ord("r"), ord("R")):
                    tester.reset_calibration_in_progress()
                    need_reprocess = True
                elif key in (ord("d"), ord("D")):
                    tester.apply_camera_default_calibration()
                    need_reprocess = True
                elif key in (ord("s"), ord("S")):
                    tester.auto_square_calibration()
                    need_reprocess = True
                continue

            # Standard Navigation Mode key handling
            if key in (ord("q"), ord("Q"), 27):
                cur_idx = -1  # Signal exit program
                break
            elif key in (ord(" "), ord("\r"), ord("\n"), 13):
                cur_idx += 1
                break
            elif key in (ord("b"), ord("B")):
                cur_idx = max(0, cur_idx - 1)
                break
            elif key in (ord("c"), ord("C")):
                tester.start_calibration()
            elif key in (ord("d"), ord("D")):
                tester.apply_camera_default_calibration()
                need_reprocess = True
            elif key in (ord("r"), ord("R")):
                tester.reset_calibration_in_progress()
                need_reprocess = True
            elif key in (ord("v"), ord("V")):
                tester.display_mode = (tester.display_mode + 1) % 3
                mode_names = [
                    "ALL BBoxes (Full Vehicle + Cab Face + Tires)",
                    "FINAL ONLY (2 Wheel Dots + Baseline + Lanes)",
                    "Cab Face + 2 Wheel Dots + Lanes",
                ]
                print(f"👁️ Switched Display Mode: [{tester.display_mode}] {mode_names[tester.display_mode]}")
                need_reprocess = True
            elif key in (ord("t"), ord("T")):
                tester.target_nearest = not getattr(tester, "target_nearest", True)
                need_reprocess = True
                mode_str = "NEAREST ONLY" if tester.target_nearest else "ALL VEHICLES"
                print(f"🎯 Toggled vehicle target mode: {mode_str}")
            elif key in (ord("+"), ord("=")):
                tester.conf_thresh = min(0.95, round(tester.conf_thresh + 0.05, 2))
                tester.detector.conf_threshold = tester.conf_thresh
                need_reprocess = True
                print(f"🔼 Adjusted Conf Threshold: {tester.conf_thresh:.2f}")
            elif key in (ord("-"), ord("_")):
                tester.conf_thresh = max(0.05, round(tester.conf_thresh - 0.05, 2))
                tester.detector.conf_threshold = tester.conf_thresh
                need_reprocess = True
                print(f"🔽 Adjusted Conf Threshold: {tester.conf_thresh:.2f}")
            elif key in (ord("m"), ord("M")):
                new_m = tester.toggle_detector()
                need_reprocess = True
                print(f"🔄 Switched detector to: LibreDFINE-{new_m.upper()}")
            elif key in (ord("s"), ord("S")):
                out_p = output_dir / f"lane_snap_{Path(img_p).stem}.png"
                cv2.imwrite(str(out_p), dash)
                print(f"📸 Saved snapshot to: {out_p}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

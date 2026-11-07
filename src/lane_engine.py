from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import cv2
import numpy as np
import torch


@dataclass
class Point2D:
    x: float
    y: float

    def to_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    def to_list(self) -> List[float]:
        return [float(self.x), float(self.y)]

    @classmethod
    def from_tuple(cls, pt: Tuple[float, float]) -> "Point2D":
        return cls(x=float(pt[0]), y=float(pt[1]))


def canonicalize_road_quad(points: List[Point2D]) -> List[Point2D]:
    """
    Takes any 4 points clicked on the road surface in camera view (in ANY user order)
    and canonicalizes them to:
      [0] Far-Left   (Upper-left, furthest down the road)
      [1] Far-Right  (Upper-right, furthest down the road)
      [2] Near-Right (Lower-right, closest to camera)
      [3] Near-Left  (Lower-left, closest to camera)
    This completely eliminates user order dependency and guarantees the BEV
    is always upright, unrotated, and aligned with traffic flow.
    """
    if len(points) != 4:
        return points
    pts_sorted_y = sorted(points, key=lambda p: p.y)
    far_pair = sorted(pts_sorted_y[:2], key=lambda p: p.x)
    near_pair = sorted(pts_sorted_y[2:], key=lambda p: p.x)

    p_far_left = far_pair[0]
    p_far_right = far_pair[1]
    p_near_left = near_pair[0]
    p_near_right = near_pair[1]

    return [p_far_left, p_far_right, p_near_right, p_near_left]


@dataclass
class RoadCalibration:
    ground_quad: List[Point2D]
    lanes: Optional[Dict[str, List[Point2D]]] = None
    bev_width: int = 400
    bev_height: int = 800
    ref_width: int = 1280
    ref_height: int = 720
    world_width_m: float = 3.5
    world_length_m: float = 35.0
    homography_matrix: Optional[np.ndarray] = None
    inv_homography_matrix: Optional[np.ndarray] = None

    def __post_init__(self):
        if len(self.ground_quad) == 4:
            # Canonicalize 4 points so order clicked never matters
            self.ground_quad = canonicalize_road_quad(self.ground_quad)
            if not self.lanes or "left" not in self.lanes or "right" not in self.lanes:
                self.lanes = {
                    "left": [self.ground_quad[0], self.ground_quad[3]],
                    "right": [self.ground_quad[1], self.ground_quad[2]],
                }
            if self.homography_matrix is None:
                self.compute_homography()

    def rescale_to_image(self, img_w: int, img_h: int) -> "RoadCalibration":
        """
        Dynamically rescales the calibrated road corridor to match the resolution
        of the target image (e.g. 1280x720, 640x480, 1920x1080), guaranteeing
        seamless calibration sharing across any resolution without off-screen handles.
        """
        if img_w == self.ref_width and img_h == self.ref_height:
            return self

        sx = float(img_w) / float(self.ref_width)
        sy = float(img_h) / float(self.ref_height)

        scaled_quad = [Point2D(p.x * sx, p.y * sy) for p in self.ground_quad]
        scaled_lanes = {
            k: [Point2D(p.x * sx, p.y * sy) for p in pts]
            for k, pts in self.lanes.items()
        } if self.lanes else None

        return RoadCalibration(
            ground_quad=scaled_quad,
            lanes=scaled_lanes,
            bev_width=self.bev_width,
            bev_height=self.bev_height,
            ref_width=img_w,
            ref_height=img_h,
            world_width_m=self.world_width_m,
            world_length_m=self.world_length_m,
        )

    def compute_homography(self) -> None:
        if len(self.ground_quad) != 4:
            raise ValueError("ground_quad must have exactly 4 points: [TL, TR, BR, BL]")
        src_pts = np.float32([p.to_list() for p in self.ground_quad])
        dst_pts = np.float32([
            [0, 0],
            [self.bev_width - 1, 0],
            [self.bev_width - 1, self.bev_height - 1],
            [0, self.bev_height - 1]
        ])
        H, _ = cv2.findHomography(src_pts, dst_pts)
        if H is None:
            raise ValueError("Failed to compute valid homography matrix from given quad points")
        inv_H = np.linalg.inv(H)
        self.homography_matrix = H
        self.inv_homography_matrix = inv_H

    @property
    def pixels_per_meter_x(self) -> float:
        return float(self.bev_width) / float(self.world_width_m)

    @property
    def pixels_per_meter_y(self) -> float:
        return float(self.bev_height) / float(self.world_length_m)

    def warp_image_to_bev(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.homography_matrix is None:
            self.compute_homography()
        return cv2.warpPerspective(img_bgr, self.homography_matrix, (self.bev_width, self.bev_height))

    def project_points_to_bev(self, points: List[Point2D]) -> List[Point2D]:
        if self.homography_matrix is None:
            self.compute_homography()
        if not points:
            return []
        pts_arr = np.array([p.to_list() for p in points], dtype=np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts_arr, self.homography_matrix).reshape(-1, 2)
        return [Point2D(x=float(p[0]), y=float(p[1])) for p in warped_pts]

    def project_points_from_bev(self, points: List[Point2D]) -> List[Point2D]:
        if self.inv_homography_matrix is None:
            self.compute_homography()
        if not points:
            return []
        pts_arr = np.array([p.to_list() for p in points], dtype=np.float32).reshape(-1, 1, 2)
        unwarped_pts = cv2.perspectiveTransform(pts_arr, self.inv_homography_matrix).reshape(-1, 2)
        return [Point2D(x=float(p[0]), y=float(p[1])) for p in unwarped_pts]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ground_quad": [p.to_list() for p in self.ground_quad],
            "lanes": {k: [p.to_list() for p in v] for k, v in self.lanes.items()},
            "bev_width": self.bev_width,
            "bev_height": self.bev_height,
            "ref_width": self.ref_width,
            "ref_height": self.ref_height,
            "world_width_m": self.world_width_m,
            "world_length_m": self.world_length_m,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoadCalibration":
        ground_quad = [Point2D.from_tuple(p) for p in data["ground_quad"]]
        lanes = {k: [Point2D.from_tuple(p) for p in v] for k, v in data.get("lanes", {}).items()}
        return cls(
            ground_quad=ground_quad,
            lanes=lanes,
            bev_width=data.get("bev_width", 400),
            bev_height=data.get("bev_height", 800),
            ref_width=data.get("ref_width", 1280),
            ref_height=data.get("ref_height", 720),
            world_width_m=data.get("world_width_m", 3.5),
            world_length_m=data.get("world_length_m", 35.0),
        )

    def save_json(self, file_path: Union[str, Path], profile_key: Optional[str] = None) -> None:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        key = profile_key or f"{self.ref_width}x{self.ref_height}"
        existing_data = {}
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = {}

        if "profiles" in existing_data and isinstance(existing_data["profiles"], dict):
            existing_data["profiles"][key] = self.to_dict()
            if "default_profile" not in existing_data:
                existing_data["default_profile"] = key
            output_data = existing_data
        else:
            # Create multi-profile format preserving default
            output_data = {
                "default_profile": key,
                "profiles": {
                    key: self.to_dict()
                }
            }
            # If previous file had a single flat calibration, preserve it if different key
            if "ground_quad" in existing_data and "profiles" not in existing_data:
                old_key = f"{existing_data.get('ref_width', 1280)}x{existing_data.get('ref_height', 720)}"
                if old_key != key:
                    output_data["profiles"][old_key] = existing_data

        with open(p, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

    @classmethod
    def load_json(cls, file_path: Union[str, Path], resolution: Optional[Tuple[int, int]] = None) -> "RoadCalibration":
        p = Path(file_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "profiles" in data and isinstance(data["profiles"], dict):
            profiles = data["profiles"]
            if resolution is not None:
                target_key = f"{resolution[0]}x{resolution[1]}"
                if target_key in profiles:
                    return cls.from_dict(profiles[target_key])

            # Fall back to default_profile or first profile
            default_key = data.get("default_profile")
            if default_key and default_key in profiles:
                return cls.from_dict(profiles[default_key])
            if profiles:
                first_key = next(iter(profiles))
                return cls.from_dict(profiles[first_key])

        return cls.from_dict(data)


@dataclass
class GroundContact:
    left_wheel: Point2D
    right_wheel: Point2D
    center_bottom: Point2D
    bev_left_wheel: Point2D
    bev_right_wheel: Point2D
    bev_center_bottom: Point2D
    trailer_tires: Optional[List[Dict[str, Any]]] = None
    cab_face: Optional[List[float]] = None


@dataclass
class StraddleAssessment:
    is_straddling: bool
    status_label: str
    violation_side: Optional[str]
    left_wheel_dist_to_lane_cm: float
    right_wheel_dist_to_lane_cm: float
    straddle_depth_cm: float
    details: str


@dataclass
class VehicleLaneResult:
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str
    ground_contact: GroundContact
    assessment: StraddleAssessment
    detected_tires: Optional[List[Dict[str, Any]]] = None


def interpolate_x_on_line(p1: Point2D, p2: Point2D, target_y: float) -> Optional[float]:
    dy = p2.y - p1.y
    if abs(dy) < 5.0:
        return None
    t = (target_y - p1.y) / dy
    t = max(-3.0, min(3.0, t))
    return p1.x + t * (p2.x - p1.x)


def estimate_vehicle_ground_contact(
    bbox: List[float],
    calib: RoadCalibration,
    class_name: str = "truck",
    wheel_inset_ratio: float = 0.09,
    detected_tires: Optional[List[Dict[str, Any]]] = None,
) -> GroundContact:
    """
    Estimates the left and right front wheel ground contact points based on physical
    vehicle dimensions and calibrated lane width, with 2-Stage AI Front Wheel Detection.
    Strictly focuses on the front steer axle (nearest Y to camera) using true tire footprint
    centers and standard physical track width, preventing baseline overshoot.
    """
    x1, y1, x2, y2 = bbox

    # Standard physical front axle track width in meters
    cls_lower = class_name.lower()
    if cls_lower in ("truck", "bus", "train"):
        base_track_w_m = 2.05
    elif cls_lower in ("car", "van", "suv"):
        base_track_w_m = 1.65
    else:
        base_track_w_m = 0.90  # motorcycle / small

    cab_face_bbox: Optional[List[float]] = None
    if detected_tires:
        for t in detected_tires:
            if t.get("class_name") == "cab_face":
                cab_face_bbox = t.get("bbox")
                break
            elif "cab_bbox" in t:
                cab_face_bbox = t.get("cab_bbox")
                break

    # Physical track width constraint based on BEV cab width
    track_w_m = base_track_w_m
    if cab_face_bbox is not None:
        cab_bottom_y = float(cab_face_bbox[3])
        bev_cab_l = calib.project_points_to_bev([Point2D(x=float(cab_face_bbox[0]), y=cab_bottom_y)])[0]
        bev_cab_r = calib.project_points_to_bev([Point2D(x=float(cab_face_bbox[2]), y=cab_bottom_y)])[0]
        cab_w_bev = abs(bev_cab_r.x - bev_cab_l.x)
        cab_w_m = cab_w_bev / calib.pixels_per_meter_x

        if cab_w_m < 1.95:
            # Compact truck, pickup truck with canopy, or van (even if classified as 'truck')
            # Real physical track width is ~0.88 of cab width
            track_w_m = min(base_track_w_m, max(1.40, cab_w_m * 0.88))
        else:
            # Full commercial heavy truck / bus
            track_w_m = min(base_track_w_m, cab_w_m * 0.90)

    track_w_bev = track_w_m * calib.pixels_per_meter_x
    v_cx = (x1 + x2) / 2.0

    p_left: Optional[Point2D] = None
    p_right: Optional[Point2D] = None

    if detected_tires:
        # Filter for front-most steer axle tires (nearest Y to camera, ignoring cab_face boxes)
        wheel_candidates = [
            t for t in detected_tires
            if t.get("class_name") != "cab_face" and t.get("is_front_steer", True)
        ]
        if wheel_candidates:
            max_ty = max(t.get("y2", t["bbox"][3]) for t in wheel_candidates)
            steer_tires = [t for t in wheel_candidates if t.get("y2", t["bbox"][3]) >= max_ty - 25.0]
        else:
            steer_tires = []

        if len(steer_tires) >= 2:
            # Sort horizontally: first is left front wheel, second is right front wheel
            sorted_tires = sorted(steer_tires, key=lambda t: (t.get("x1", t["bbox"][0]) + t.get("x2", t["bbox"][2])) / 2.0)
            best_l = sorted_tires[0]
            best_r = sorted_tires[-1]
            avg_y = (best_l.get("y2", best_l["bbox"][3]) + best_r.get("y2", best_r["bbox"][3])) / 2.0
            p_left = Point2D(x=float((best_l.get("x1", best_l["bbox"][0]) + best_l.get("x2", best_l["bbox"][2])) / 2.0), y=float(avg_y))
            p_right = Point2D(x=float((best_r.get("x1", best_r["bbox"][0]) + best_r.get("x2", best_r["bbox"][2])) / 2.0), y=float(avg_y))
        elif len(steer_tires) == 1:
            best_t = steer_tires[0]
            tx = float((best_t.get("x1", best_t["bbox"][0]) + best_t.get("x2", best_t["bbox"][2])) / 2.0)
            ty = float(best_t.get("y2", best_t["bbox"][3]))

            if cab_face_bbox is not None:
                # With verified front cab face (no trailer / side mirror overshoot):
                # Any tire to the right of cab center is definitively the RIGHT front wheel.
                # Any tire to the left of cab center is definitively the LEFT front wheel.
                cab_cx = (cab_face_bbox[0] + cab_face_bbox[2]) / 2.0
                anchor_is_left = (tx < cab_cx)
            else:
                # Project bounding box horizontal extents at tire road-contact level to BEV
                bev_left_edge = calib.project_points_to_bev([Point2D(x=x1, y=ty)])[0]
                bev_right_edge = calib.project_points_to_bev([Point2D(x=x2, y=ty)])[0]
                bev_tire = calib.project_points_to_bev([Point2D(x=tx, y=ty)])[0]

                span_left_m = abs(bev_tire.x - bev_left_edge.x) / calib.pixels_per_meter_x
                span_right_m = abs(bev_right_edge.x - bev_tire.x) / calib.pixels_per_meter_x
                total_span_m = abs(bev_right_edge.x - bev_left_edge.x) / calib.pixels_per_meter_x

                # Check if other tires on the vehicle indicate trailer side
                vehicle_all_tires = best_t.get("all_vehicle_tires") or detected_tires or []
                num_tires_left = sum(
                    1 for t in vehicle_all_tires
                    if ((t.get("x1", t["bbox"][0]) + t.get("x2", t["bbox"][2])) / 2.0) < tx - 15.0
                )
                num_tires_right = sum(
                    1 for t in vehicle_all_tires
                    if ((t.get("x1", t["bbox"][0]) + t.get("x2", t["bbox"][2])) / 2.0) > tx + 15.0
                )

                is_articulated = (total_span_m > 2.60) and (cls_lower in ("truck", "bus", "train"))

                if is_articulated:
                    # On an articulated / trailer truck, one side is the long trailer (with all side wheels)
                    # and the other side is the front cab face (~2.0 - 2.5m wide).
                    # The steer wheel visible on the side facing the camera is on the trailer side of the cab:
                    # - If trailer is on the left: steer wheel is the LEFT wheel, project RIGHT (+track_w_bev towards x2)
                    # - If trailer is on the right: steer wheel is the RIGHT wheel, project LEFT (-track_w_bev towards x1)
                    if num_tires_left > num_tires_right or (num_tires_left == num_tires_right and span_left_m > span_right_m):
                        anchor_is_left = True
                    else:
                        anchor_is_left = False
                else:
                    # Symmetric / standard vehicle: anchor based on horizontal proximity to bbox center
                    anchor_is_left = (tx - x1) <= (x2 - tx)

            if anchor_is_left:
                # Anchor is left front wheel -> project right front wheel to the RIGHT (+track_w_bev)
                p_left = Point2D(x=tx, y=ty)
                bev_l = calib.project_points_to_bev([p_left])[0]
                bev_r = Point2D(x=bev_l.x + track_w_bev, y=bev_l.y)
                proj = calib.project_points_from_bev([bev_r])[0]
                if cab_face_bbox is not None:
                    ref_y = float(cab_face_bbox[3])
                    proj_y = float(np.clip(proj.y, ref_y - 8.0, ref_y + 8.0))
                    proj_x = float(min(proj.x, cab_face_bbox[2] + 10.0))
                    p_right = Point2D(x=proj_x, y=proj_y)
                else:
                    p_right = proj
            else:
                # Anchor is right front wheel -> project left front wheel to the LEFT (-track_w_bev)
                p_right = Point2D(x=tx, y=ty)
                bev_r = calib.project_points_to_bev([p_right])[0]
                bev_l = Point2D(x=bev_r.x - track_w_bev, y=bev_r.y)
                proj = calib.project_points_from_bev([bev_l])[0]
                if cab_face_bbox is not None:
                    ref_y = float(cab_face_bbox[3])
                    proj_y = float(np.clip(proj.y, ref_y - 8.0, ref_y + 8.0))
                    proj_x = float(max(proj.x, cab_face_bbox[0] - 10.0))
                    p_left = Point2D(x=proj_x, y=proj_y)
                else:
                    p_left = proj

    # Robust Fallback if no front tires were detected
    bev_left_wheel: Optional[Point2D] = bev_l if (p_left is not None and 'bev_l' in locals()) else None
    bev_right_wheel: Optional[Point2D] = bev_r if (p_right is not None and 'bev_r' in locals()) else None

    if p_left is None or p_right is None:
        if cab_face_bbox is not None:
            # Cab face provides clean ground width without trailer or side mirror overhang
            cab_cx = (cab_face_bbox[0] + cab_face_bbox[2]) / 2.0
            cab_bottom_y = cab_face_bbox[3]
            bev_cab_center = calib.project_points_to_bev([Point2D(x=cab_cx, y=cab_bottom_y)])[0]
            half_w = track_w_bev / 2.0
            bev_l = Point2D(x=bev_cab_center.x - half_w, y=bev_cab_center.y)
            bev_r = Point2D(x=bev_cab_center.x + half_w, y=bev_cab_center.y)
            p_left = calib.project_points_from_bev([bev_l])[0]
            p_right = calib.project_points_from_bev([bev_r])[0]
            bev_left_wheel = bev_l
            bev_right_wheel = bev_r
        else:
            bev_left_edge = calib.project_points_to_bev([Point2D(x=x1, y=y2)])[0]
            bev_right_edge = calib.project_points_to_bev([Point2D(x=x2, y=y2)])[0]
            bev_span_m = abs(bev_right_edge.x - bev_left_edge.x) / calib.pixels_per_meter_x

            if bev_span_m > 2.80 and cls_lower in ("truck", "bus", "train"):
                cab_inset_bev = 0.35 * calib.pixels_per_meter_x
                bev_l = Point2D(x=bev_left_edge.x + cab_inset_bev, y=bev_left_edge.y)
                bev_r = Point2D(x=bev_l.x + track_w_bev, y=bev_l.y)
            else:
                bev_center = calib.project_points_to_bev([Point2D(x=v_cx, y=y2)])[0]
                half_w = track_w_bev / 2.0
                bev_l = Point2D(x=bev_center.x - half_w, y=bev_center.y)
                bev_r = Point2D(x=bev_center.x + half_w, y=bev_center.y)

            p_left = calib.project_points_from_bev([bev_l])[0]
            p_right = calib.project_points_from_bev([bev_r])[0]
            bev_left_wheel = bev_l
            bev_right_wheel = bev_r

    p_center = Point2D(x=(p_left.x + p_right.x) / 2.0, y=(p_left.y + p_right.y) / 2.0)
    if bev_left_wheel is None or bev_right_wheel is None:
        projected = calib.project_points_to_bev([p_left, p_right, p_center])
        bev_left_wheel = projected[0]
        bev_right_wheel = projected[1]
        bev_center_bottom = projected[2]
    else:
        bev_center_bottom = Point2D(x=(bev_left_wheel.x + bev_right_wheel.x) / 2.0,
                                    y=(bev_left_wheel.y + bev_right_wheel.y) / 2.0)

    return GroundContact(
        left_wheel=p_left,
        right_wheel=p_right,
        center_bottom=p_center,
        bev_left_wheel=bev_left_wheel,
        bev_right_wheel=bev_right_wheel,
        bev_center_bottom=bev_center_bottom,
        trailer_tires=None,
        cab_face=cab_face_bbox,
    )


def assess_lane_straddling(
    ground_contact: GroundContact,
    calib: RoadCalibration,
    tolerance_cm: float = 0.0,
    vehicle_class: str = "truck",
    tire_half_w_cm: Optional[float] = None,
) -> StraddleAssessment:
    left_lane_pts = calib.lanes.get("left")
    right_lane_pts = calib.lanes.get("right")
    if not left_lane_pts or not right_lane_pts or len(left_lane_pts) < 2 or len(right_lane_pts) < 2:
        return StraddleAssessment(
            is_straddling=False,
            status_label="NO_LANE_CALIBRATION",
            violation_side=None,
            left_wheel_dist_to_lane_cm=0.0,
            right_wheel_dist_to_lane_cm=0.0,
            straddle_depth_cm=0.0,
            details="Lane boundaries not configured",
        )

    # Determine physical tire contact patch half-width:
    # A commercial truck steer tire has contact width ~28-30cm (half-width = 14cm).
    # A passenger vehicle tire has contact width ~20-22cm (half-width = 10cm).
    if tire_half_w_cm is None:
        cls_lower = (vehicle_class or "truck").lower()
        if cls_lower in ("truck", "bus", "train"):
            tire_half_w_cm = 14.0
        elif cls_lower in ("car", "van", "suv"):
            tire_half_w_cm = 10.0
        else:
            tire_half_w_cm = 5.0

    left_lane_bev = calib.project_points_to_bev(left_lane_pts)
    right_lane_bev = calib.project_points_to_bev(right_lane_pts)

    left_wheel_y = ground_contact.bev_left_wheel.y
    right_wheel_y = ground_contact.bev_right_wheel.y

    left_line_x_at_left_wheel = interpolate_x_on_line(left_lane_bev[0], left_lane_bev[1], left_wheel_y)
    right_line_x_at_right_wheel = interpolate_x_on_line(right_lane_bev[0], right_lane_bev[1], right_wheel_y)

    px_per_cm = calib.pixels_per_meter_x / 100.0

    if left_line_x_at_left_wheel is None:
        left_line_x_at_left_wheel = 0.0
    if right_line_x_at_right_wheel is None:
        right_line_x_at_right_wheel = float(calib.bev_width - 1)

    # Physical distance from tire center to lane line (positive = inside lane corridor)
    diff_left = ground_contact.bev_left_wheel.x - left_line_x_at_left_wheel
    diff_right = right_line_x_at_right_wheel - ground_contact.bev_right_wheel.x
    left_center_dist_cm = diff_left / px_per_cm
    right_center_dist_cm = diff_right / px_per_cm

    # Clearance from outer tire edge to lane line (negative = tire edge penetrates lane line)
    left_clearance_cm = left_center_dist_cm - tire_half_w_cm
    right_clearance_cm = right_center_dist_cm - tire_half_w_cm

    # Check if vehicle is in the active evaluation corridor
    avg_bev_y = (left_wheel_y + right_wheel_y) / 2.0
    if avg_bev_y < -150.0:
        dist_m = abs(avg_bev_y) / calib.pixels_per_meter_y
        return StraddleAssessment(
            is_straddling=False,
            status_label="APPROACHING",
            violation_side=None,
            left_wheel_dist_to_lane_cm=left_clearance_cm,
            right_wheel_dist_to_lane_cm=right_clearance_cm,
            straddle_depth_cm=0.0,
            details=f"Vehicle approaching corridor ({dist_m:.1f}m away)",
        )

    straddle_left = False
    straddle_right = False
    depth_cm = 0.0

    if left_clearance_cm < -tolerance_cm:
        straddle_left = True
        depth_cm = max(depth_cm, abs(left_clearance_cm))

    if right_clearance_cm < -tolerance_cm:
        straddle_right = True
        depth_cm = max(depth_cm, abs(right_clearance_cm))

    depth_cm = min(500.0, depth_cm)

    if straddle_left and straddle_right:
        return StraddleAssessment(
            is_straddling=True,
            status_label="OUT_OF_LANE",
            violation_side="both",
            left_wheel_dist_to_lane_cm=left_clearance_cm,
            right_wheel_dist_to_lane_cm=right_clearance_cm,
            straddle_depth_cm=depth_cm,
            details=f"Crossed both lanes (L: {left_clearance_cm:+.1f}cm, R: {right_clearance_cm:+.1f}cm)",
        )
    elif straddle_left:
        return StraddleAssessment(
            is_straddling=True,
            status_label="STRADDLE_LEFT",
            violation_side="left",
            left_wheel_dist_to_lane_cm=left_clearance_cm,
            right_wheel_dist_to_lane_cm=right_clearance_cm,
            straddle_depth_cm=depth_cm,
            details=f"Left tire crossed lane by {abs(left_clearance_cm):.1f}cm",
        )
    elif straddle_right:
        return StraddleAssessment(
            is_straddling=True,
            status_label="STRADDLE_RIGHT",
            violation_side="right",
            left_wheel_dist_to_lane_cm=left_clearance_cm,
            right_wheel_dist_to_lane_cm=right_clearance_cm,
            straddle_depth_cm=depth_cm,
            details=f"Right tire crossed lane by {abs(right_clearance_cm):.1f}cm",
        )
    else:
        return StraddleAssessment(
            is_straddling=False,
            status_label="IN_LANE",
            violation_side=None,
            left_wheel_dist_to_lane_cm=left_clearance_cm,
            right_wheel_dist_to_lane_cm=right_clearance_cm,
            straddle_depth_cm=0.0,
            details=f"Inside lane (Clearance L: {left_clearance_cm:.1f}cm, R: {right_clearance_cm:.1f}cm)",
        )


class FreeLicenseVehicleDetector:
    """Vehicle detection wrapper using D-FINE Nano/Small via LibreYOLO (MIT / Apache 2.0)."""

    def __init__(self, weights_path: Path, conf_threshold: float = 0.35, device: str = "cpu"):
        from libreyolo import LibreYOLO
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = LibreYOLO(str(weights_path), device=device)
        self.vehicle_classes = [2, 3, 5, 6, 7]  # car, motorcycle, bus, train, truck

    def detect_vehicles(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            image_bgr,
            conf=self.conf_threshold,
            classes=self.vehicle_classes,
        )
        raw_boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                if w < 35 or h < 35 or (w * h) < 1200:
                    continue
                name = self.model.names.get(cls_id, f"vehicle_{cls_id}")
                if cls_id == 6:  # Map 'train' (long articulated semi-truck) to 'truck'
                    name = "truck"
                raw_boxes.append({
                    "bbox": xyxy,
                    "conf": conf,
                    "cls_id": cls_id,
                    "name": name,
                })

        return self._suppress_duplicate_vehicle_boxes(raw_boxes)

    def _suppress_duplicate_vehicle_boxes(
        self,
        detections: List[Dict[str, Any]],
        iou_thresh: float = 0.35,
        iomin_thresh: float = 0.50,
    ) -> List[Dict[str, Any]]:
        if not detections:
            return []
        detections.sort(key=lambda d: d["conf"], reverse=True)
        kept = []
        for det in detections:
            bbox1 = det["bbox"]
            overlap = False
            for k in kept:
                bbox2 = k["bbox"]
                iou, iomin = self._compute_iou_and_iomin(bbox1, bbox2)
                if iou > iou_thresh or iomin > iomin_thresh:
                    overlap = True
                    break
            if not overlap:
                kept.append(det)
        return kept

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        iou, _ = FreeLicenseVehicleDetector._compute_iou_and_iomin(box1, box2)
        return iou

    @staticmethod
    def _compute_iou_and_iomin(box1: List[float], box2: List[float]) -> Tuple[float, float]:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        min_area = min(area1, area2)
        iomin = inter_area / min_area if min_area > 0 else 0.0
        return iou, iomin


class TruckWheelDetector:
    """Stage 2: Real-time Vehicle Wheel and Tire Detector (MIT License)."""

    def __init__(self, weights_path: Union[str, Path], conf_threshold: float = 0.05, device: str = "cpu"):
        from ultralytics import YOLO
        self.device = device
        self.conf_threshold = conf_threshold
        self.weights_path = Path(weights_path)
        self.model = YOLO(str(self.weights_path))

    def detect_all_tires(
        self,
        image_bgr: np.ndarray,
        conf: Optional[float] = None,
        imgsz: int = 640,
    ) -> List[Dict[str, Any]]:
        """
        Runs tire detection on the full camera frame with letterboxing, preserving
        natural road perspective and fine tire edges without blurred sub-crop artifacts.
        """
        th = conf if conf is not None else self.conf_threshold
        results = self.model.predict(image_bgr, conf=th, imgsz=imgsz, verbose=False)
        detected_tires: List[Dict[str, Any]] = []

        for r in results:
            for box in r.boxes:
                cid = int(box.cls[0])
                score = float(box.conf[0])
                name = self.model.names.get(cid, "")
                if name.lower() in ("tire", "wheel", "front_wheel", "cab_face"):
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    gx1, gy1, gx2, gy2 = coords
                    detected_tires.append({
                        "conf": score,
                        "bbox": [gx1, gy1, gx2, gy2],
                        "x1": gx1,
                        "y1": gy1,
                        "x2": gx2,
                        "y2": gy2,
                        "bottom_contact": Point2D((gx1 + gx2) / 2.0, gy2),
                        "class_name": name.lower(),
                    })

        return detected_tires

    def detect_tires_on_vehicle(
        self,
        image_bgr: np.ndarray,
        vehicle_bbox: List[float],
        all_tires: Optional[List[Dict[str, Any]]] = None,
        min_conf: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """
        Associates detected tires with the specified vehicle bounding box and tags
        nearest-Y tires as front steer axle.
        Enforces strict physical vehicle containment, minimum confidence gating,
        and axle co-planarity (Y-level parity), preventing asphalt road shadows from corrupting ground contact.
        """
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vh = vy2 - vy1

        if all_tires is None:
            all_tires = self.detect_all_tires(image_bgr)

        # Check for matching front cab face proposal (eliminates trailer & side mirror errors)
        matching_cab = None
        for t in all_tires:
            if t.get("class_name") == "cab_face" and t.get("conf", 0.0) >= 0.20:
                bx1, by1, bx2, by2 = t["bbox"]
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
                if (vx1 - 30.0 <= cx <= vx2 + 30.0) and (vy1 - 30.0 <= cy <= vy2 + 30.0):
                    if matching_cab is None or t["conf"] > matching_cab["conf"]:
                        matching_cab = t

        ref_x1, ref_y1, ref_x2, ref_y2 = matching_cab["bbox"] if matching_cab else (vx1, vy1, vx2, vy2)

        # Spatial containment filter:
        # 1. Tire must sit inside vehicle/cab horizontally.
        # 2. Tire must touch the road surface near vehicle bottom.
        #    This strictly eliminates side mirrors, windshields, and door handles floating in the air.
        # 3. Minimum confidence gating (rejects noisy shadows / tar marks).
        matched: List[Dict[str, Any]] = []
        for t in all_tires:
            if t.get("class_name") == "cab_face":
                continue
            if t.get("conf", 0.0) < min_conf:
                continue
            bx1, by1, bx2, by2 = t["bbox"]
            cx = (bx1 + bx2) / 2.0
            
            if matching_cab is not None:
                # Confined to front cab face with dynamic scale-aware margins
                cab_w = ref_x2 - ref_x1
                cab_h = ref_y2 - ref_y1
                margin_x = max(35.0, 0.30 * cab_w)
                margin_y = max(45.0, 0.35 * cab_h)
                if (ref_x1 - margin_x <= cx <= min(ref_x2 + margin_x, vx2 + 25.0)) and (by2 >= ref_y2 - margin_y and by2 <= ref_y2 + 25.0):
                    matched.append(t)
            else:
                if (vx1 - 15.0 <= cx <= vx2 + 15.0) and (bx1 >= vx1 - 20.0 and bx2 <= vx2 + 20.0):
                    if (by2 >= vy2 - 50.0 and by2 <= vy2 + 10.0 and by1 < vy2):
                        matched.append(t)

        if not matched:
            if matching_cab is not None:
                return [{
                    "class_name": "cab_face",
                    "bbox": matching_cab["bbox"],
                    "conf": matching_cab["conf"],
                    "cab_bbox": matching_cab["bbox"],
                    "is_front_steer": False,
                }]
            return []

        # Keep strictly front-most tires (nearest Y to camera = steer axle)
        max_y = max(t["y2"] for t in matched)
        front_proposals = [t for t in matched if t["y2"] >= max_y - 20.0]

        # Spatial clustering (NMS) for overlapping proposals on the same physical wheel
        front_proposals.sort(key=lambda t: t.get("conf", 0.0), reverse=True)
        clustered_tires: List[Dict[str, Any]] = []
        for t in front_proposals:
            tcx = (t["bbox"][0] + t["bbox"][2]) / 2.0
            tcy = (t["bbox"][1] + t["bbox"][3]) / 2.0
            is_duplicate = False
            for c in clustered_tires:
                ccx = (c["bbox"][0] + c["bbox"][2]) / 2.0
                ccy = (c["bbox"][1] + c["bbox"][3]) / 2.0
                if abs(tcx - ccx) < 45.0 and abs(tcy - ccy) < 25.0:
                    is_duplicate = True
                    break
            if not is_duplicate:
                clustered_tires.append(t)

        # A vehicle front steer axle physically has at most ONE wheel on the left and ONE on the right
        if len(clustered_tires) >= 2:
            # Sort horizontally: leftmost is front-left wheel, rightmost is front-right wheel
            clustered_tires.sort(key=lambda t: (t["bbox"][0] + t["bbox"][2]) / 2.0)
            t_left = clustered_tires[0]
            t_right = clustered_tires[-1]
            conf_l = t_left.get("conf", 0.0)
            conf_r = t_right.get("conf", 0.0)
            y_diff = abs(t_left["y2"] - t_right["y2"])

            # If one wheel is clearly dominant (LOS visible) while the opposite wheel is low-confidence
            # (occluded / fog lamp / shadow artifact), use ONLY the high-confidence dominant wheel as anchor!
            if conf_l >= 0.40 and conf_r < 0.35:
                front_steer = [t_left]
            elif conf_r >= 0.40 and conf_l < 0.35:
                front_steer = [t_right]
            elif y_diff <= 18.0 and min(conf_l, conf_r) >= 0.30:
                front_steer = [t_left, t_right]
            else:
                # One wheel is skewed or a residual shadow: retain only the single higher-confidence wheel
                front_steer = [max([t_left, t_right], key=lambda t: t.get("conf", 0.0))]
        elif len(clustered_tires) == 1:
            front_steer = [clustered_tires[0]]
        else:
            front_steer = []

        for t in front_steer:
            t["is_front_steer"] = True
            t["all_vehicle_tires"] = matched
            if matching_cab is not None:
                t["cab_bbox"] = matching_cab["bbox"]

        if not front_steer and matching_cab is not None:
            return [{
                "class_name": "cab_face",
                "bbox": matching_cab["bbox"],
                "conf": matching_cab["conf"],
                "cab_bbox": matching_cab["bbox"],
                "is_front_steer": False,
            }]

        return front_steer


class TemporalLaneTracker:
    """Tracks vehicles across consecutive frames and applies temporal persistence filtering."""

    def __init__(self, min_straddle_frames: int = 5, iou_match_thresh: float = 0.35):
        self.min_straddle_frames = min_straddle_frames
        self.iou_match_thresh = iou_match_thresh
        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, vehicle_results: List[VehicleLaneResult]) -> List[Tuple[int, VehicleLaneResult, bool]]:
        matched_results = []
        unmatched_new = list(vehicle_results)
        updated_track_ids = set()

        for track_id, track_info in list(self.active_tracks.items()):
            best_idx = -1
            best_iou = self.iou_match_thresh
            for idx, res in enumerate(unmatched_new):
                iou = FreeLicenseVehicleDetector._compute_iou(track_info["bbox"], res.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0:
                matched_res = unmatched_new.pop(best_idx)
                track_info["bbox"] = matched_res.bbox
                track_info["frames_seen"] += 1
                if matched_res.assessment.is_straddling:
                    track_info["straddle_streak"] += 1
                else:
                    track_info["straddle_streak"] = max(0, track_info["straddle_streak"] - 1)

                is_persisted = track_info["straddle_streak"] >= self.min_straddle_frames
                matched_results.append((track_id, matched_res, is_persisted))
                updated_track_ids.add(track_id)

        for new_res in unmatched_new:
            new_id = self.next_track_id
            self.next_track_id += 1
            streak = 1 if new_res.assessment.is_straddling else 0
            self.active_tracks[new_id] = {
                "bbox": new_res.bbox,
                "frames_seen": 1,
                "straddle_streak": streak,
            }
            is_persisted = streak >= self.min_straddle_frames
            matched_results.append((new_id, new_res, is_persisted))
            updated_track_ids.add(new_id)

        for track_id in list(self.active_tracks.keys()):
            if track_id not in updated_track_ids:
                del self.active_tracks[track_id]

        return matched_results

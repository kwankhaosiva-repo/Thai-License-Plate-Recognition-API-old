"""
src/rtsp_streamer.py

Edge RTSP Stream Ingestor & Frame Dispatcher for Thai & Laos LPR.
Connects to an RTSP camera stream (or local video / webcam), samples keyframes,
and posts them to the LPR Recognition API (Local, ngrok tunnel, or GCP Cloud Run).

Cost-saving architecture:
- Avoids expensive 24/7 "CPU always allocated" Cloud Run pricing by dispatching
  lightweight sampled frames (1-3 FPS) over HTTP requests.
- Cloud Run scales down to zero when idle, keeping costs at virtually $0.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_rtsp_streamer(
    source: str,
    endpoint: str = "http://localhost:8080/api/detect/image",
    target_fps: float = 2.0,
    conf_m1: float = 0.35,
    conf_m2: float = 0.25,
    headless: bool = False,
    motion_gate: bool = True,
):
    """
    Connects to RTSP/Camera source, samples frames at target_fps,
    and dispatches detection requests to the specified endpoint.
    """
    print("=" * 70)
    print("📹 Thai & Laos LPR — RTSP Edge Streamer")
    print(f"   Source Stream  : {source}")
    print(f"   Target Endpoint: {endpoint}")
    print(f"   Sample Rate    : {target_fps} FPS")
    print(f"   Motion Gating  : {'Enabled (skip static scenes)' if motion_gate else 'Disabled'}")
    print(f"   Headless Mode  : {headless}")
    print("=" * 70)

    # Parse numeric camera index if passed (e.g. "0")
    stream_src = int(source) if source.isdigit() else source

    cap = cv2.VideoCapture(stream_src)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video stream source: {source}")
        sys.exit(1)

    # Low latency RTSP buffer settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    min_interval = 1.0 / max(0.2, target_fps)
    last_sent_time = 0.0

    last_plate_text = "Scanning..."
    last_province = ""
    last_country = ""
    last_latency = 0
    last_status = "Ready"

    prev_gray: Optional[np.ndarray] = None
    motion_threshold = 12.0  # Mean absolute difference threshold

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Stream frame dropped or disconnected. Retrying in 2s...")
                time.sleep(2.0)
                cap.open(stream_src)
                continue

            now = time.time()
            elapsed_since_sent = now - last_sent_time

            # Motion detection check to prevent sending identical empty frames
            has_motion = True
            if motion_gate:
                small = cv2.resize(frame, (160, 90))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    mean_diff = float(np.mean(diff))
                    has_motion = mean_diff >= motion_threshold
                prev_gray = gray

            # Dispatch frame when interval satisfied and motion detected
            if elapsed_since_sent >= min_interval and has_motion:
                last_sent_time = now

                # Encode frame to JPEG
                _, img_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                files = [("files", ("frame.jpg", img_encoded.tobytes(), "image/jpeg"))]
                data = {
                    "conf_m1": conf_m1,
                    "conf_m2": conf_m2,
                    "debug": False,
                }

                try:
                    t_req_start = time.time()
                    resp = requests.post(endpoint, files=files, data=data, timeout=5.0)
                    req_latency = int((time.time() - t_req_start) * 1000)

                    if resp.status_code == 200:
                        res_json = resp.json()
                        results = res_json.get("results", [])
                        if results:
                            r0 = results[0]
                            last_plate_text = r0.get("plate_text", "No Plate")
                            last_province = r0.get("province", "")
                            last_country = r0.get("country", "")
                            last_latency = r0.get("latency_ms", req_latency)
                            last_status = "Detected" if r0.get("plate_text") else "No Plate"
                            
                            # Print detection summary
                            if last_status == "Detected":
                                print(f"[{time.strftime('%H:%M:%S')}] 🚗 {last_country} | {last_plate_text} ({last_province}) | Latency: {last_latency}ms")
                        else:
                            last_status = "No Detections"
                    else:
                        last_status = f"HTTP {resp.status_code}"
                except requests.RequestException as e:
                    last_status = f"Conn Err: {type(e).__name__}"

            # Render live overlay window if not headless
            if not headless:
                disp = frame.copy()
                h, w = disp.shape[:2]

                # Top info banner
                cv2.rectangle(disp, (0, 0), (w, 55), (10, 15, 25), -1)
                cv2.line(disp, (0, 55), (w, 55), (0, 240, 255), 1)

                info_txt = f"LPR Stream: {last_plate_text} | {last_province} | {last_latency}ms | Status: {last_status}"
                cv2.putText(disp, info_txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 240, 255), 2)

                cv2.imshow("Thai & Laos LPR — RTSP Edge Ingestor", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):  # ESC or 'q'
                    break

    except KeyboardInterrupt:
        print("\nStopping RTSP streamer...")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTSP Edge Streamer for Thai & Laos LPR")
    parser.add_argument("--source", type=str, default="0", help="RTSP stream URL (rtsp://...) or webcam index (default: 0)")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080/api/detect/image", help="Target API endpoint (Local, ngrok, or Cloud Run)")
    parser.add_argument("--fps", type=float, default=2.0, help="Target sampling rate in FPS (default: 2.0)")
    parser.add_argument("--conf_m1", type=float, default=0.35, help="Model 1 Plate confidence threshold")
    parser.add_argument("--conf_m2", type=float, default=0.25, help="Model 2 Components confidence threshold")
    parser.add_argument("--no-motion", dest="motion_gate", action="store_false", help="Disable motion detection gating")
    parser.add_argument("--headless", action="store_true", help="Run without graphical preview window")
    args = parser.parse_args()

    run_rtsp_streamer(
        source=args.source,
        endpoint=args.endpoint,
        target_fps=args.fps,
        conf_m1=args.conf_m1,
        conf_m2=args.conf_m2,
        headless=args.headless,
        motion_gate=args.motion_gate,
    )

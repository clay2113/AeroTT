from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
MODEL_PATH = ROOT / "yolov8n-seg.pt"

# Keep Ultralytics settings in the project so the app does not need to write
# into the user's roaming profile.
ULTRALYTICS_DIR = ROOT / ".ultralytics"
ULTRALYTICS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

import cv2
import numpy as np
import torch
from ultralytics import YOLO


MODEL_LOCK = Lock()
MODEL: YOLO | None = None
DEVICE: str | None = None
TARGET_CLASSES = {0: "person", 1: "bicycle"}


def select_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"

    major, minor = torch.cuda.get_device_capability()
    architecture = f"sm_{major}{minor}"
    if architecture not in torch.cuda.get_arch_list():
        return "cpu"
    return "cuda:0"


def get_model() -> YOLO:
    global MODEL, DEVICE
    with MODEL_LOCK:
        if MODEL is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            DEVICE = select_device()
            MODEL = YOLO(str(MODEL_PATH))
        return MODEL


def warm_up_model() -> None:
    model = get_model()
    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    model.predict(blank, imgsz=320, conf=0.25, classes=[0], device=DEVICE or select_device(), verbose=False)


def clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def parse_data_url(data_url: str) -> bytes:
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    return base64.b64decode(encoded)


def encode_image(image: np.ndarray, extension: str, params: list[int] | None = None) -> str:
    ok, encoded = cv2.imencode(extension, image, params or [])
    if not ok:
        raise ValueError(f"Could not encode image as {extension}")
    mime = "image/png" if extension == ".png" else "image/jpeg"
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def build_empty_mask(frame: np.ndarray) -> np.ndarray:
    return np.zeros(frame.shape[:2], dtype=np.uint8)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return mask

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)

    components, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    if components <= 1:
        return cleaned

    min_area = max(30, int(cleaned.shape[0] * cleaned.shape[1] * 0.00035))
    filtered = np.zeros_like(cleaned)
    for idx in range(1, components):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == idx] = 255
    return filtered


def mask_bounds(mask: np.ndarray) -> dict[str, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return {
        "x": int(xs.min()),
        "y": int(ys.min()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }


def render_overlay(frame: np.ndarray, mask: np.ndarray, bounds: dict[str, int] | None) -> np.ndarray:
    overlay = frame.copy()
    color = np.zeros_like(frame)
    color[:, :, 0] = 255
    color[:, :, 1] = 210
    color[:, :, 2] = 30
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(frame, 0.42, color, 0.58, 0)[mask_bool]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (34, 255, 180), 2)

    if bounds:
        x, y, w, h = bounds["x"], bounds["y"], bounds["width"], bounds["height"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 220, 150), 2)

    return overlay


def render_mask(mask: np.ndarray) -> np.ndarray:
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[:, :, 0] = (mask > 0).astype(np.uint8) * 230
    mask_rgb[:, :, 1] = (mask > 0).astype(np.uint8) * 255
    mask_rgb[:, :, 2] = (mask > 0).astype(np.uint8) * 235
    return mask_rgb


def detection_color(class_id: int) -> tuple[int, int, int]:
    if class_id == 1:
        return (255, 215, 85)
    return (90, 235, 150)


def render_classifier(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    classified = frame.copy()

    for detection in detections:
        box = detection.get("box") or {}
        x1 = int(box.get("x1", 0))
        y1 = int(box.get("y1", 0))
        x2 = int(box.get("x2", 0))
        y2 = int(box.get("y2", 0))
        class_id = int(detection.get("classId", 0))
        color = detection_color(class_id)
        label = f"{detection.get('label', 'object')} {float(detection.get('confidence', 0)) * 100:.0f}%"

        cv2.rectangle(classified, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_width, label_height = text_size
        label_top = max(0, y1 - label_height - baseline - 8)
        label_bottom = label_top + label_height + baseline + 8
        label_right = min(frame.shape[1] - 1, x1 + label_width + 12)
        cv2.rectangle(classified, (x1, label_top), (label_right, label_bottom), color, cv2.FILLED)
        cv2.putText(
            classified,
            label,
            (x1 + 6, label_bottom - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (8, 14, 11),
            2,
            cv2.LINE_AA,
        )

    return classified


def build_description(
    detections: list[dict[str, Any]],
    mask_pixels: int,
    bounds: dict[str, int] | None,
    frontal_area_m2: float,
    cda_m2: float,
    drag_coefficient: float,
    calibration_mode: str,
    confidence_threshold: float,
    include_bicycle: bool,
) -> str:
    if not detections:
        return (
            f"No rider or bike class is above the {confidence_threshold:.2f} confidence threshold. "
            "Frontal area and CdA are paused until the classifier locks onto the subject."
        )

    by_label: dict[str, list[dict[str, Any]]] = {}
    for detection in detections:
        by_label.setdefault(str(detection["label"]), []).append(detection)

    label_parts = []
    for label, items in by_label.items():
        best = max(float(item["confidence"]) for item in items)
        suffix = "" if len(items) == 1 else f" x{len(items)}"
        label_parts.append(f"{label}{suffix} at {best * 100:.0f}%")

    subject = ", ".join(label_parts)
    box_text = ""
    if bounds:
        box_text = f" The active silhouette spans {bounds['width']} x {bounds['height']} px."

    bike_note = ""
    if include_bicycle and "bicycle" not in by_label:
        bike_note = " Bicycle class is not currently separated, so the estimate is rider-dominant."

    calibration_text = "auto mask-width calibration" if calibration_mode == "auto" else "manual point calibration"
    return (
        f"Classifier sees {subject}. Mask area is {mask_pixels:,} px, giving "
        f"A {frontal_area_m2:.4f} m^2 and CdA {cda_m2:.4f} m^2 with Cd {drag_coefficient:.2f} "
        f"using {calibration_text}.{box_text}{bike_note}"
    )


def analyze_frame(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()

    image_bytes = parse_data_url(str(payload.get("image", "")))
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image frame.")

    confidence = clamp_float(payload.get("confidence"), 0.25, 0.05, 0.9)
    drag_coefficient = clamp_float(payload.get("dragCoefficient"), 0.72, 0.2, 1.5)
    reference_real_width_m = clamp_float(payload.get("referenceRealWidthM"), 0.42, 0.05, 2.5)
    reference_pixel_width = clamp_float(payload.get("referencePixelWidth"), 0.0, 0.0, 5000.0)
    calibration_mode = str(payload.get("calibrationMode") or "auto")
    include_bicycle = bool(payload.get("includeBicycle", True))
    imgsz = clamp_int(payload.get("imageSize"), 512, 320, 960)

    classes = [0, 1] if include_bicycle else [0]
    model = get_model()
    device = DEVICE or select_device()
    result = model.predict(
        frame,
        imgsz=imgsz,
        conf=confidence,
        classes=classes,
        device=device,
        verbose=False,
    )[0]

    combined = build_empty_mask(frame)
    detections: list[dict[str, Any]] = []

    if result.boxes is not None:
        masks = result.masks.data.detach().cpu().numpy() if result.masks is not None else []
        classes_np = result.boxes.cls.detach().cpu().numpy().astype(int)
        confs_np = result.boxes.conf.detach().cpu().numpy()
        boxes_np = result.boxes.xyxy.detach().cpu().numpy()

        for idx, cls_id in enumerate(classes_np):
            if cls_id not in TARGET_CLASSES:
                continue
            if idx < len(masks):
                mask = masks[idx]
                resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask_uint8 = (resized >= 0.5).astype(np.uint8) * 255
                combined = cv2.bitwise_or(combined, mask_uint8)

            x1, y1, x2, y2 = boxes_np[idx]
            detections.append(
                {
                    "classId": int(cls_id),
                    "label": TARGET_CLASSES.get(int(cls_id), str(cls_id)),
                    "confidence": round(float(confs_np[idx]), 4),
                    "box": {
                        "x1": int(max(0, round(float(x1)))),
                        "y1": int(max(0, round(float(y1)))),
                        "x2": int(min(frame.shape[1] - 1, round(float(x2)))),
                        "y2": int(min(frame.shape[0] - 1, round(float(y2)))),
                    },
                }
            )

    detections.sort(key=lambda item: float(item["confidence"]), reverse=True)
    combined = clean_mask(combined)
    bounds = mask_bounds(combined)
    mask_pixels = int(np.count_nonzero(combined))

    effective_reference_px = reference_pixel_width
    if calibration_mode == "auto" or effective_reference_px <= 0:
        effective_reference_px = float(bounds["width"]) if bounds else 0.0

    if mask_pixels > 0 and effective_reference_px > 0:
        meters_per_pixel = reference_real_width_m / effective_reference_px
        frontal_area_m2 = mask_pixels * meters_per_pixel * meters_per_pixel
    else:
        meters_per_pixel = 0.0
        frontal_area_m2 = 0.0

    cda_m2 = frontal_area_m2 * drag_coefficient
    elapsed_ms = (time.perf_counter() - started) * 1000

    overlay = render_overlay(frame, combined, bounds)
    mask_view = render_mask(combined)
    classifier = render_classifier(frame, detections)
    description = build_description(
        detections,
        mask_pixels,
        bounds,
        frontal_area_m2,
        cda_m2,
        drag_coefficient,
        "auto" if calibration_mode == "auto" else "manual",
        confidence,
        include_bicycle,
    )

    return {
        "ok": True,
        "overlayImage": encode_image(overlay, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 82]),
        "maskImage": encode_image(mask_view, ".png"),
        "classifierImage": encode_image(classifier, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 84]),
        "description": description,
        "metrics": {
            "frontalAreaM2": frontal_area_m2,
            "cdaM2": cda_m2,
            "dragCoefficient": drag_coefficient,
            "maskPixels": mask_pixels,
            "referencePixelWidth": effective_reference_px,
            "referenceRealWidthM": reference_real_width_m,
            "metersPerPixel": meters_per_pixel,
            "processingMs": elapsed_ms,
            "backendFps": 1000 / elapsed_ms if elapsed_ms > 0 else 0,
            "frameWidth": int(frame.shape[1]),
            "frameHeight": int(frame.shape[0]),
            "device": device,
            "calibrationMode": "auto" if calibration_mode == "auto" else "manual",
            "bounds": bounds,
            "detections": detections,
        },
    }


class AeroRequestHandler(BaseHTTPRequestHandler):
    server_version = "AeroLive/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} {fmt % args}")

    def send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def send_static(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = unquote(parsed.path)

        if route == "/api/status":
            self.send_json(
                {
                    "ok": True,
                    "modelPath": str(MODEL_PATH),
                    "modelExists": MODEL_PATH.exists(),
                    "modelLoaded": MODEL is not None,
                    "device": DEVICE or select_device(),
                }
            )
            return

        if route in {"/", "/index.html"}:
            self.send_static(WEB_ROOT / "index.html")
            return

        if route.startswith("/web/"):
            relative = route.removeprefix("/web/").lstrip("/")
            target = (WEB_ROOT / relative).resolve()
            if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self.send_static(target)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self.send_json({"ok": False, "error": "Empty request body."}, HTTPStatus.BAD_REQUEST)
            return
        if content_length > 12_000_000:
            self.send_json({"ok": False, "error": "Frame payload is too large."}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return

        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
            response = analyze_frame(payload)
            self.send_json(response)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def run_server(host: str, port: int) -> None:
    print("Loading segmentation model...", flush=True)
    get_model()
    print("Warming up inference...", flush=True)
    warm_up_model()
    server = ThreadingHTTPServer((host, port), AeroRequestHandler)
    url_host = "localhost" if host in {"127.0.0.1", "0.0.0.0"} else host
    print(f"Aero live app running at http://{url_host}:{port}", flush=True)
    print(f"Using {MODEL_PATH.name} on {DEVICE or select_device()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live frontal-area and CdA estimator.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()

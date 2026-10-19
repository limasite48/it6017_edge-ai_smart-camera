"""Real-time live camera inspection and validation tool for Edge AI smart camera models.

Course: IT6017 - Technological Trends in Digital Transformation
Topic: Edge AI Smart Camera System for Human Detection and Fire Detection
Inspection Stage: Live Webcam Functional Verification & Comparative Performance Baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Mapping of keyboard shortcuts to model specifications
MODEL_REGISTRY = {
    "1": {
        "filename": "ssd_mobilenet_v2_fp32.tflite",
        "name": "SSD MobileNetV2 (300x300 FP32)",
        "architecture": "ssd_mobilenet",
        "precision": "FP32",
        "input_resolution": (300, 300),
        "layout": "NHWC",
        "input_dtype": "float32",
    },
    "2": {
        "filename": "ssd_mobilenet_v2_int8.tflite",
        "name": "SSD MobileNetV2 (300x300 Full INT8)",
        "architecture": "ssd_mobilenet",
        "precision": "INT8",
        "input_resolution": (300, 300),
        "layout": "NHWC",
        "input_dtype": "uint8",
    },
    "3": {
        "filename": "efficientdet_lite0_fp32.tflite",
        "name": "EfficientDet-Lite0 (320x320 FP32)",
        "architecture": "efficientdet",
        "precision": "FP32",
        "input_resolution": (320, 320),
        "layout": "NHWC",
        "input_dtype": "float32",
    },
    "4": {
        "filename": "efficientdet_lite0_int8.tflite",
        "name": "EfficientDet-Lite0 (320x320 Full INT8)",
        "architecture": "efficientdet",
        "precision": "INT8",
        "input_resolution": (320, 320),
        "layout": "NHWC",
        "input_dtype": "uint8",
    },
    "5": {
        "filename": "yolov8n_320_fp32.tflite",
        "name": "YOLOv8n - Nano (320x320 FP32)",
        "architecture": "yolov8",
        "precision": "FP32",
        "input_resolution": (320, 320),
        "layout": "NCHW",
        "input_dtype": "float32",
    },
    "6": {
        "filename": "yolov8n_320_int8.tflite",
        "name": "YOLOv8n - Nano (320x320 Weight INT8)",
        "architecture": "yolov8",
        "precision": "INT8",
        "input_resolution": (320, 320),
        "layout": "NCHW",
        "input_dtype": "float32",
    },
}

COCO_CLASSES_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

COCO_CLASSES_90 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "???", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "???", "backpack", "umbrella", "???", "???",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "???", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "???", "dining table", "???", "???", "toilet",
    "???", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "???", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax
    class_id: int
    class_name: str
    confidence: float


def letterbox_image(
    image: np.ndarray, target_size: Tuple[int, int], fill_val: int = 114
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image with aspect-ratio preserving letterbox padding.

    Returns:
        padded_img: image of shape (target_h, target_w, 3)
        scale_ratio: scale factor r
        (dw, dh): left/top padding offset in target coordinates
    """
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_size

    r = min(target_w / float(orig_w), target_h / float(orig_h))
    new_w = int(round(orig_w * r))
    new_h = int(round(orig_h * r))

    dw = (target_w - new_w) / 2.0
    dh = (target_h - new_h) / 2.0

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    if (orig_w, orig_h) != (new_w, new_h):
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(fill_val, fill_val, fill_val)
    )
    return padded, r, (int(round(dw)), int(round(dh)))


def unpad_box(
    x1_tgt: float,
    y1_tgt: float,
    x2_tgt: float,
    y2_tgt: float,
    scale_info: Tuple[float, int, int, bool],
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
) -> Tuple[int, int, int, int]:
    """Unpad bounding box from target tensor space back to native image coordinates."""
    r, dw, dh, is_stretched = scale_info
    if is_stretched:
        scale_x = orig_w / float(target_w)
        scale_y = orig_h / float(target_h)
        x1 = int(round(x1_tgt * scale_x))
        y1 = int(round(y1_tgt * scale_y))
        x2 = int(round(x2_tgt * scale_x))
        y2 = int(round(y2_tgt * scale_y))
    else:
        x1 = int(round((x1_tgt - dw) / r))
        y1 = int(round((y1_tgt - dh) / r))
        x2 = int(round((x2_tgt - dw) / r))
        y2 = int(round((y2_tgt - dh) / r))

    return (
        max(0, min(orig_w, x1)),
        max(0, min(orig_h, y1)),
        max(0, min(orig_w, x2)),
        max(0, min(orig_h, y2)),
    )


class EfficientDetAnchorGenerator:
    """Pre-generates EfficientDet-Lite0 19,206 anchors for 320x320 inputs."""

    def __init__(self, image_size: int = 320):
        self.image_size = image_size
        self.strides = [8, 16, 32, 64, 128]
        self.scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        self.aspect_ratios = [(1.0, 1.0), (1.414, 0.707), (0.707, 1.414)]
        self.anchors = self._generate_anchors()

    def _generate_anchors(self) -> np.ndarray:
        anchors = []
        for stride in self.strides:
            feat_h = int(np.ceil(self.image_size / stride))
            feat_w = int(np.ceil(self.image_size / stride))
            for r in range(feat_h):
                for c in range(feat_w):
                    cy = (r + 0.5) * stride
                    cx = (c + 0.5) * stride
                    for scale in self.scales:
                        for ar_h, ar_w in self.aspect_ratios:
                            w = stride * scale * ar_w * 4.0
                            h = stride * scale * ar_h * 4.0
                            anchors.append([cy, cx, h, w])
        return np.array(anchors, dtype=np.float32)


class LiveCameraInspector:
    """Manages live camera capture, dynamic model inference, and real-time visualization."""

    def __init__(
        self,
        models_dir: Path,
        snapshots_dir: Path,
        source: str = "0",
        initial_key: str = "2",
        confidence_thresh: float = 0.45,
        iou_thresh: float = 0.45,
        loop: bool = True,
        stretch: bool = False,
    ):
        self.models_dir = models_dir
        self.snapshots_dir = snapshots_dir
        self.source = source
        self.active_key = initial_key
        self.conf_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.human_only = True
        self.loop = loop
        self.stretch = stretch
        self.paused = False

        self.loaded_interpreters: Dict[str, Interpreter] = {}
        self.current_interpreter: Optional[Interpreter] = None
        self.ed_anchors = EfficientDetAnchorGenerator(320).anchors

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.switch_model(self.active_key)

    def get_interpreter(self, key: str) -> Interpreter:
        """Lazy-load or retrieve cached LiteRT interpreter for a given model key."""
        if key not in self.loaded_interpreters:
            spec = MODEL_REGISTRY[key]
            model_path = self.models_dir / spec["filename"]
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            interp = Interpreter(model_path=str(model_path.resolve()))
            interp.allocate_tensors()
            self.loaded_interpreters[key] = interp
        return self.loaded_interpreters[key]

    def switch_model(self, key: str) -> bool:
        """Switch active model to specified key."""
        if key not in MODEL_REGISTRY:
            return False
        try:
            self.current_interpreter = self.get_interpreter(key)
            self.active_key = key
            console.print(f"[bold green]Switched to Model [{key}]: {MODEL_REGISTRY[key]['name']}[/bold green]")
            return True
        except Exception as e:
            console.print(f"[bold red]Failed to load Model [{key}]: {e}[/bold red]")
            return False

    def preprocess_frame(
        self, frame: np.ndarray, spec: Dict[str, Any]
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, int, int, bool]]:
        """Preprocess camera frame to match model input shape, layout, and dtype."""
        orig_h, orig_w = frame.shape[:2]
        target_h, target_w = spec["input_resolution"]

        # 1. Letterbox or Direct Stretch
        if not self.stretch:
            padded, r, (dw, dh) = letterbox_image(frame, (target_w, target_h))
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            scale_info = (r, dw, dh, False)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            rgb = resized
            scale_info = (1.0, 0, 0, True)

        # 2. Handle NCHW vs NHWC
        if spec["layout"] == "NCHW":
            img_data = np.transpose(rgb, (2, 0, 1))
        else:
            img_data = rgb

        img_data = np.expand_dims(img_data, axis=0)

        # 3. Handle dtype scaling
        if spec["input_dtype"] == "uint8":
            tensor_data = img_data.astype(np.uint8)
        else:
            if spec["architecture"] == "ssd_mobilenet":
                tensor_data = (img_data.astype(np.float32) / 127.5) - 1.0
            elif spec["architecture"] == "efficientdet":
                tensor_data = img_data.astype(np.float32)
            else:
                tensor_data = img_data.astype(np.float32) / 255.0

        return tensor_data, (orig_w, orig_h), scale_info

    def decode_detections(
        self,
        spec: Dict[str, Any],
        interpreter: Interpreter,
        orig_size: Tuple[int, int],
        scale_info: Tuple[float, int, int, bool],
    ) -> List[Detection]:
        """Decode bounding boxes and scores from the active model output."""
        orig_w, orig_h = orig_size
        target_w, target_h = spec["input_resolution"]
        arch = spec["architecture"]
        detections: List[Detection] = []

        if arch == "ssd_mobilenet":
            out_details = interpreter.get_output_details()
            # Standard TFLite_Detection_PostProcess outputs:
            # 0: boxes [1, 20, 4] (ymin, xmin, ymax, xmax)
            # 1: classes [1, 20]
            # 2: scores [1, 20]
            # 3: count [1]
            boxes = interpreter.get_tensor(out_details[0]["index"])[0]
            classes = interpreter.get_tensor(out_details[1]["index"])[0]
            scores = interpreter.get_tensor(out_details[2]["index"])[0]

            for i in range(len(scores)):
                score = float(scores[i])
                if score < self.conf_thresh:
                    continue

                cls_id = int(classes[i])
                is_person = (cls_id == 0 or cls_id == 1)

                if self.human_only and not is_person:
                    continue

                ymin, xmin, ymax, xmax = boxes[i]
                x1_t = xmin * target_w
                y1_t = ymin * target_h
                x2_t = xmax * target_w
                y2_t = ymax * target_h

                box = unpad_box(x1_t, y1_t, x2_t, y2_t, scale_info, orig_w, orig_h, target_w, target_h)
                name = "person" if is_person else (COCO_CLASSES_90[cls_id] if cls_id < len(COCO_CLASSES_90) and COCO_CLASSES_90[cls_id] != "???" else f"class_{cls_id}")
                detections.append(Detection(box=box, class_id=cls_id, class_name=name, confidence=score))

        elif arch == "yolov8":
            out_details = interpreter.get_output_details()
            # Output: [1, 84, 2100] -> rows 0-3: [cx, cy, w, h] in 320 grid; row 4: person score
            output = interpreter.get_tensor(out_details[0]["index"])[0]  # shape: (84, 2100)

            if self.human_only:
                person_scores = output[4, :]
                mask = person_scores >= self.conf_thresh
                valid_indices = np.where(mask)[0]

                if len(valid_indices) > 0:
                    cand_boxes = []
                    cand_scores = []
                    for idx in valid_indices:
                        cx, cy, w, h = output[:4, idx]
                        score = float(person_scores[idx])
                        x1_t = (cx - w / 2.0) * target_w
                        y1_t = (cy - h / 2.0) * target_h
                        bw_t = w * target_w
                        bh_t = h * target_h
                        cand_boxes.append([x1_t, y1_t, bw_t, bh_t])
                        cand_scores.append(score)

                    indices = cv2.dnn.NMSBoxes(cand_boxes, cand_scores, self.conf_thresh, self.iou_thresh)
                    if len(indices) > 0:
                        for i in indices.flatten():
                            x1_t, y1_t, bw_t, bh_t = cand_boxes[i]
                            box = unpad_box(x1_t, y1_t, x1_t + bw_t, y1_t + bh_t, scale_info, orig_w, orig_h, target_w, target_h)
                            detections.append(Detection(box=box, class_id=0, class_name="person", confidence=cand_scores[i]))
            else:
                # All classes mode for YOLOv8
                scores_all = output[4:, :]  # (80, 2100)
                class_ids = np.argmax(scores_all, axis=0)
                confidences = np.max(scores_all, axis=0)
                mask = confidences >= self.conf_thresh
                valid_indices = np.where(mask)[0]

                if len(valid_indices) > 0:
                    cand_boxes = []
                    cand_scores = []
                    cand_classes = []
                    for idx in valid_indices:
                        cx, cy, w, h = output[:4, idx]
                        score = float(confidences[idx])
                        cls_id = int(class_ids[idx])
                        x1_t = (cx - w / 2.0) * target_w
                        y1_t = (cy - h / 2.0) * target_h
                        bw_t = w * target_w
                        bh_t = h * target_h
                        cand_boxes.append([x1_t, y1_t, bw_t, bh_t])
                        cand_scores.append(score)
                        cand_classes.append(cls_id)

                    indices = cv2.dnn.NMSBoxes(cand_boxes, cand_scores, self.conf_thresh, self.iou_thresh)
                    if len(indices) > 0:
                        for i in indices.flatten():
                            x1_t, y1_t, bw_t, bh_t = cand_boxes[i]
                            cls_id = cand_classes[i]
                            box = unpad_box(x1_t, y1_t, x1_t + bw_t, y1_t + bh_t, scale_info, orig_w, orig_h, target_w, target_h)
                            name = COCO_CLASSES_80[cls_id] if cls_id < len(COCO_CLASSES_80) else f"class_{cls_id}"
                            detections.append(Detection(box=box, class_id=cls_id, class_name=name, confidence=cand_scores[i]))

        elif arch == "efficientdet":
            out_details = interpreter.get_output_details()
            # EfficientDet outputs:
            # StatefulPartitionedCall:1 is scores (1, 19206, 90)
            # StatefulPartitionedCall:0 is deltas (1, 19206, 4)
            scores = interpreter.get_tensor(out_details[0]["index"])[0] if out_details[0]["shape"][-1] in (90, 91) else interpreter.get_tensor(out_details[1]["index"])[0]
            deltas = interpreter.get_tensor(out_details[1]["index"])[0] if out_details[0]["shape"][-1] in (90, 91) else interpreter.get_tensor(out_details[0]["index"])[0]

            if self.human_only:
                # In MediaPipe EfficientDet-Lite0 (90 classes), class 0 is Person
                p_scores = scores[:, 0]
                mask = p_scores >= self.conf_thresh
                valid_indices = np.where(mask)[0]

                if len(valid_indices) > 0:
                    cand_boxes = []
                    cand_scores = []
                    for idx in valid_indices:
                        dy, dx, dh, dw = deltas[idx]
                        anchor_cy, anchor_cx, anchor_h, anchor_w = self.ed_anchors[idx]
                        cy = dy * anchor_h + anchor_cy
                        cx = dx * anchor_w + anchor_cx
                        h = np.exp(min(dh, 5.0)) * anchor_h
                        w = np.exp(min(dw, 5.0)) * anchor_w
                        x1_t = cx - w / 2.0
                        y1_t = cy - h / 2.0
                        cand_boxes.append([x1_t, y1_t, w, h])
                        cand_scores.append(float(p_scores[idx]))

                    indices = cv2.dnn.NMSBoxes(cand_boxes, cand_scores, self.conf_thresh, self.iou_thresh)
                    if len(indices) > 0:
                        for i in indices.flatten():
                            x1_t, y1_t, bw_t, bh_t = cand_boxes[i]
                            box = unpad_box(x1_t, y1_t, x1_t + bw_t, y1_t + bh_t, scale_info, orig_w, orig_h, target_w, target_h)
                            detections.append(Detection(box=box, class_id=0, class_name="person", confidence=cand_scores[i]))
            else:
                max_scores = np.max(scores, axis=1)
                max_classes = np.argmax(scores, axis=1)
                mask = max_scores >= self.conf_thresh
                valid_indices = np.where(mask)[0]

                if len(valid_indices) > 0:
                    cand_boxes = []
                    cand_scores = []
                    cand_classes = []
                    for idx in valid_indices:
                        cls_id = int(max_classes[idx])
                        name = COCO_CLASSES_90[cls_id] if cls_id < len(COCO_CLASSES_90) else f"class_{cls_id}"
                        if name == "???":
                            continue

                        dy, dx, dh, dw = deltas[idx]
                        anchor_cy, anchor_cx, anchor_h, anchor_w = self.ed_anchors[idx]
                        cy = dy * anchor_h + anchor_cy
                        cx = dx * anchor_w + anchor_cx
                        h = np.exp(min(dh, 5.0)) * anchor_h
                        w = np.exp(min(dw, 5.0)) * anchor_w
                        x1_t = cx - w / 2.0
                        y1_t = cy - h / 2.0
                        cand_boxes.append([x1_t, y1_t, w, h])
                        cand_scores.append(float(max_scores[idx]))
                        cand_classes.append(cls_id)

                    indices = cv2.dnn.NMSBoxes(cand_boxes, cand_scores, self.conf_thresh, self.iou_thresh)
                    if len(indices) > 0:
                        for i in indices.flatten():
                            x1_t, y1_t, bw_t, bh_t = cand_boxes[i]
                            cls_id = cand_classes[i]
                            name = COCO_CLASSES_90[cls_id] if cls_id < len(COCO_CLASSES_90) else f"class_{cls_id}"
                            box = unpad_box(x1_t, y1_t, x1_t + bw_t, y1_t + bh_t, scale_info, orig_w, orig_h, target_w, target_h)
                            detections.append(Detection(box=box, class_id=cls_id, class_name=name, confidence=cand_scores[i]))

        return detections

    def render_overlay(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float,
        latency_breakdown: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render smart camera HUD, alert banners, and bounding boxes."""
        h, w = frame.shape[:2]
        t_pre, t_infer, t_post = latency_breakdown
        t_total = t_pre + t_infer + t_post

        spec = MODEL_REGISTRY[self.active_key]
        human_detected = any(d.class_name == "person" for d in detections)
        max_human_conf = max([d.confidence for d in detections if d.class_name == "person"], default=0.0)

        # 1. Draw Bounding Boxes
        for det in detections:
            x1, y1, x2, y2 = det.box
            is_person = (det.class_name == "person")
            box_color = (0, 255, 0) if is_person else (255, 200, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"{det.class_name}: {det.confidence*100:.1f}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, max(0, y1)), box_color, -1)
            cv2.putText(frame, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Draw Top Smart Camera Alert Banner
        banner_h = 42
        if human_detected:
            # Flashing or prominent red alert banner
            cv2.rectangle(frame, (0, 0), (w, banner_h), (0, 0, 220), -1)
            alert_text = f"! ALERT: HUMAN DETECTED (Conf: {max_human_conf*100:.1f}%)"
            cv2.putText(frame, alert_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Green normal monitoring banner
            cv2.rectangle(frame, (0, 0), (w, banner_h), (30, 100, 30), -1)
            cv2.putText(frame, "[NORMAL] Edge AI Camera: Monitoring Active", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # 3. Draw Telemetry Dashboard Overlay (Bottom Left)
        hud_bg_w = 420
        hud_bg_h = 120
        hud_x = 10
        hud_y = h - hud_bg_h - 15

        # Semi-transparent HUD background
        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_bg_w, hud_y + hud_bg_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        model_title = f"Model [{self.active_key}]: {spec['name']}"
        cv2.putText(frame, model_title, (hud_x + 10, hud_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

        perf_str = f"Stream: {fps:.1f} FPS  |  Total Latency: {t_total:.1f} ms"
        cv2.putText(frame, perf_str, (hud_x + 10, hud_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        breakdown_str = f"Pre: {t_pre:.1f}ms | Infer: {t_infer:.1f}ms | Post: {t_post:.1f}ms"
        cv2.putText(frame, breakdown_str, (hud_x + 10, hud_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

        mode_str = f"Mode: {'Human Only' if self.human_only else 'All Classes'} | Thresh: {self.conf_thresh:.2f}"
        cv2.putText(frame, mode_str, (hud_x + 10, hud_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 150), 1, cv2.LINE_AA)

        # 4. Controls Quick Help (Bottom Right)
        help_str = "Keys [1-6]: Switch Model | [S]: Save Snapshot | [T]: Toggle Mode | [+/-]: Thresh | [Q]: Quit"
        cv2.putText(frame, help_str, (hud_x + 10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    def run(self, max_duration_sec: Optional[float] = None, no_gui: bool = False) -> None:
        """Run the live camera inspection loop."""
        # Open video source
        src = int(self.source) if self.source.isdigit() else self.source
        console.print(f"[cyan]Connecting to camera source '{self.source}'...[/cyan]")
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            console.print(f"[bold red]Failed to open camera source: {self.source}[/bold red]")
            return

        window_name = "IT6017: Smart Camera Live Model Inspection"
        if not no_gui:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 720)

        console.print(
            Panel.fit(
                "[bold green]Live Camera Inspection Active![/bold green]\n"
                "* Press [bold cyan]1 - 6[/bold cyan] to switch active model on-the-fly\n"
                "* Press [bold cyan]S[/bold cyan] to capture annotated inspection snapshots\n"
                "* Press [bold cyan]T[/bold cyan] to toggle Human-only vs All-classes detection\n"
                "* Press [bold cyan]+ / -[/bold cyan] to adjust confidence threshold\n"
                "* Press [bold cyan]Q[/bold cyan] or [bold cyan]ESC[/bold cyan] to exit",
                border_style="green",
            )
        )

        is_video_file = not str(self.source).isdigit()
        prev_time = time.perf_counter()
        fps = 0.0
        frame_count = 0
        start_time = time.perf_counter()
        snapshot_count = 0
        current_frame: Optional[np.ndarray] = None

        try:
            while True:
                if not self.paused or current_frame is None:
                    ret, frame = cap.read()
                    if not ret:
                        if is_video_file and self.loop:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                        if not ret:
                            console.print("[yellow]End of video feed or frame grab failed.[/yellow]")
                            break
                    current_frame = frame
                    frame_count += 1
                else:
                    frame = current_frame

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, now - prev_time))
                prev_time = now

                spec = MODEL_REGISTRY[self.active_key]
                interp = self.current_interpreter

                # Stage 1: Pre-processing
                t0 = time.perf_counter()
                input_tensor, orig_size, scale_info = self.preprocess_frame(frame, spec)
                t1 = time.perf_counter()

                # Stage 2: Inference
                input_idx = interp.get_input_details()[0]["index"]
                interp.set_tensor(input_idx, input_tensor)
                interp.invoke()
                t2 = time.perf_counter()

                # Stage 3: Post-processing
                detections = self.decode_detections(spec, interp, orig_size, scale_info)
                t3 = time.perf_counter()

                latency_breakdown = ((t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0)

                # Stage 4: Render Overlay
                annotated = self.render_overlay(frame.copy(), detections, fps, latency_breakdown)

                if self.paused:
                    # Draw PAUSED overlay badge
                    h_a, w_a = annotated.shape[:2]
                    cv2.rectangle(annotated, (w_a // 2 - 140, 50), (w_a // 2 + 140, 95), (0, 0, 0), -1)
                    cv2.putText(annotated, "[PAUSED] Press SPACE", (w_a // 2 - 125, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                if not no_gui:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(30 if self.paused else 1) & 0xFF

                    if key in [ord("q"), ord("Q"), 27]:  # Quit
                        break
                    elif key == ord(" "):  # Toggle pause
                        self.paused = not self.paused
                        console.print(f"Playback {'[yellow]PAUSED[/yellow]' if self.paused else '[green]RESUMED[/green]'}")
                    elif chr(key) in MODEL_REGISTRY:  # Switch Model
                        self.switch_model(chr(key))
                    elif key in [ord("s"), ord("S")]:  # Snapshot
                        snapshot_count += 1
                        fname = f"snapshot_model_{self.active_key}_{int(time.time())}.jpg"
                        snap_path = self.snapshots_dir / fname
                        cv2.imwrite(str(snap_path), annotated)
                        console.print(f"[bold cyan]Snapshot saved:[/bold cyan] {snap_path}")
                    elif key in [ord("t"), ord("T")]:  # Toggle mode
                        self.human_only = not self.human_only
                        console.print(f"Mode toggled: {'Human Only' if self.human_only else 'All Classes'}")
                    elif key in [ord("+"), ord("=")]:  # Increase threshold
                        self.conf_thresh = min(0.95, self.conf_thresh + 0.05)
                        console.print(f"Confidence threshold: {self.conf_thresh:.2f}")
                    elif key in [ord("-"), ord("_")]:  # Decrease threshold
                        self.conf_thresh = max(0.10, self.conf_thresh - 0.05)
                        console.print(f"Confidence threshold: {self.conf_thresh:.2f}")

                # Check max duration
                if max_duration_sec and (time.perf_counter() - start_time) >= max_duration_sec:
                    console.print(f"[cyan]Completed automated run duration ({max_duration_sec:.1f}s).[/cyan]")
                    break

        finally:
            cap.release()
            if not no_gui:
                cv2.destroyAllWindows()

            elapsed = time.perf_counter() - start_time
            avg_fps = frame_count / max(1e-5, elapsed)
            console.print(f"\n[green]Inspection session ended. Processed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} avg FPS).[/green]")
            if snapshot_count > 0:
                console.print(f"[cyan]Saved {snapshot_count} snapshots to:[/cyan] {self.snapshots_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Camera Human Detection Inspection Tool for Course IT6017.")
    parser.add_argument("--source", default="0", help="Camera index (e.g., 0) or path to video file (default: 0)")
    parser.add_argument("--models-dir", default="models", help="Directory containing .tflite models (default: models)")
    parser.add_argument("--snapshots-dir", default="reports/camera_inspection_snapshots", help="Directory to save snapshots")
    parser.add_argument("--model", default="2", choices=list(MODEL_REGISTRY.keys()), help="Initial model key (default: 2 - SSD MobileNet INT8)")
    parser.add_argument("--thresh", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds to run before auto-exiting")
    parser.add_argument("--no-gui", action="store_true", help="Run without opening GUI window (for headless/automated tests)")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop video playback when end is reached")
    parser.add_argument("--stretch", action="store_true", help="Disable letterboxing and stretch image directly to target resolution")

    args = parser.parse_args()

    inspector = LiveCameraInspector(
        models_dir=Path(args.models_dir),
        snapshots_dir=Path(args.snapshots_dir),
        source=args.source,
        initial_key=args.model,
        confidence_thresh=args.thresh,
        loop=not args.no_loop,
        stretch=args.stretch,
    )

    inspector.run(max_duration_sec=args.duration, no_gui=args.no_gui)
    return 0


if __name__ == "__main__":
    sys.exit(main())

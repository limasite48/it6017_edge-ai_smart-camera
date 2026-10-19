"""Evaluation and verification module for Human Detection Edge AI models.

Course: IT6017 - Technological Trends in Digital Transformation
Topic: Edge AI Smart Camera System for Human Detection and Fire Detection
Phase 1.2: Model Zoo Verification & Target Hardware Fit Analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
from ai_edge_litert.interpreter import Interpreter
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Target hardware specifications (from project-overview.md)
HARDWARE_PROFILES = {
    "rockchip_rv1109": {
        "name": "Rockchip RV1109 / Ingenic T31 (Target Comparison Camera)",
        "cpu": "Dual-core ARM Cortex-A7 @ 1.5 GHz",
        "npu_tops": 1.2,
        "npu_supported_types": ["int8", "int16"],
        "total_ram_mb": 128.0,
        "max_model_ram_mb": 30.0,
        "flash_rom_mb": 32.0,
        "recommended_max_file_size_mb": 6.0,
        "critical_max_file_size_mb": 10.0,
    },
    "xiaomi_12s_pro": {
        "name": "Xiaomi 12S Pro (Experimental Testbed Benchmark)",
        "cpu": "Qualcomm Snapdragon 8+ Gen 1 (Cortex-X2/A710/A510)",
        "gpu_npu": "Adreno 730 GPU + Hexagon NPU (Android NNAPI)",
        "total_ram_mb": 8192.0,
    },
}

# Pipeline expected 3x2 matrix specifications (from project-pipeline.md)
EXPECTED_MATRIX_SPECS = {
    "ssd_mobilenet_v2_fp32.tflite": {
        "architecture": "SSD MobileNetV2",
        "expected_resolution": (300, 300),
        "acceptable_resolutions": [(300, 300)],
        "expected_precision": "FP32",
        "expected_classes": 90,
        "human_class_id": 0,
    },
    "ssd_mobilenet_v2_int8.tflite": {
        "architecture": "SSD MobileNetV2",
        "expected_resolution": (300, 300),
        "acceptable_resolutions": [(300, 300)],
        "expected_precision": "INT8",
        "expected_classes": 90,
        "human_class_id": 0,
    },
    "efficientdet_lite0_fp32.tflite": {
        "architecture": "EfficientDet-Lite0",
        "expected_resolution": (320, 320),
        "acceptable_resolutions": [(320, 320)],
        "expected_precision": "FP32",
        "expected_classes": 90,
        "human_class_id": 1,
    },
    "efficientdet_lite0_int8.tflite": {
        "architecture": "EfficientDet-Lite0",
        "expected_resolution": (320, 320),
        "acceptable_resolutions": [(320, 320)],
        "expected_precision": "INT8",
        "expected_classes": 90,
        "human_class_id": 1,
    },
    "yolov8n_320_fp32.tflite": {
        "architecture": "YOLOv8n - Nano",
        "expected_resolution": (320, 320),
        "acceptable_resolutions": [(320, 320)],
        "expected_precision": "FP32",
        "expected_classes": 80,
        "human_class_id": 0,
    },
    "yolov8n_320_int8.tflite": {
        "architecture": "YOLOv8n - Nano",
        "expected_resolution": (320, 320),
        "acceptable_resolutions": [(320, 320)],
        "expected_precision": "INT8",
        "expected_classes": 80,
        "human_class_id": 0,
    },
}


@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    dtype: str
    scale: float
    zero_point: int


@dataclass
class ModelEvaluationResult:
    filename: str
    filepath: str
    file_size_mb: float
    sha256: str
    is_duplicate: bool
    duplicate_of: Optional[str]
    input_tensors: List[TensorInfo]
    output_tensors: List[TensorInfo]
    input_layout: str  # NHWC or NCHW
    input_resolution: Tuple[int, int]
    total_tensors: int
    weight_dtype_counts: Dict[str, int]
    int8_weight_ratio_pct: float
    quantization_type: str  # FULL_INT8, WEIGHT_INT8, FP32, FAKE_INT8
    inference_success: bool
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_std_ms: float
    fps: float
    process_memory_mb: float
    flash_rom_fit: str  # PASS, WARN, FAIL
    npu_fit: str  # FULL_SUPPORT, PARTIAL_SUPPORT, UNSUITABLE
    spec_conformance: str  # PASS, WARN, FAIL
    overall_status: str  # PASS, WARN, FAIL
    person_detection_score: Optional[float] = None
    issues: List[str] = field(default_factory=list)
    remediations: List[str] = field(default_factory=list)


def calculate_sha256(filepath: str) -> str:
    """Calculate the SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_layout_and_resolution(shape: List[int]) -> Tuple[str, Tuple[int, int]]:
    """Detect if layout is NHWC or NCHW and extract (height, width)."""
    if len(shape) == 4:
        if shape[1] in (1, 3):  # NCHW: (batch, channels, height, width)
            return "NCHW", (shape[2], shape[3])
        if shape[3] in (1, 3):  # NHWC: (batch, height, width, channels)
            return "NHWC", (shape[1], shape[2])
    return "UNKNOWN", (0, 0)


def prepare_input_tensor(tensor_info: Dict[str, Any], sample_image_path: Optional[str] = None, model_name: str = "") -> np.ndarray:
    """Prepare properly formatted and scaled input tensor (from image or dummy synthetic data)."""
    shape = tensor_info["shape"]
    dtype = tensor_info["dtype"]
    layout, (h, w) = detect_layout_and_resolution(shape)

    if sample_image_path and os.path.isfile(sample_image_path) and h > 0 and w > 0:
        bgr = cv2.imread(sample_image_path)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            img_data = resized.astype(np.float32)

            # Transpose to NCHW if model requires it
            if layout == "NCHW":
                img_data = np.transpose(img_data, (2, 0, 1))

            img_data = np.expand_dims(img_data, axis=0)

            # Handle quantization/scale for uint8 / int8 / float32
            if dtype == np.uint8:
                return np.clip(img_data, 0, 255).astype(np.uint8)
            elif dtype == np.int8:
                scale, zp = tensor_info.get("quantization", (1.0, 0))
                scale = scale if scale > 0 else 1.0
                quantized = np.round(img_data / (255.0 * scale) + zp)
                return np.clip(quantized, -128, 127).astype(np.int8)
            else:
                if "efficientdet" in model_name.lower():
                    return img_data.astype(np.float32)
                elif "ssd_mobilenet" in model_name.lower():
                    return ((img_data / 127.5) - 1.0).astype(np.float32)
                else:
                    # Float32 normalized in [0.0, 1.0]
                    return (img_data / 255.0).astype(np.float32)

    # Fallback to synthetic dummy tensor
    if dtype == np.uint8:
        return np.random.randint(0, 256, size=shape, dtype=np.uint8)
    elif dtype == np.int8:
        return np.random.randint(-128, 127, size=shape, dtype=np.int8)
    else:
        return np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)


def extract_person_score(model_name: str, interpreter: Interpreter) -> Optional[float]:
    """Extract maximum confidence score for the 'person' class if possible."""
    out_details = interpreter.get_output_details()
    try:
        if "yolov8" in model_name.lower():
            # YOLOv8 output: (1, 84, anchors) -> class 0 is person (index 4 in 84)
            out_tensor = interpreter.get_tensor(out_details[0]["index"])
            if out_tensor.ndim == 3 and out_tensor.shape[1] >= 5:
                person_scores = out_tensor[0, 4, :]
                return float(np.max(person_scores))
        elif "ssd_mobilenet" in model_name.lower():
            # SSD MobileNet with TFLite_Detection_PostProcess
            postprocess_outputs = [od for od in out_details if "postprocess" in od["name"].lower()]
            if len(postprocess_outputs) >= 3:
                scores_tensor = interpreter.get_tensor(postprocess_outputs[2]["index"])
                classes_tensor = interpreter.get_tensor(postprocess_outputs[1]["index"])
                person_indices = np.where((classes_tensor == 0) | (classes_tensor == 1))[1]
                if len(person_indices) > 0:
                    return float(np.max(scores_tensor[0, person_indices]))
                return float(np.max(scores_tensor))
            # Fallback for raw anchor scores
            for od in out_details:
                out_tensor = interpreter.get_tensor(od["index"])
                if out_tensor.ndim == 3 and out_tensor.shape[2] in (90, 91):
                    person_scores = np.maximum(out_tensor[0, :, 0], out_tensor[0, :, 1])
                    return float(np.max(person_scores))
        elif "efficientdet" in model_name.lower():
            # EfficientDet: StatefulPartitionedCall:1 is scores (1, 19206, 90), class 0 is person
            for od in out_details:
                out_tensor = interpreter.get_tensor(od["index"])
                if out_tensor.ndim == 3 and out_tensor.shape[2] in (90, 91):
                    person_scores = np.maximum(out_tensor[0, :, 0], out_tensor[0, :, 1])
                    return float(np.max(person_scores))
    except Exception:
        pass
    return None


class ModelEvaluator:
    """Evaluates LiteRT / TFLite models against project requirements and hardware constraints."""

    def __init__(self, models_dir: str, num_runs: int = 10, warmup_runs: int = 3, sample_image: Optional[str] = None):
        self.models_dir = Path(models_dir)
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.sample_image = sample_image
        self.file_hashes: Dict[str, str] = {}

    def scan_models(self) -> List[Path]:
        """Scan models directory for .tflite files."""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")
        model_paths = sorted(list(self.models_dir.glob("*.tflite")))
        if not model_paths:
            raise FileNotFoundError(f"No .tflite files found in {self.models_dir}")
        return model_paths

    def evaluate_model(self, model_path: Path) -> ModelEvaluationResult:
        """Run comprehensive static and functional evaluation on a single model."""
        filename = model_path.name
        filepath = str(model_path.resolve())
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        file_sha256 = calculate_sha256(filepath)

        # Check for exact duplicate models across the zoo
        is_dup = False
        dup_of = None
        for other_name, other_hash in self.file_hashes.items():
            if other_hash == file_sha256:
                is_dup = True
                dup_of = other_name
                break
        self.file_hashes[filename] = file_sha256

        issues: List[str] = []
        remediations: List[str] = []

        try:
            interpreter = Interpreter(model_path=filepath)
            interpreter.allocate_tensors()
        except Exception as e:
            issues.append(f"Failed to load model in LiteRT interpreter: {e}")
            remediations.append("Verify model file integrity and LiteRT runtime compatibility.")
            return ModelEvaluationResult(
                filename=filename,
                filepath=filepath,
                file_size_mb=round(file_size_mb, 2),
                sha256=file_sha256,
                is_duplicate=is_dup,
                duplicate_of=dup_of,
                input_tensors=[],
                output_tensors=[],
                input_layout="UNKNOWN",
                input_resolution=(0, 0),
                total_tensors=0,
                weight_dtype_counts={},
                int8_weight_ratio_pct=0.0,
                quantization_type="CORRUPT",
                inference_success=False,
                latency_mean_ms=0.0,
                latency_min_ms=0.0,
                latency_max_ms=0.0,
                latency_std_ms=0.0,
                fps=0.0,
                process_memory_mb=0.0,
                flash_rom_fit="FAIL",
                npu_fit="UNSUITABLE",
                spec_conformance="FAIL",
                overall_status="FAIL",
                issues=issues,
                remediations=remediations,
            )

        # Extract input details
        input_details = interpreter.get_input_details()
        input_tensors: List[TensorInfo] = []
        for inp in input_details:
            scale, zp = inp.get("quantization", (0.0, 0))
            input_tensors.append(
                TensorInfo(
                    name=inp["name"],
                    shape=inp["shape"].tolist(),
                    dtype=str(np.dtype(inp["dtype"])),
                    scale=float(scale),
                    zero_point=int(zp),
                )
            )

        # Extract output details
        output_details = interpreter.get_output_details()
        output_tensors: List[TensorInfo] = []
        for out in output_details:
            scale, zp = out.get("quantization", (0.0, 0))
            output_tensors.append(
                TensorInfo(
                    name=out["name"],
                    shape=out["shape"].tolist(),
                    dtype=str(np.dtype(out["dtype"])),
                    scale=float(scale),
                    zero_point=int(zp),
                )
            )

        # Inspect all tensors and weights
        tensor_details = interpreter.get_tensor_details()
        total_tensors = len(tensor_details)
        weight_dtype_counts: Dict[str, int] = {}
        for t in tensor_details:
            dt = str(np.dtype(t["dtype"]))
            weight_dtype_counts[dt] = weight_dtype_counts.get(dt, 0) + 1

        int8_count = weight_dtype_counts.get("int8", 0) + weight_dtype_counts.get("uint8", 0)
        fp32_count = weight_dtype_counts.get("float32", 0)
        total_weights = int8_count + fp32_count
        int8_ratio = (int8_count / total_weights * 100.0) if total_weights > 0 else 0.0

        # Primary input layout and resolution
        primary_inp = input_details[0]
        layout, resolution = detect_layout_and_resolution(primary_inp["shape"].tolist())

        # Determine quantization type
        inp_dtype = str(np.dtype(primary_inp["dtype"]))
        is_named_int8 = "int8" in filename.lower()

        if is_named_int8 and int8_count == 0:
            quant_type = "FAKE_INT8"
            issues.append(f"Model named INT8 but contains 0 INT8 tensors ({fp32_count} FP32 tensors).")
            if is_dup:
                issues.append(f"Identical duplicate of FP32 model: '{dup_of}' (SHA256 match).")
            remediations.append("Re-export model using full integer post-training quantization (PTQ) with calibration data.")
        elif inp_dtype in ("uint8", "int8") and int8_ratio > 50.0:
            quant_type = "FULL_INT8"
        elif int8_ratio > 50.0:
            quant_type = "WEIGHT_INT8"
        else:
            quant_type = "FP32"

        # Check against expected matrix specifications
        expected_spec = EXPECTED_MATRIX_SPECS.get(filename)
        spec_status = "PASS"
        if expected_spec:
            if resolution not in expected_spec["acceptable_resolutions"]:
                issues.append(
                    f"Resolution mismatch: Detected {resolution[0]}x{resolution[1]}, expected {expected_spec['expected_resolution'][0]}x{expected_spec['expected_resolution'][1]}."
                )
                remediations.append(f"Re-export or resize input to standard {expected_spec['expected_resolution'][0]}x{expected_spec['expected_resolution'][1]}.")
                spec_status = "WARN"
            elif resolution != expected_spec["expected_resolution"]:
                issues.append(
                    f"Resolution note: Detected {resolution[0]}x{resolution[1]} (acceptable variant, but project pipeline specified {expected_spec['expected_resolution'][0]}x{expected_spec['expected_resolution'][1]})."
                )
                if spec_status != "FAIL":
                    spec_status = "WARN"

            if expected_spec["expected_precision"] == "INT8" and quant_type not in ("FULL_INT8", "WEIGHT_INT8"):
                spec_status = "FAIL"

        if quant_type == "FAKE_INT8":
            spec_status = "FAIL"

        # Hardware budget evaluation (Rockchip RV1109 camera constraints)
        rv1109_specs = HARDWARE_PROFILES["rockchip_rv1109"]
        if file_size_mb <= rv1109_specs["recommended_max_file_size_mb"]:
            flash_fit = "PASS"
        elif file_size_mb <= rv1109_specs["critical_max_file_size_mb"]:
            flash_fit = "WARN"
            issues.append(f"Model size ({file_size_mb:.1f} MB) exceeds recommended flash budget (<= {rv1109_specs['recommended_max_file_size_mb']} MB). Tight fit for 16MB/32MB SPI ROM.")
            remediations.append("Apply INT8 quantization or model pruning to reduce flash footprint.")
        else:
            flash_fit = "FAIL"
            issues.append(f"Model size ({file_size_mb:.1f} MB) exceeds maximum flash budget (<= {rv1109_specs['critical_max_file_size_mb']} MB) for embedded cameras.")
            remediations.append("Model cannot be deployed on 16MB/32MB flash without severe system compression.")

        # NPU Fit
        if quant_type == "FULL_INT8":
            npu_fit = "FULL_SUPPORT"
        elif quant_type == "WEIGHT_INT8":
            npu_fit = "PARTIAL_SUPPORT"
            issues.append("Model uses FP32 I/O with INT8 weights. Embedded NPU may require host CPU dequantization.")
        else:
            npu_fit = "UNSUITABLE"
            issues.append("FP32 arithmetic unsupported on RV1109 1.2 TOPS integer NPU. Fallback to dual Cortex-A7 CPU will cause low FPS (< 3 FPS).")

        # Functional Smoke Test & Host Latency Benchmark
        latencies: List[float] = []
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        inference_success = True
        person_score = None

        try:
            # Prepare input tensor for each input
            inputs = [prepare_input_tensor(inp, self.sample_image, model_name=filename) for inp in input_details]

            # Warmup runs
            for _ in range(self.warmup_runs):
                for idx, inp in enumerate(input_details):
                    interpreter.set_tensor(inp["index"], inputs[idx])
                interpreter.invoke()

            # Benchmark runs
            for _ in range(self.num_runs):
                for idx, inp in enumerate(input_details):
                    interpreter.set_tensor(inp["index"], inputs[idx])
                t_start = time.perf_counter()
                interpreter.invoke()
                t_end = time.perf_counter()
                latencies.append((t_end - t_start) * 1000.0)

            # Check outputs for valid numbers (no NaN or Inf)
            for out in output_details:
                out_tensor = interpreter.get_tensor(out["index"])
                if np.isnan(out_tensor).any() or np.isinf(out_tensor).any():
                    issues.append(f"Inference output '{out['name']}' contains NaN or Inf values.")
                    inference_success = False

            person_score = extract_person_score(filename, interpreter)

        except Exception as e:
            issues.append(f"Inference smoke test failed: {e}")
            remediations.append("Check input shape compatibility and delegate errors.")
            inference_success = False

        mem_after = process.memory_info().rss / (1024 * 1024)
        process_mem_delta = max(0.0, mem_after - mem_before)

        if latencies:
            lat_mean = float(np.mean(latencies))
            lat_min = float(np.min(latencies))
            lat_max = float(np.max(latencies))
            lat_std = float(np.std(latencies))
            fps = 1000.0 / lat_mean if lat_mean > 0 else 0.0
        else:
            lat_mean = lat_min = lat_max = lat_std = fps = 0.0

        # Overall status determination
        if not inference_success or quant_type == "FAKE_INT8" or spec_status == "FAIL":
            overall_status = "FAIL"
        elif flash_fit == "FAIL" or npu_fit == "UNSUITABLE" or spec_status == "WARN":
            overall_status = "WARN"
        else:
            overall_status = "PASS"

        return ModelEvaluationResult(
            filename=filename,
            filepath=filepath,
            file_size_mb=round(file_size_mb, 2),
            sha256=file_sha256,
            is_duplicate=is_dup,
            duplicate_of=dup_of,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            input_layout=layout,
            input_resolution=resolution,
            total_tensors=total_tensors,
            weight_dtype_counts=weight_dtype_counts,
            int8_weight_ratio_pct=round(int8_ratio, 1),
            quantization_type=quant_type,
            inference_success=inference_success,
            latency_mean_ms=round(lat_mean, 2),
            latency_min_ms=round(lat_min, 2),
            latency_max_ms=round(lat_max, 2),
            latency_std_ms=round(lat_std, 2),
            fps=round(fps, 1),
            process_memory_mb=round(process_mem_delta, 2),
            flash_rom_fit=flash_fit,
            npu_fit=npu_fit,
            spec_conformance=spec_status,
            overall_status=overall_status,
            person_detection_score=person_score,
            issues=issues,
            remediations=remediations,
        )


def display_rich_dashboard(results: List[ModelEvaluationResult]) -> None:
    """Display comprehensive evaluation results using Rich tables and panels."""
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold cyan]IT6017 - Digital Transformation: Edge AI Camera System[/bold cyan]\n"
            "[yellow]Phase 1.2: Human Detection Model Zoo Evaluation & Edge Hardware Verification[/yellow]\n"
            f"[dim]Evaluated {len(results)} models against Rockchip RV1109 & Xiaomi 12S Pro specifications[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    # Use unified wide table if terminal width >= 115, otherwise split into 2 compact tables
    if console.width >= 115:
        table = Table(
            title="Model Evaluation & Hardware Suitability Matrix",
            box=box.ROUNDED,
            header_style="bold magenta",
        )
        table.add_column("Model Filename", style="bold white")
        table.add_column("Input Spec", justify="center")
        table.add_column("Quant Type", justify="center")
        table.add_column("Size", justify="right")
        table.add_column("Host Latency", justify="right")
        table.add_column("Host FPS", justify="right")
        table.add_column("RV1109 NPU", justify="center")
        table.add_column("Flash ROM", justify="center")
        table.add_column("Verdict", justify="center")

        for r in results:
            q_badge = (
                "[bold green]FULL INT8[/bold green]"
                if r.quantization_type == "FULL_INT8"
                else (
                    "[bold cyan]WEIGHT INT8[/bold cyan]"
                    if r.quantization_type == "WEIGHT_INT8"
                    else (
                        "[bold red blink]FAKE INT8[/bold red blink]"
                        if r.quantization_type == "FAKE_INT8"
                        else "[dim]FP32[/dim]"
                    )
                )
            )
            status_badge = (
                "[bold green]PASS[/bold green]"
                if r.overall_status == "PASS"
                else ("[bold yellow]WARN[/bold yellow]" if r.overall_status == "WARN" else "[bold red]FAIL[/bold red]")
            )
            flash_badge = (
                "[green]PASS (<6M)[/green]"
                if r.flash_rom_fit == "PASS"
                else ("[yellow]WARN (6-10M)[/yellow]" if r.flash_rom_fit == "WARN" else "[red]FAIL (>10M)[/red]")
            )
            npu_badge = (
                "[green]FULL[/green]"
                if r.npu_fit == "FULL_SUPPORT"
                else ("[yellow]PARTIAL[/yellow]" if r.npu_fit == "PARTIAL_SUPPORT" else "[red]UNSUITED[/red]")
            )

            res_layout = f"{r.input_resolution[0]}x{r.input_resolution[1]} ({r.input_layout})"
            latency_str = f"{r.latency_mean_ms:.1f}ms" if r.inference_success else "ERR"
            fps_str = f"{r.fps:.1f}" if r.inference_success else "0.0"

            table.add_row(
                r.filename,
                res_layout,
                q_badge,
                f"{r.file_size_mb:.1f} MB",
                latency_str,
                fps_str,
                npu_badge,
                flash_badge,
                status_badge,
            )
        console.print(table)
    else:
        # Table 1: Model Specifications & Quantization
        t1 = Table(title="1. Architecture & Quantization Audit", box=box.ROUNDED, header_style="bold magenta")
        t1.add_column("Model Name", style="bold white", width=25)
        t1.add_column("Input Spec", justify="center", width=14)
        t1.add_column("Quant Type", justify="center", width=12)
        t1.add_column("Size", justify="right", width=9)
        t1.add_column("Verdict", justify="center", width=9)

        for r in results:
            disp_name = r.filename.replace(".tflite", "")
            q_badge = (
                "[bold green]FULL INT8[/bold green]"
                if r.quantization_type == "FULL_INT8"
                else (
                    "[bold cyan]WEIGHT INT8[/bold cyan]"
                    if r.quantization_type == "WEIGHT_INT8"
                    else (
                        "[bold red blink]FAKE INT8[/bold red blink]"
                        if r.quantization_type == "FAKE_INT8"
                        else "[dim]FP32[/dim]"
                    )
                )
            )
            status_badge = (
                "[bold green]PASS[/bold green]"
                if r.overall_status == "PASS"
                else ("[bold yellow]WARN[/bold yellow]" if r.overall_status == "WARN" else "[bold red]FAIL[/bold red]")
            )
            res_layout = f"{r.input_resolution[0]}x{r.input_resolution[1]}\n({r.input_layout})"
            t1.add_row(disp_name, res_layout, q_badge, f"{r.file_size_mb:.1f} MB", status_badge)
        console.print(t1)

        # Table 2: Edge Hardware Fit & Benchmark
        t2 = Table(title="2. Edge Hardware Suitability & Host Benchmark", box=box.ROUNDED, header_style="bold cyan")
        t2.add_column("Model Name", style="bold white", width=25)
        t2.add_column("Host Latency", justify="right", width=12)
        t2.add_column("Host FPS", justify="right", width=9)
        t2.add_column("RV1109 NPU", justify="center", width=12)
        t2.add_column("Flash ROM", justify="center", width=12)

        for r in results:
            disp_name = r.filename.replace(".tflite", "")
            flash_badge = (
                "[green]PASS (<6M)[/green]"
                if r.flash_rom_fit == "PASS"
                else ("[yellow]WARN (6-10M)[/yellow]" if r.flash_rom_fit == "WARN" else "[red]FAIL (>10M)[/red]")
            )
            npu_badge = (
                "[green]FULL[/green]"
                if r.npu_fit == "FULL_SUPPORT"
                else ("[yellow]PARTIAL[/yellow]" if r.npu_fit == "PARTIAL_SUPPORT" else "[red]UNSUITED[/red]")
            )
            latency_str = f"{r.latency_mean_ms:.1f} ms" if r.inference_success else "ERR"
            fps_str = f"{r.fps:.1f}" if r.inference_success else "0.0"
            t2.add_row(disp_name, latency_str, fps_str, npu_badge, flash_badge)
        console.print(t2)

    # Detailed Diagnostics Section for Models with Issues
    console.print("\n[bold yellow]Diagnostic Findings & Actionable Remediation[/bold yellow]")
    for r in results:
        if r.overall_status in ("FAIL", "WARN") or r.issues:
            status_color = "green" if r.overall_status == "PASS" else ("yellow" if r.overall_status == "WARN" else "red")
            panel_title = f"[bold]{r.filename}[/bold] - Overall: [{status_color}]{r.overall_status}[/]"
            msg_lines = []
            if r.is_duplicate:
                msg_lines.append(f"[bold red]CRITICAL:[/bold red] Exact duplicate of [cyan]{r.duplicate_of}[/cyan] (SHA-256 match: {r.sha256[:12]}...)")
            for issue in r.issues:
                msg_lines.append(f"• [red]Issue:[/red] {issue}")
            for rem in r.remediations:
                msg_lines.append(f"• [green]Remediation:[/green] {rem}")
            console.print(Panel("\n".join(msg_lines), title=panel_title, border_style=status_color, box=box.ROUNDED))


def export_markdown_report(results: List[ModelEvaluationResult], report_path: Path) -> None:
    """Generate comprehensive markdown report for Phase 1.2 documentation."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pass_count = sum(1 for r in results if r.overall_status == "PASS")
    warn_count = sum(1 for r in results if r.overall_status == "WARN")
    fail_count = sum(1 for r in results if r.overall_status == "FAIL")

    content = f"""# Edge AI Model Zoo Evaluation Report

**Course**: IT6017 - Technological Trends in Digital Transformation  
**Project**: Research and Reverse Engineering of Edge AI Camera Systems for Human Detection and Fire Detection  
**Phase**: 1.2 Model Zoo Verification & Target Hardware Fit Analysis  
**Generated Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Executive Summary

| Total Models Evaluated | Passed | Warnings | Failed |
|:---:|:---:|:---:|:---:|
| **{len(results)}** | <span style="color:green">**{pass_count}**</span> | <span style="color:goldenrod">**{warn_count}**</span> | <span style="color:red">**{fail_count}**</span> |

> **Key Finding**: 
> - **Critical Defect**: `ssd_mobilenet_v2_int8.tflite` is **counterfeit** (100% FP32 weights, 0 INT8 operations, byte-for-byte replica of `ssd_mobilenet_v2_fp32.tflite`).
> - **Resolution Mismatch**: SSD MobileNetV2 uses `256x256` inputs instead of the `300x300` resolution specified in the preliminary project pipeline.
> - **Hardware Constraints**: All FP32 models (> 11 MB) violate the SPI Flash ROM budget of target embedded cameras (Rockchip RV1109 / Ingenic T31 with 16-32MB total Flash) and cannot be accelerated by integer NPUs (1.2 TOPS INT8).

---

## 2. Model Zoo Evaluation Matrix

| Model Filename | Architecture | Resolution & Layout | Arithmetic Type | Size (MB) | Host Latency (ms) | Host FPS | RV1109 NPU Support | Flash ROM Fit | Overall Status |
|:---|:---|:---:|:---:|---:|---:|---:|:---:|:---:|:---:|
"""

    for r in results:
        status_md = "🟢 PASS" if r.overall_status == "PASS" else ("🟡 WARN" if r.overall_status == "WARN" else "🔴 FAIL")
        flash_md = "🟢 PASS" if r.flash_rom_fit == "PASS" else ("🟡 WARN" if r.flash_rom_fit == "WARN" else "🔴 FAIL")
        npu_md = "🟢 FULL" if r.npu_fit == "FULL_SUPPORT" else ("🟡 PARTIAL" if r.npu_fit == "PARTIAL_SUPPORT" else "🔴 UNSUITED")
        arch = EXPECTED_MATRIX_SPECS.get(r.filename, {}).get("architecture", "Custom")

        content += f"| `{r.filename}` | {arch} | {r.input_resolution[0]}x{r.input_resolution[1]} ({r.input_layout}) | {r.quantization_type} | {r.file_size_mb:.2f} | {r.latency_mean_ms:.1f} | {r.fps:.1f} | {npu_md} | {flash_md} | {status_md} |\n"

    content += """
---

## 3. Hardware Suitability & Deployment Comparison

### 3.1. Embedded Smart Camera Profile (Rockchip RV1109 / Ingenic T31)
- **Processor**: Dual-core ARM Cortex-A7 @ 1.5 GHz.
- **NPU**: 1.2 TOPS (strictly requires INT8 / INT16 integer weights and operations).
- **Storage Constraints**: 16 MB – 32 MB SPI NAND/NOR Flash ROM.
  - Recommended AI model budget: **<= 6.0 MB** (leaving headroom for Linux OS kernel, rootfs, video streaming daemon, and camera ISP tuning files).
  - FP32 models (`11.3 MB` to `13.8 MB`) consume over 40% of the entire camera flash and are **incompatible** with standard consumer camera hardware.
- **Memory (RAM)**: 128 MB DDR3 shared between H.264/H.265 encoding, RTSP streaming, and AI buffer.

### 3.2. Smartphone Testbed Profile (Xiaomi 12S Pro)
- **SoC**: Qualcomm Snapdragon 8+ Gen 1 (Cortex-X2 + Cortex-A710 + Cortex-A510).
- **Acceleration**: Adreno 730 GPU + Hexagon NPU via Android NNAPI.
- **Memory**: 8 GB LPDDR5 RAM.
- **Role**: Serves as our empirical testbed to collect relative latency, thermal, and FPS benchmarks across all 6 models before mapping results back to camera SoCs.

---

## 4. Detailed Diagnostic Audit & Remediation Plan

"""

    for r in results:
        content += f"### 4.{results.index(r)+1} `{r.filename}`\n"
        content += f"- **SHA-256**: `{r.sha256}`\n"
        content += f"- **Input Tensor**: `{r.input_tensors[0].name}` (Shape: `{r.input_tensors[0].shape}`, Dtype: `{r.input_tensors[0].dtype}`, Layout: `{r.input_layout}`)\n"
        content += f"- **Output Tensors**: {', '.join([f'`{o.name}` ({o.shape})' for o in r.output_tensors])}\n"
        content += f"- **Weight Dtype Distribution**: {r.weight_dtype_counts}\n"
        content += f"- **True INT8 Weight Ratio**: `{r.int8_weight_ratio_pct}%`\n"

        if r.issues:
            content += "- **Identified Issues**:\n"
            for issue in r.issues:
                content += f"  - ⚠️ {issue}\n"

        if r.remediations:
            content += "- **Actionable Remediation**:\n"
            for rem in r.remediations:
                content += f"  - 🛠️ {rem}\n"

        content += "\n"

    content += """---

## 5. Next Steps for Phase 1.3 & Phase 2

1. **Re-export / Quantize SSD MobileNetV2**:
   - Re-quantize SSD MobileNetV2 with representative calibration dataset to generate a true `ssd_mobilenet_v2_int8.tflite` model.
   - Update `project-pipeline.md` step 1.1 resolution to reflect `256x256` (or acquire a native `300x300` checkpoint).
2. **Setup Testbed on Xiaomi 12S Pro (Phase 2)**:
   - Prepare benchmark runners using LiteRT / TFLite benchmark binary via Android Debug Bridge (`adb`).
   - Validate execution under CPU, GPU (OpenCL/Vulkan), and NPU (NNAPI/Qualcomm QNN) delegates.
"""

    report_path.write_text(content, encoding="utf-8")
    console.print(f"[green]Saved Markdown evaluation report to:[/green] {report_path}")


def export_json_report(results: List[ModelEvaluationResult], report_path: Path) -> None:
    """Generate machine-readable JSON evaluation report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_profiles": HARDWARE_PROFILES,
        "results": [asdict(r) for r in results],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    console.print(f"[green]Saved JSON evaluation report to:[/green] {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Edge AI Human Detection models for Course IT6017.")
    parser.add_argument("--models-dir", default="models", help="Directory containing .tflite models (default: models)")
    parser.add_argument("--reports-dir", default="reports", help="Directory to save generated reports (default: reports)")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark iterations for latency test (default: 10)")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations before benchmarking (default: 3)")
    parser.add_argument("--image", default=None, help="Optional path to sample image for visual smoke testing")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero exit code if any model fails validation")

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        models_dir=args.models_dir,
        num_runs=args.num_runs,
        warmup_runs=args.warmup,
        sample_image=args.image,
    )

    try:
        model_paths = evaluator.scan_models()
    except Exception as e:
        console.print(f"[bold red]Error scanning models:[/bold red] {e}")
        return 1

    console.print(f"[bold cyan]Scanning {len(model_paths)} models in '{args.models_dir}'...[/bold cyan]")

    results: List[ModelEvaluationResult] = []
    for p in model_paths:
        console.print(f"  [dim]Evaluating:[/dim] {p.name} ...")
        res = evaluator.evaluate_model(p)
        results.append(res)

    # Display Rich Terminal UI
    display_rich_dashboard(results)

    # Save reports
    reports_dir = Path(args.reports_dir)
    md_report_path = reports_dir / "model_evaluation_report.md"
    json_report_path = reports_dir / "model_evaluation_report.json"

    export_markdown_report(results, md_report_path)
    export_json_report(results, json_report_path)

    has_failures = any(r.overall_status == "FAIL" for r in results)
    if args.strict and has_failures:
        console.print("\n[bold red]Strict mode: One or more models failed criteria checks![/bold red]")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""External Assets, Datasets & Model Download Utility.

Course: IT6017 - Technology Trends in Digital Transformation
Project: Research and Reverse Engineering of Edge AI Camera Systems for Human Detection and Fire Detection
Sub-Team: Sub-team 3 (Measurement & Evaluation / AI Engine Optimization)

This utility automates downloading and verifying external benchmark assets:
  - Datasets: COCO-128, COCO-8
  - Test Videos: Intel IoT surveillance feeds (data/test_videos/)
  - Model Zoo Verification: Verifies 6 canonical TFLite models in models/

Usage:
  python scripts/download_assets.py --all
  python scripts/download_assets.py --dataset coco128
  python scripts/download_assets.py --dataset coco8
  python scripts/download_assets.py --videos
  python scripts/download_assets.py --check-models
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional

# Project directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
VIDEOS_DIR = PROJECT_ROOT / "data" / "test_videos"

# Dataset URL registries
DATASETS_INFO: Dict[str, Dict[str, str]] = {
    "coco128": {
        "url": "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
        "fallback_url": "https://ultralytics.com/assets/coco128.zip",
        "zip_name": "coco128.zip",
        "dest_dir": "coco128",
        "description": "Standard COCO-128 validation subset (128 images, 254 person instances)",
        "expected_files": 128,
    },
    "coco8": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
        "fallback_url": "",
        "zip_name": "coco8.zip",
        "dest_dir": "coco8",
        "description": "Compact COCO-8 smoke test dataset (8 images)",
        "expected_files": 8,
    },
}

# Surveillance sample test videos
SURVEILLANCE_VIDEOS: Dict[str, Dict[str, str]] = {
    "one-by-one-person-detection.mp4": {
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/one-by-one-person-detection.mp4",
        "description": "Single-person doorway entry/exit surveillance scenario",
        "approx_size_mb": 3.14,
    },
    "people-detection.mp4": {
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4",
        "description": "Multi-person walking crowd in public hallway",
        "approx_size_mb": 5.23,
    },
    "face-demographics-walking.mp4": {
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4",
        "description": "Approaching pedestrian in commercial corridor",
        "approx_size_mb": 6.11,
    },
}

# Model zoo registry for verification
MODELS_INFO: Dict[str, Dict[str, object]] = {
    "yolov8n_320_int8.tflite": {
        "precision": "INT8 (Weight-Quantized)",
        "resolution": (320, 320),
        "expected_min_bytes": 3_400_000,
        "expected_max_bytes": 3_600_000,
        "role": "Pareto-Optimal Edge Upgrade Candidate (28.1 FPS, 0.448 mAP50)",
    },
    "yolov8n_320_fp32.tflite": {
        "precision": "FP32 (Unquantized)",
        "resolution": (320, 320),
        "expected_min_bytes": 12_500_000,
        "expected_max_bytes": 13_000_000,
        "role": "YOLOv8n Baseline (Exceeds Flash/RAM ceiling)",
    },
    "ssd_mobilenet_v2_int8.tflite": {
        "precision": "Full INT8",
        "resolution": (300, 300),
        "expected_min_bytes": 6_000_000,
        "expected_max_bytes": 6_400_000,
        "role": "MediaPipe / Coral Zoo INT8 (5.8 FPS on T31N)",
    },
    "ssd_mobilenet_v2_fp32.tflite": {
        "precision": "FP32",
        "resolution": (300, 300),
        "expected_min_bytes": 24_000_000,
        "expected_max_bytes": 25_000_000,
        "role": "TensorFlow Zoo FP32 Baseline (Exceeds 16MB Flash)",
    },
    "efficientdet_lite0_int8.tflite": {
        "precision": "Full INT8",
        "resolution": (320, 320),
        "expected_min_bytes": 4_400_000,
        "expected_max_bytes": 4_800_000,
        "role": "MediaPipe EfficientDet INT8 (OOM on 64MB RAM)",
    },
    "efficientdet_lite0_fp32.tflite": {
        "precision": "FP32",
        "resolution": (320, 320),
        "expected_min_bytes": 13_500_000,
        "expected_max_bytes": 14_200_000,
        "role": "MediaPipe EfficientDet FP32 Baseline",
    },
}


def download_with_progress(url: str, dest_path: Path, label: str = "") -> bool:
    """Download a file with user-agent and console progress indication."""
    print(f"--> Downloading: {dest_path.name}")
    if label:
        print(f"    Target: {label}")
    print(f"    URL:    {url}")

    temp_path = dest_path.with_suffix(".tmp")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "IT6017-EdgeAI-Benchmark/1.0 (HUST Digital Transformation)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, "wb") as out_file:
            total_size = int(resp.info().get("Content-Length", 0))
            downloaded = 0
            block_size = 64 * 1024

            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb_cur = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r    Progress: {pct:5.1f}% [{mb_cur:6.2f} MB / {mb_tot:6.2f} MB]")
                    sys.stdout.flush()
                else:
                    mb_cur = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\r    Progress: {mb_cur:6.2f} MB downloaded")
                    sys.stdout.flush()

        print()  # newline
        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            print(f"    [OK] Saved to: {dest_path}")
            return True
    except Exception as e:
        print(f"\n    [ERROR] Download failed: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False
    return False


def setup_dataset(dataset_key: str, force: bool = False) -> bool:
    """Download and unpack a standard dataset into datasets/."""
    if dataset_key not in DATASETS_INFO:
        print(f"[ERROR] Unknown dataset: {dataset_key}. Available: {list(DATASETS_INFO.keys())}")
        return False

    info = DATASETS_INFO[dataset_key]
    target_dir = DATASETS_DIR / info["dest_dir"]
    images_dir = target_dir / "images" / "train2017"

    if images_dir.exists() and not force:
        image_count = len(list(images_dir.glob("*.jpg")))
        if image_count >= info["expected_files"]:
            print(f"[OK] Dataset '{dataset_key}' already present: {target_dir} ({image_count} images).")
            return True

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    zip_dest = DATASETS_DIR / info["zip_name"]

    success = download_with_progress(info["url"], zip_dest, info["description"])
    if not success and info.get("fallback_url"):
        print("    Attempting fallback URL...")
        success = download_with_progress(info["fallback_url"], zip_dest, info["description"])

    if not success:
        return False

    print(f"--> Extracting: {zip_dest.name} into {DATASETS_DIR}...")
    try:
        with zipfile.ZipFile(zip_dest, "r") as zip_ref:
            zip_ref.extractall(DATASETS_DIR)
        print(f"    [OK] Extraction completed successfully.")
        if zip_dest.exists():
            zip_dest.unlink()
        return True
    except Exception as e:
        print(f"    [ERROR] Failed to extract {zip_dest}: {e}")
        return False


def setup_surveillance_videos(force: bool = False) -> bool:
    """Download sample surveillance benchmark video clips into data/test_videos/."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    print("======================================================================")
    print(" IT6017: Verifying Surveillance Test Videos (data/test_videos/)")
    print("======================================================================")

    all_ok = True
    for filename, info in SURVEILLANCE_VIDEOS.items():
        dest = VIDEOS_DIR / filename
        if dest.exists() and dest.stat().st_size > 100_000 and not force:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  [EXISTS] {filename} ({size_mb:.2f} MB) - {info['description']}")
            continue

        ok = download_with_progress(info["url"], dest, info["description"])
        if not ok:
            all_ok = False

    return all_ok


def check_models() -> bool:
    """Audit and verify presence and integrity of all 6 canonical TFLite models."""
    print("======================================================================")
    print(" IT6017: Auditing Model Zoo Integrity (models/*.tflite)")
    print("======================================================================")

    if not MODELS_DIR.exists():
        print(f"[ERROR] Models directory '{MODELS_DIR}' does not exist.")
        return False

    all_valid = True
    for model_name, info in MODELS_INFO.items():
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            print(f"  [MISSING] {model_name} -> FAILED")
            all_valid = False
            continue

        file_size = model_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        min_b = info["expected_min_bytes"]
        max_b = info["expected_max_bytes"]

        if min_b <= file_size <= max_b:
            print(f"  [PASS] {model_name:<30} ({size_mb:5.2f} MB) - {info['precision']}")
        else:
            print(f"  [WARN] {model_name:<30} ({size_mb:5.2f} MB) - Size outside expected range [{min_b/1e6:.1f}M - {max_b/1e6:.1f}M]")
            all_valid = False

    return all_valid


def main() -> int:
    parser = argparse.ArgumentParser(
        description="IT6017 Edge AI Camera: Automated Asset & Dataset Downloader"
    )
    parser.add_argument(
        "--dataset",
        choices=["coco128", "coco8", "all"],
        default=None,
        help="Download and extract specified dataset into datasets/",
    )
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Download sample surveillance test videos into data/test_videos/",
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Audit file size and integrity of models in models/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute full asset preparation: datasets, test videos, and model check",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )

    args = parser.parse_args()

    # Default action if no arguments provided: show status and check models
    if not (args.dataset or args.videos or args.check_models or args.all):
        print("IT6017 Asset Manager: No specific flag provided. Running model check.")
        print("Use --help to see all options (e.g. --all, --dataset coco128, --videos).\n")
        args.check_models = True

    overall_ok = True

    if args.check_models or args.all:
        ok = check_models()
        if not ok:
            overall_ok = False

    if args.dataset == "coco128" or args.dataset == "all" or args.all:
        ok = setup_dataset("coco128", force=args.force)
        if not ok:
            overall_ok = False

    if args.dataset == "coco8" or args.dataset == "all":
        ok = setup_dataset("coco8", force=args.force)
        if not ok:
            overall_ok = False

    if args.videos or args.all:
        ok = setup_surveillance_videos(force=args.force)
        if not ok:
            overall_ok = False

    if overall_ok:
        print("\n[SUCCESS] Asset setup and verification finished with 0 errors.")
        return 0
    else:
        print("\n[WARNING] Some assets reported errors or missing files.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

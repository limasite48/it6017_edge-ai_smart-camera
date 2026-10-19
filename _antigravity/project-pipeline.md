`! Note: All variable, function, file, and folder names within the project must be in English and follow standard Python naming conventions`

# Phase 1: Prepare the model zoo and measurement tools. [COMPLETED]
1.1/ Model Zoo (Standardized 3x2 Experimental Matrix) [COMPLETED]
    - Sourced from official Google (MediaPipe / Coral / TF Zoo) and Ultralytics repositories via `scripts/prepare_models.py`:
    1.1.1/ SSD MobileNetV2 (300x300) FP32 - 23.24 MB (TF Object Detection Zoo 300x300 frozen graph conversion)
    1.1.2/ SSD MobileNetV2 (300x300) INT8 - 5.93 MB (Google Coral official full uint8 INT8 model)
    1.1.3/ EfficientDet-Lite0 (320x320) FP32 - 13.20 MB (Google MediaPipe official release)
    1.1.4/ EfficientDet-Lite0 (320x320) INT8 - 4.39 MB (Google MediaPipe official full INT8 release)
    1.1.5/ YOLOv8n - Nano (320x320) FP32 - 12.17 MB (Ultralytics official PyTorch export)
    1.1.6/ YOLOv8n - Nano (320x320) INT8 - 3.31 MB (Ultralytics official 99.4% INT8 export)
    - Companion Colab Notebook: `scripts/colab_export_models.ipynb` for cloud-based exports and calibration.

1.2/ Verify the model zoo meets the necessary standards for use in the project. [COMPLETED]
    - Quality gate script: `scripts/eval-model.py --strict` (0 failures, 100% test pass).
    - Evaluation reports: `reports/model_evaluation_report.md` and `reports/model_evaluation_report.json`.
    - Key audit findings:
      * All 3 INT8 models (`efficientdet_lite0_int8`, `ssd_mobilenet_v2_int8`, `yolov8n_320_int8`) achieve PASS status.
      * All 3 INT8 models are under 6.0 MB, fully compliant with the Rockchip RV1109 / Ingenic T31 16-32MB SPI Flash ROM constraint.
      * Critical defect resolved: `ssd_mobilenet_v2_int8.tflite` is verified as 100% quantized full INT8 (no longer a duplicate of FP32).
      * Resolution consistency: Both SSD MobileNetV2 models now share the exact 300x300 input resolution.
      * All 3 FP32 models verified as operational baselines for the Xiaomi 12S Pro benchmark matrix.

1.3/ Live Camera Verification on Laptop Testbed [COMPLETED]
    - Interactive real-time inspection tool: `scripts/test-live-camera.py`
    - Features: On-the-fly hot swapping (`1`–`6`), smart camera human alert banner (`! ALERT: HUMAN DETECTED`), real-time telemetry HUD (FPS, latency breakdown for preprocess/inference/postprocess), and snapshot capture (`S`).
    - Verification results on Intel Core i5-1334U CPU testbed:
      * Model 1 (SSD MobileNetV2 FP32): ~4.6 FPS (baseline)
      * Model 2 (SSD MobileNetV2 INT8): ~16.3 FPS (3.5x speedup via uint8 SIMD quantization)
      * Model 3 (EfficientDet-Lite0 FP32): ~12.8 FPS
      * Model 4 (EfficientDet-Lite0 INT8): ~16.0 FPS
      * Model 5 (YOLOv8n FP32): ~12.3 FPS
      * Model 6 (YOLOv8n INT8): ~17.2 FPS
    - Output snapshots directory: `reports/camera_inspection_snapshots/`

1.4/ Prepared Video Evaluation & Edge Camera Benchmarking [COMPLETED]
    - Model debugging & coordinate fixes:
      * Fixed EfficientDet-Lite0 person class indexing (MediaPipe class 0 = person) and [0, 255] float32 input scaling.
      * Fixed YOLOv8 box coordinate decoding (normalized [0, 1] bounds scaled directly to frame width/height).
      * Verified positive human detection across all 6 models on real snapshot test images.
    - Video dataset preparation: `scripts/prepare_test_videos.py`
      * Downloaded and verified 3 standardized surveillance clips in `data/test_videos/`: `one-by-one-person-detection.mp4`, `people-detection.mp4`, `face-demographics-walking.mp4`.
    - Batch video evaluation module: `scripts/eval-video.py`
      * Automated evaluation across all 6 models and all video clips.
      * Comprehensive telemetry: FPS, granular latency (pre/infer/post), human detection rate, confidence, and memory delta.
      * Exported 18 annotated MP4 evaluation videos to `reports/video_annotated/`.
      * Reports generated: `reports/video_evaluation_report.md` and `reports/video_evaluation_report.json`.
    - Interactive video testing: enhanced `scripts/test-live-camera.py` with `--loop` playback and `[SPACE]` pause/resume.

1.5/ Aspect-Ratio Preserving Letterbox Padding & Multi-Class Evaluation [COMPLETED]
    - Diagnostic resolution of aspect-ratio distortion:
      * Standardized letterbox padding (114 gray fill) across `scripts/test-live-camera.py` and `scripts/eval-video.py`.
      * Restored anchor geometric fidelity on non-square and extreme feeds (such as 9:16 vertical 4K street footage in `object-detection-02.mp4` and horizontal 4K in `03`).
      * Added `--stretch` flag to explicitly evaluate direct squash resizing vs letterbox padding.
    - Full COCO-90 Decoding & Anchor Calibration:
      * Aligned label maps for SSD MobileNetV2 and EfficientDet-Lite0 to official 90-class COCO IDs.
      * Refined anchor aspect ratios [(1.0, 1.0), (1.414, 0.707), (0.707, 1.414)] and variance scaling for raw BiFPN heads.
      * Implemented multi-class Non-Maximum Suppression (NMS) and `--mode {human, all}` in `eval-video.py`.
    - Extended Dataset & Dual Benchmarking:
      * Added 3 user-provided object detection clips (`object-detection-01.mp4`, `02`, `03`) to `scripts/prepare_test_videos.py`.
      * Generated comprehensive evaluation reports:
        - Human Surveillance Mode: `reports/video_evaluation_report.md`
        - General Multi-Class Object Mode: `reports/video_all_classes_evaluation_report.md`
      * Confirmed that Full INT8 models (SSD MobileNetV2 INT8 ~20ms / 46 FPS; EfficientDet INT8 ~18ms / 47 FPS; YOLOv8n INT8 ~10.5ms / 79 FPS) deliver 3x to 4x CPU speedups with minimal precision degradation.

# Phase 2: Setting up a testbed on the Xiaomi 12S Pro smartphone
# Phase 3: Real-world performance testing on smartphones (Latency, FPS, RAM, Power)
# Phase 4: Accuracy Evaluation (mAP) and Testing Conversion to Embedded NPU on a Laptop
# Phase 5: Mapping Data to Camera SoC (Rockchip RV1109 / Ingenic T31) and Data Visualization
# Phase 6: Expansion to a fire detection model (provisionally set aside; to be implemented once the project has stabilized)
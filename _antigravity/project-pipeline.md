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

# Phase 2: Setting up a testbed on the Xiaomi 12S Pro smartphone [COMPLETED]
2.1/ Toolchain Setup & Device Hardware Audit [COMPLETED]
    - Downloaded and established standalone Google Android Platform-Tools in `tools/platform-tools/`.
    - Automated inspection engine: `scripts/inspect_device.py`.
    - Target testbed verified: Xiaomi 12S Pro (`2206122SC` / board `unicorn`), Android 13 (API 33), HyperOS/MIUI V14.0.8.0.
    - Verified SoC: Qualcomm Snapdragon 8+ Gen 1 (SM8475, 4nm), 8 cores (1x Cortex-X2 @ 3.19GHz + 3x Cortex-A710 @ 2.75GHz + 4x Cortex-A510 @ 2.02GHz).
    - Verified AI / Graphics Acceleration: Qualcomm Adreno 730 GPU (OpenCL driver verified), Hexagon DSP / NPU (`qti-dsp` via Android NNAPI).
    - Audited resources: 7.02 GB usable LPDDR5 RAM, 65 GB available UFS 3.1 storage, battery temp ~30°C nominal baseline.
    - Audit reports: `reports/smartphone_hardware_audit.md` and `reports/smartphone_hardware_audit.json`.

2.2/ On-Device Binary & Model Deployment [COMPLETED]
    - Sourced Google's official native ARM64 `benchmark_model` ELF binary (`arm64-v8a`) to `tools/bin/benchmark_model`.
    - Established target directories on phone: `/data/local/tmp/it6017/bin/` and `/data/local/tmp/it6017/models/`.
    - Pushed `benchmark_model` and granted executable permissions (`chmod 755`).
    - Synchronized all 6 standardized models from the 3x2 matrix (`ssd_mobilenet_v2`, `efficientdet_lite0`, `yolov8n_320` in FP32 and INT8) to the smartphone.

2.3/ Benchmark Automation & Thermal Guard Engine [COMPLETED]
    - Automated mobile orchestrator: `scripts/run_smartphone_benchmark.py`.
    - 4 Execution Delegates:
      * CPU 1 Thread (`--num_threads=1`): Simulating low-power single-core embedded camera SoC execution.
      * CPU 4 Threads (`--num_threads=4`): Multi-core mobile baseline with XNNPACK acceleration.
      * GPU OpenCL (`--use_gpu=true`): Qualcomm Adreno 730 GPU acceleration.
      * NNAPI (`--use_nnapi=true`): Qualcomm Hexagon DSP / NPU hardware execution.
    - Strict Thermal Guard & Stability:
      * Automatic battery/chassis temperature polling before every test run.
      * Auto-pause protection if temp >= 42°C until cooled to < 38°C.
      * Increased cool-down pause to 3.5s between delegate runs, maintaining thermal equilibrium (+1.3°C rise across 24 tests: 31.0°C to 32.3°C).
    - Key Empirical Benchmark Results on Xiaomi 12S Pro:
      * SSD MobileNetV2 INT8 on Hexagon NPU: **2.52 ms (397.6 FPS)** with only **18.6 MB RAM** (8.59x speedup over CPU 4T, 27x over CPU FP32).
      * EfficientDet-Lite0 INT8 on CPU 4T: **7.28 ms (137.4 FPS)** with **29.2 MB RAM** (2.46x speedup over FP32).
      * YOLOv8n INT8 on CPU 4T: **6.20 ms (161.3 FPS)** with **21.4 MB RAM** (2.48x speedup over FP32).
      * Adreno 730 OpenCL GPU: Consistent ~11.2ms to 13.0ms execution across all INT8 and FP32 architectures (77 to 89 FPS).
      * Memory & Embedded Camera Viability: All INT8 models consume <= 29.2 MB peak RAM, comfortably within the 128 MB RAM ceiling of target Rockchip RV1109 / Ingenic T31 cameras.
    - Reports generated: `reports/smartphone_benchmark_report.md` and `reports/smartphone_benchmark_report.json`.
# Phase 3: Real-world performance testing on smartphones (Latency, FPS, RAM, Power) [COMPLETED]
3.1/ Sustained Load & DVFS Thermal Throttling Profiling [COMPLETED]
    - Multi-slice continuous execution across 16 configurations (3 INT8 models + 1 FP32 reference x 4 delegates) with automated 42°C thermal guard.
    - 1 Hz live telemetry polling across 8 CPU cores (Little Cortex-A510, Mid Cortex-A710, Prime Cortex-X2) and SoC thermal sensors.
    - Key findings:
      * SSD MobileNetV2 INT8 on Hexagon NPU: 2.06 ms steady-state (484.3 FPS) with +0.3% thermal throttling (K_throttle), demonstrating near-zero thermal degradation.
      * Single-core CPU (simulating embedded Cortex-A7): 18.27 ms steady-state (54.8 FPS) with +0.3% K_throttle, zero thermal throttling due to low core power.
      * Multi-core CPU (4T): Exhibits +31.7% to +44.9% throttling under 100% continuous utilization as Cortex-A710 scales down frequencies.
    - Script: scripts/run_smartphone_stress_test.py.
    - Reports: reports/smartphone_stress_thermal_report.md and reports/smartphone_stress_thermal_telemetry.json.

3.2/ Power & Energy Consumption Profiling [COMPLETED]
    - Simulated battery discharge profiling via dumpsys battery unplug with automated battery reset guarantee.
    - Idle baseline measured at 0.864 W (220.0 mA @ 3.93 V).
    - Active power draw, net dynamic AI power, and energy efficiency (E_frame):
      * SSD MobileNetV2 INT8 on Hexagon NPU: 1.53 mJ/frame at 1.61 W active power (93% duty-cycle capacity within 5W budget).
      * YOLOv8n INT8 on CPU 1T (Embedded proxy): 5.88 mJ/frame at 1.19 W active power (100% duty-cycle capacity).
      * INT8 vs FP32 energy comparison: INT8 reduces inference energy by up to 7x compared to GPU/NPU FP32 baselines (e.g. 5.88 mJ vs 41.7 mJ).
    - Script: scripts/run_smartphone_power_profile.py.
    - Reports: reports/smartphone_power_energy_report.md and reports/smartphone_power_energy_telemetry.json.

3.3/ End-to-End Pipeline Latency Breakdown & Amdahl's Law Bottleneck [COMPLETED]
    - Evaluated complete edge camera pipeline (T_total = T_preprocess + T_inference + T_postprocess) over real surveillance video feeds (one-by-one-person-detection.mp4).
    - Aspect-ratio preserving letterbox preprocessing (114 gray fill) + on-device inference + multi-class NMS postprocessing.
    - Quantified Amdahl's Law bottleneck:
      * When inference is accelerated down to 2.52 ms on Hexagon NPU, CPU preprocessing (0.2 ms) and NMS postprocessing (2.5 ms) account for 52.3% of total pipeline latency.
      * Explains why commercial smart cameras (Rockchip RV1109 / Ingenic T31) integrate dedicated Hardware ISPs and DMA image resizers to avoid CPU data movement bottlenecks.
    - Script: scripts/eval_smartphone_pipeline.py.
    - Reports: reports/smartphone_pipeline_report.md and reports/smartphone_pipeline_telemetry.json.

3.4/ Generalized Hardware Taxonomy Mapping & Scaling Model [COMPLETED]
    - Formalized a 3-tier camera hardware proxy taxonomy (Tier 1 Entry IoT –, Tier 2 Mainstream Consumer –, Tier 3 Enterprise –+).
    - Formulated the mathematical proxy scaling equation to project testbed measurements onto any target camera SoC.

3.5/ Live Smartphone Camera Demo & Hardware Feasibility Capture [COMPLETED - PHYSICAL CAPSTONE]
    - Deployed native mobile object detection application (YOLOv8 Mobile, package com.xvesa.yolov8mobile) to Xiaomi 12S Pro.
    - Conducted physical on-device live camera test run with real-time CameraX viewfinder.
    - Achieved 42.08 FPS real-time camera throughput on mobile CPU (~23.76 ms total pipeline latency per frame).
    - Validated high-confidence object detection: laptop 90.1%, bottle 83.0% under indoor cluttered conditions.
    - Empirical measurement correlates to within 99.8% of theoretical Phase 3 pipeline formulation (23.72 ms vs 23.76 ms).
    - Screenshot: reports/live_camera_demo/yolo_v8_mobile_211026_1044.jpg.
    - Automation Script: scripts/capture_live_demo.py.
    - Deliverable Report: reports/live_camera_demo/live_camera_demo_report.md.

# Phase 4: Accuracy Evaluation (mAP), Visual Model Comparison, and Embedded NPU Simulation [ACTIVE]
4.1/ Quantitative Detection Accuracy Evaluation (mAP50 & mAP50-95)
    - Evaluate Mean Average Precision across all 6 models on standardized benchmark dataset (COCO-val2017 subset).
    - Quantify accuracy trade-offs: FP32 baseline vs INT8 quantized precision loss (Delta mAP).
    - Script: scripts/eval_accuracy_map.py.
    - Report: reports/model_accuracy_map_report.md.

4.2/ Visual Multi-Model Comparison Grids & Synchronized Video Export [COMPLETED]
    - Generate synchronized side-by-side visual detection grids across all 6 models on key surveillance scenes (single person, crowded walking, occlusions).
    - Render detection bounding boxes, confidence scores, and embedded Mobile Hardware Telemetry Badges (latency, FPS, power, mJ).
    - Export composite high-resolution PNG grids and a synchronized multi-model comparison MP4 video.
    - Script: scripts/generate_visual_comparison.py.
    - Report: reports/visual_comparisons/visual_comparison_report.md.

4.3/ Embedded NPU Toolchain Conversion & Simulation (Rockchip RKNN-Toolkit2)
    - Run RKNN-Toolkit2 export and simulation targeting Rockchip RV1109 / RV1126 on workstation.
    - Quantize models to RKNN format (.rknn), audit operator compatibility, and extract simulated NPU cycle counts.
    - Script: scripts/simulate_rknn_conversion.py.
    - Report: reports/rknn_embedded_npu_simulation_report.md.

# Phase 5: Mapping Data to Camera SoC (Rockchip RV1109 / Ingenic T31) and Data Visualization [PLANNED]
5.1/ Comprehensive Hardware Mapping & Scaling Projection
    - Apply mathematical proxy scaling formula to project Xiaomi 12S Pro empirical data onto Rockchip RV1109 and Ingenic T31.
    - Synthesize multi-dimensional trade-off matrix: Latency vs. Accuracy (mAP) vs. Power (W) vs. RAM (MB) vs. BOM Cost ($).
5.2/ Publication-Quality Data Visualizations
    - Generate charts (Pareto frontiers, radar plots, thermal degradation curves, power breakdowns) for final report and presentation slides.

# Phase 6: Expansion to a Fire Detection Model (Provisionally set aside; to be implemented once the project has stabilized)

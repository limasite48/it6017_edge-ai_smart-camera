# IT6017: Edge AI Smart Camera - Video Benchmark & Evaluation Report

**Generated:** 2026-10-19 12:14:38
**Host Testbed:** Intel Core i5-1334U (12 vCPU), Windows 11
**Target Hardware Baseline:** Rockchip RV1109 / Ingenic T31 (1.2 TOPS INT8 NPU, 32MB Flash)
**Evaluation Mode:** Human Surveillance Detection (Class: person)
**Pre-processing Strategy:** Letterbox Padding (Aspect-ratio Preserved, Gray 114)

---

## 1. Executive Summary & Comparative Matrix

| Model Name | Precision | Video Clip | Frames | FPS | Infer Latency | Person Det Rate | Avg Conf | Annotated Video |
|:---|:---:|:---|---:|---:|---:|---:|---:|:---:|
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `face-demographics-walking.mp4` | 60 | **10.1** | 96.69 ms | **68.3%** | 95.2% | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `face-demographics-walking.mp4` | 60 | **45.2** | 20.45 ms | **68.3%** | 95.0% | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `face-demographics-walking.mp4` | 60 | **26.5** | 33.38 ms | **53.3%** | 73.4% | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `face-demographics-walking.mp4` | 60 | **47.1** | 17.74 ms | **70.0%** | 82.3% | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `face-demographics-walking.mp4` | 60 | **25.7** | 36.54 ms | **70.0%** | 88.0% | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `face-demographics-walking.mp4` | 60 | **75.4** | 10.85 ms | **70.0%** | 58.9% | None |
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `object-detection-01.mp4` | 60 | **10.2** | 91.54 ms | **6.7%** | 56.0% | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `object-detection-01.mp4` | 60 | **40.6** | 20.07 ms | **6.7%** | 61.3% | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `object-detection-01.mp4` | 60 | **25.5** | 32.67 ms | **11.7%** | 48.7% | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `object-detection-01.mp4` | 60 | **41.6** | 17.76 ms | **0.0%** | 0.0% | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `object-detection-01.mp4` | 60 | **24.7** | 35.39 ms | **0.0%** | 0.0% | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `object-detection-01.mp4` | 60 | **66.2** | 10.14 ms | **0.0%** | 0.0% | None |
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `object-detection-02.mp4` | 60 | **7.7** | 88.83 ms | **100.0%** | 71.8% | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `object-detection-02.mp4` | 60 | **18.5** | 20.6 ms | **100.0%** | 71.8% | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `object-detection-02.mp4` | 60 | **14.8** | 33.21 ms | **5.0%** | 49.6% | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `object-detection-02.mp4` | 60 | **18.7** | 18.56 ms | **100.0%** | 55.3% | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `object-detection-02.mp4` | 60 | **14.5** | 35.79 ms | **71.7%** | 57.7% | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `object-detection-02.mp4` | 60 | **23.2** | 10.99 ms | **61.7%** | 52.2% | None |
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `object-detection-03.mp4` | 60 | **8.3** | 89.29 ms | **16.7%** | 63.3% | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `object-detection-03.mp4` | 60 | **19.8** | 20.02 ms | **16.7%** | 63.7% | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `object-detection-03.mp4` | 60 | **15.1** | 33.7 ms | **0.0%** | 0.0% | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `object-detection-03.mp4` | 60 | **19.8** | 18.28 ms | **0.0%** | 0.0% | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `object-detection-03.mp4` | 60 | **15.1** | 35.67 ms | **0.0%** | 0.0% | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `object-detection-03.mp4` | 60 | **23.9** | 10.55 ms | **0.0%** | 0.0% | None |
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `one-by-one-person-detection.mp4` | 60 | **10.8** | 90.63 ms | **48.3%** | 95.5% | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `one-by-one-person-detection.mp4` | 60 | **46.2** | 20.17 ms | **48.3%** | 95.6% | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `one-by-one-person-detection.mp4` | 60 | **27.4** | 33.09 ms | **43.3%** | 64.3% | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `one-by-one-person-detection.mp4` | 60 | **47.4** | 17.91 ms | **48.3%** | 69.1% | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `one-by-one-person-detection.mp4` | 60 | **25.7** | 36.63 ms | **48.3%** | 90.3% | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `one-by-one-person-detection.mp4` | 60 | **79.5** | 10.46 ms | **48.3%** | 60.2% | None |
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `people-detection.mp4` | 60 | **10.8** | 90.42 ms | **25.0%** | 76.3% | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `people-detection.mp4` | 60 | **47.2** | 19.89 ms | **23.3%** | 78.9% | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `people-detection.mp4` | 60 | **27.6** | 32.81 ms | **16.7%** | 56.8% | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `people-detection.mp4` | 60 | **47.6** | 17.87 ms | **21.7%** | 66.8% | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `people-detection.mp4` | 60 | **26.4** | 35.86 ms | **16.7%** | 68.6% | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `people-detection.mp4` | 60 | **80.1** | 10.4 ms | **15.0%** | 59.1% | None |

---

## 2. Granular Latency Breakdown & Telemetry

| Model Name | Pre-process | Inference (Mean ± Std) | Post-process | Total Latency | Min / Max Infer |
|:---|---:|---:|---:|---:|---:|
| **SSD MobileNetV2 (300x300 FP32)** | 1.26 ms | 96.69 ± 10.87 ms | 0.1 ms | **98.05 ms** | 90.02 / 164.33 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.3 ms | 20.45 ± 1.49 ms | 0.09 ms | **20.85 ms** | 19.24 / 28.53 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 0.77 ms | 33.38 ± 1.85 ms | 2.42 ms | **36.58 ms** | 31.7 / 43.74 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.29 ms | 17.74 ± 0.44 ms | 2.11 ms | **20.13 ms** | 16.98 / 19.54 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 1.13 ms | 36.54 ± 1.17 ms | 0.2 ms | **37.87 ms** | 35.04 / 42.8 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 1.15 ms | 10.85 ± 1.27 ms | 0.18 ms | **12.19 ms** | 9.58 / 18.75 ms |
| **SSD MobileNetV2 (300x300 FP32)** | 1.29 ms | 91.54 ± 2.19 ms | 0.08 ms | **92.9 ms** | 87.34 / 96.91 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.32 ms | 20.07 ± 0.71 ms | 0.08 ms | **20.47 ms** | 18.96 / 21.62 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 0.67 ms | 32.67 ± 1.13 ms | 1.8 ms | **35.15 ms** | 31.17 / 37.26 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.38 ms | 17.76 ± 0.54 ms | 1.81 ms | **19.95 ms** | 17.03 / 20.13 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 0.98 ms | 35.39 ± 0.93 ms | 0.15 ms | **36.52 ms** | 34.28 / 39.71 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 1.01 ms | 10.14 ± 0.37 ms | 0.12 ms | **11.28 ms** | 9.51 / 11.01 ms |
| **SSD MobileNetV2 (300x300 FP32)** | 1.55 ms | 88.83 ± 1.86 ms | 0.12 ms | **90.5 ms** | 85.3 / 93.75 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.65 ms | 20.6 ± 1.55 ms | 0.12 ms | **21.37 ms** | 18.94 / 28.32 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 1.07 ms | 33.21 ± 1.56 ms | 1.64 ms | **35.92 ms** | 30.86 / 37.17 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.75 ms | 18.56 ± 1.81 ms | 1.81 ms | **21.12 ms** | 16.9 / 24.78 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 1.54 ms | 35.79 ± 1.36 ms | 0.17 ms | **37.5 ms** | 34.02 / 41.37 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 1.43 ms | 10.99 ± 1.49 ms | 0.17 ms | **12.58 ms** | 9.78 / 16.84 ms |
| **SSD MobileNetV2 (300x300 FP32)** | 1.58 ms | 89.29 ± 2.68 ms | 0.08 ms | **90.95 ms** | 85.17 / 99.87 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.68 ms | 20.02 ± 1.05 ms | 0.08 ms | **20.79 ms** | 18.83 / 24.33 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 1.11 ms | 33.7 ± 2.63 ms | 1.57 ms | **36.37 ms** | 30.78 / 43.25 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.67 ms | 18.28 ± 2.21 ms | 1.73 ms | **20.68 ms** | 16.79 / 30.65 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 1.46 ms | 35.67 ± 1.85 ms | 0.13 ms | **37.26 ms** | 33.8 / 44.56 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 1.44 ms | 10.55 ± 0.87 ms | 0.14 ms | **12.13 ms** | 9.59 / 13.22 ms |
| **SSD MobileNetV2 (300x300 FP32)** | 1.06 ms | 90.63 ± 1.39 ms | 0.09 ms | **91.77 ms** | 87.92 / 93.97 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.24 ms | 20.17 ± 0.58 ms | 0.09 ms | **20.5 ms** | 19.22 / 21.62 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 0.6 ms | 33.09 ± 0.87 ms | 1.73 ms | **35.42 ms** | 31.59 / 35.92 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.29 ms | 17.91 ± 0.55 ms | 1.73 ms | **19.93 ms** | 16.88 / 19.88 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 0.93 ms | 36.63 ± 1.76 ms | 0.16 ms | **37.72 ms** | 34.74 / 47.45 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 0.93 ms | 10.46 ± 0.5 ms | 0.17 ms | **11.56 ms** | 9.53 / 12.52 ms |
| **SSD MobileNetV2 (300x300 FP32)** | 1.1 ms | 90.42 ± 1.74 ms | 0.08 ms | **91.6 ms** | 87.14 / 96.17 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.24 ms | 19.89 ± 0.42 ms | 0.08 ms | **20.22 ms** | 19.24 / 21.25 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 0.62 ms | 32.81 ± 1.02 ms | 1.66 ms | **35.1 ms** | 31.41 / 36.36 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.29 ms | 17.87 ± 1.37 ms | 1.71 ms | **19.87 ms** | 16.96 / 26.55 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 0.92 ms | 35.86 ± 0.78 ms | 0.14 ms | **36.93 ms** | 34.63 / 38.09 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 0.92 ms | 10.4 ± 0.56 ms | 0.14 ms | **11.46 ms** | 9.58 / 13.18 ms |

---

## 3. Key Observations & Hardware Insights

1. **Full INT8 Quantization Efficiency**: SSD MobileNetV2 INT8 and EfficientDet-Lite0 INT8 demonstrate substantial CPU throughput gains over FP32 equivalents while preserving detection stability across scenes.
2. **Aspect Ratio Preservation via Letterbox Padding**: Applying letterbox padding resolves aspect ratio distortion across non-square inputs (especially 9:16 vertical 4K and horizontal 4K video feeds), enabling accurate anchor mapping for EfficientDet and YOLOv8.
3. **Multi-Class vs Dedicated Person Detection**: Dedicated person decoding optimizes post-processing compute for edge surveillance cameras, whereas multi-class decoding provides full scene context with robust multi-class NMS.

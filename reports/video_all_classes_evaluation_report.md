# IT6017: Edge AI Smart Camera - Video Benchmark & Evaluation Report

**Generated:** 2026-10-19 12:20:39
**Host Testbed:** Intel Core i5-1334U (12 vCPU), Windows 11
**Target Hardware Baseline:** Rockchip RV1109 / Ingenic T31 (1.2 TOPS INT8 NPU, 32MB Flash)
**Evaluation Mode:** General Multi-Class Object Detection (All COCO Classes)
**Pre-processing Strategy:** Letterbox Padding (Aspect-ratio Preserved, Gray 114)

---

## 1. Executive Summary & Comparative Matrix

| Model Name | Precision | Video Clip | Frames | FPS | Infer Latency | Overall Det Rate | Top Detected Classes | Annotated Video |
|:---|:---:|:---|---:|---:|---:|---:|:---|:---:|
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `face-demographics-walking.mp4` | 60 | **10.4** | 93.1 ms | **68.3%** | person (41) | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `face-demographics-walking.mp4` | 60 | **46.0** | 20.08 ms | **68.3%** | person (41) | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `face-demographics-walking.mp4` | 60 | **25.7** | 33.1 ms | **53.3%** | person (51), tv (21) | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `face-demographics-walking.mp4` | 60 | **44.3** | 17.77 ms | **70.0%** | person (67) | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `face-demographics-walking.mp4` | 60 | **25.7** | 36.3 ms | **70.0%** | person (42), tv (4) | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `face-demographics-walking.mp4` | 60 | **75.6** | 10.67 ms | **70.0%** | person (42), tv (1) | None |
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `object-detection-01.mp4` | 60 | **10.2** | 91.54 ms | **81.7%** | cat (28), cow (6), sheep (6), person (4), bird (3) | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `object-detection-01.mp4` | 60 | **40.0** | 20.56 ms | **78.3%** | cat (26), sheep (6), cow (4), bird (4), person (4) | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `object-detection-01.mp4` | 60 | **22.6** | 34.72 ms | **16.7%** | person (10), cat (3) | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `object-detection-01.mp4` | 60 | **36.3** | 18.46 ms | **86.7%** | cat (93), dog (4) | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `object-detection-01.mp4` | 60 | **23.5** | 36.25 ms | **60.0%** | bird (27), sheep (8), dog (1) | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `object-detection-01.mp4` | 60 | **58.7** | 10.83 ms | **58.3%** | bird (29), sheep (5), dog (1) | None |
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `object-detection-02.mp4` | 60 | **7.3** | 91.29 ms | **100.0%** | person (231), car (10), traffic light (3) | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `object-detection-02.mp4` | 60 | **18.1** | 21.39 ms | **100.0%** | person (231), car (10), traffic light (4) | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `object-detection-02.mp4` | 60 | **14.3** | 33.52 ms | **13.3%** | fire hydrant (7), person (3) | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `object-detection-02.mp4` | 60 | **18.1** | 18.44 ms | **100.0%** | person (97), car (27) | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `object-detection-02.mp4` | 60 | **14.4** | 36.15 ms | **73.3%** | person (68), car (16), traffic light (2), fire hydrant (1) | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `object-detection-02.mp4` | 60 | **22.1** | 11.35 ms | **61.7%** | person (52), car (16), traffic light (1) | None |
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `object-detection-03.mp4` | 60 | **8.2** | 90.7 ms | **40.0%** | train (21), person (10), car (2), bird (1) | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `object-detection-03.mp4` | 60 | **19.7** | 20.47 ms | **38.3%** | train (21), person (10), bird (1), car (1) | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `object-detection-03.mp4` | 60 | **15.1** | 32.83 ms | **0.0%** | None | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `object-detection-03.mp4` | 60 | **19.5** | 18.39 ms | **85.0%** | car (83), bus (38), truck (6), train (2) | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `object-detection-03.mp4` | 60 | **15.0** | 35.83 ms | **68.3%** | car (26), train (24), bus (6), cow (1) | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `object-detection-03.mp4` | 60 | **23.5** | 10.65 ms | **66.7%** | car (26), train (21), bus (6) | None |
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `one-by-one-person-detection.mp4` | 60 | **10.6** | 91.79 ms | **93.3%** | person (29), bench (27) | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `one-by-one-person-detection.mp4` | 60 | **47.0** | 19.98 ms | **75.0%** | person (29), bench (16) | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `one-by-one-person-detection.mp4` | 60 | **26.7** | 32.73 ms | **43.3%** | person (26) | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `one-by-one-person-detection.mp4` | 60 | **44.6** | 18.01 ms | **100.0%** | laptop (78), person (33), cat (1) | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `one-by-one-person-detection.mp4` | 60 | **26.0** | 36.17 ms | **48.3%** | person (29) | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `one-by-one-person-detection.mp4` | 60 | **78.7** | 10.37 ms | **48.3%** | person (29), tv (6) | None |
| **SSD MobileNetV2 (300x300 FP32)** | FP32 | `people-detection.mp4` | 60 | **10.7** | 91.29 ms | **25.0%** | person (15) | None |
| **SSD MobileNetV2 (300x300 Full INT8)** | FULL INT8 | `people-detection.mp4` | 60 | **46.1** | 20.21 ms | **23.3%** | person (14) | None |
| **EfficientDet-Lite0 (320x320 FP32)** | FP32 | `people-detection.mp4` | 60 | **26.3** | 33.14 ms | **18.3%** | person (10), refrigerator (1) | None |
| **EfficientDet-Lite0 (320x320 Full INT8)** | FULL INT8 | `people-detection.mp4` | 60 | **44.7** | 18.03 ms | **36.7%** | person (17), cat (7), dog (4), bed (1) | None |
| **YOLOv8n - Nano (320x320 FP32)** | FP32 | `people-detection.mp4` | 60 | **26.0** | 36.14 ms | **16.7%** | person (10) | None |
| **YOLOv8n - Nano (320x320 Weight INT8)** | WEIGHT INT8 | `people-detection.mp4` | 60 | **77.7** | 10.55 ms | **16.7%** | person (9), tv (1) | None |

---

## 2. Granular Latency Breakdown & Telemetry

| Model Name | Pre-process | Inference (Mean ± Std) | Post-process | Total Latency | Min / Max Infer |
|:---|---:|---:|---:|---:|---:|
| **SSD MobileNetV2 (300x300 FP32)** | 1.39 ms | 93.1 ± 4.26 ms | 0.1 ms | **94.59 ms** | 87.67 / 120.58 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.34 ms | 20.08 ± 0.61 ms | 0.1 ms | **20.52 ms** | 19.25 / 23.37 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 0.76 ms | 33.1 ± 1.94 ms | 3.99 ms | **37.85 ms** | 31.39 / 47.02 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.3 ms | 17.77 ± 0.5 ms | 3.4 ms | **21.46 ms** | 16.91 / 19.09 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 1.07 ms | 36.3 ± 1.35 ms | 0.38 ms | **37.76 ms** | 35.05 / 43.05 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 1.12 ms | 10.67 ± 0.68 ms | 0.36 ms | **12.14 ms** | 9.6 / 13.16 ms |
| **SSD MobileNetV2 (300x300 FP32)** | 1.28 ms | 91.54 ± 1.58 ms | 0.1 ms | **92.92 ms** | 87.29 / 95.16 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.33 ms | 20.56 ± 2.06 ms | 0.1 ms | **20.99 ms** | 18.98 / 34.8 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 0.7 ms | 34.72 ± 2.1 ms | 3.84 ms | **39.26 ms** | 31.07 / 42.61 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.35 ms | 18.46 ± 0.91 ms | 3.85 ms | **22.66 ms** | 17.15 / 20.95 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 1.11 ms | 36.25 ± 1.52 ms | 0.34 ms | **37.71 ms** | 34.51 / 42.45 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 1.15 ms | 10.83 ± 0.8 ms | 0.34 ms | **12.32 ms** | 9.92 / 13.49 ms |
| **SSD MobileNetV2 (300x300 FP32)** | 1.71 ms | 91.29 ± 4.08 ms | 0.12 ms | **93.12 ms** | 84.9 / 105.7 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.7 ms | 21.39 ± 2.5 ms | 0.13 ms | **22.22 ms** | 19.24 / 31.05 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 1.07 ms | 33.52 ± 1.83 ms | 2.87 ms | **37.47 ms** | 30.94 / 39.12 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.65 ms | 18.44 ± 1.12 ms | 3.14 ms | **22.23 ms** | 17.05 / 22.04 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 1.51 ms | 36.15 ± 1.9 ms | 0.37 ms | **38.03 ms** | 34.43 / 47.09 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 1.39 ms | 11.35 ± 2.03 ms | 0.37 ms | **13.11 ms** | 9.72 / 19.16 ms |
| **SSD MobileNetV2 (300x300 FP32)** | 1.72 ms | 90.7 ± 2.56 ms | 0.09 ms | **92.51 ms** | 86.3 / 100.98 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.67 ms | 20.47 ± 1.9 ms | 0.09 ms | **21.23 ms** | 19.01 / 27.76 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 1.02 ms | 32.83 ± 1.32 ms | 2.8 ms | **36.66 ms** | 31.07 / 37.18 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.66 ms | 18.39 ± 2.07 ms | 3.02 ms | **22.07 ms** | 16.88 / 30.04 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 1.5 ms | 35.83 ± 1.32 ms | 0.35 ms | **37.68 ms** | 34.2 / 41.39 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 1.48 ms | 10.65 ± 0.96 ms | 0.35 ms | **12.49 ms** | 9.68 / 14.46 ms |
| **SSD MobileNetV2 (300x300 FP32)** | 1.1 ms | 91.79 ± 1.65 ms | 0.1 ms | **92.99 ms** | 88.17 / 96.07 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.25 ms | 19.98 ± 0.44 ms | 0.09 ms | **20.32 ms** | 19.15 / 21.35 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 0.58 ms | 32.73 ± 1.73 ms | 3.02 ms | **36.32 ms** | 31.28 / 44.69 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.3 ms | 18.01 ± 0.55 ms | 2.99 ms | **21.3 ms** | 17.03 / 19.69 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 0.93 ms | 36.17 ± 0.62 ms | 0.35 ms | **37.45 ms** | 34.97 / 38.02 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 0.92 ms | 10.37 ± 0.47 ms | 0.32 ms | **11.62 ms** | 9.57 / 11.97 ms |
| **SSD MobileNetV2 (300x300 FP32)** | 1.09 ms | 91.29 ± 1.67 ms | 0.08 ms | **92.47 ms** | 87.35 / 95.99 ms |
| **SSD MobileNetV2 (300x300 Full INT8)** | 0.25 ms | 20.21 ± 1.17 ms | 0.09 ms | **20.55 ms** | 18.98 / 27.9 ms |
| **EfficientDet-Lite0 (320x320 FP32)** | 0.61 ms | 33.14 ± 0.91 ms | 3.08 ms | **36.84 ms** | 31.4 / 35.68 ms |
| **EfficientDet-Lite0 (320x320 Full INT8)** | 0.29 ms | 18.03 ± 0.77 ms | 2.98 ms | **21.31 ms** | 17.28 / 21.92 ms |
| **YOLOv8n - Nano (320x320 FP32)** | 0.93 ms | 36.14 ± 0.57 ms | 0.32 ms | **37.39 ms** | 35.25 / 37.99 ms |
| **YOLOv8n - Nano (320x320 Weight INT8)** | 0.94 ms | 10.55 ± 0.99 ms | 0.33 ms | **11.82 ms** | 9.65 / 17.03 ms |

---

## 3. Key Observations & Hardware Insights

1. **Full INT8 Quantization Efficiency**: SSD MobileNetV2 INT8 and EfficientDet-Lite0 INT8 demonstrate substantial CPU throughput gains over FP32 equivalents while preserving detection stability across scenes.
2. **Aspect Ratio Preservation via Letterbox Padding**: Applying letterbox padding resolves aspect ratio distortion across non-square inputs (especially 9:16 vertical 4K and horizontal 4K video feeds), enabling accurate anchor mapping for EfficientDet and YOLOv8.
3. **Multi-Class vs Dedicated Person Detection**: Dedicated person decoding optimizes post-processing compute for edge surveillance cameras, whereas multi-class decoding provides full scene context with robust multi-class NMS.

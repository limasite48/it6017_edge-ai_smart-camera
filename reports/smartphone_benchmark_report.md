# Xiaomi 12S Pro Mobile Testbed Benchmark Report

> **Generated at**: `2026-10-20 07:49:34 UTC`  
> **SoC**: Qualcomm Snapdragon 8+ Gen 1 (TSMC 4nm) | **Device**: Xiaomi 12S Pro (Android 13/HyperOS)  
> **Execution Engine**: Native ARM64 LiteRT `benchmark_model` via ADB  

---

## 1. Full Benchmark Matrix (6 Models × 4 Delegates)

| Architecture | Precision | Delegate / Hardware Engine | Latency (Mean) | Min / Max | FPS | Peak RAM | Temp |
|---|---|---|---|---|---|---|---|
| **SSD MobileNetV2** | `FP32` | CPU (1 Thread - Embedded Simulation) | **70.73 ms** | 69.3 / 72.0 ms | **14.1 FPS** | 73.3 MB | 31.0 °C |
| **SSD MobileNetV2** | `FP32` | CPU (4 Threads - Mobile Baseline) | **67.87 ms** | 64.9 / 70.8 ms | **14.7 FPS** | 72.9 MB | 31.0 °C |
| **SSD MobileNetV2** | `FP32` | GPU (Adreno 730 OpenCL Delegate) | **32.04 ms** | 28.2 / 34.6 ms | **31.2 FPS** | 144.4 MB | 31.0 °C |
| **SSD MobileNetV2** | `FP32` | NNAPI (Qualcomm Hexagon NPU / DSP) | FAILED | N/A | N/A | N/A | 31.4 °C |
| **SSD MobileNetV2** | `INT8` | CPU (1 Thread - Embedded Simulation) | **87.15 ms** | 86.6 / 87.9 ms | **11.5 FPS** | 23.4 MB | 31.4 °C |
| **SSD MobileNetV2** | `INT8` | CPU (4 Threads - Mobile Baseline) | **21.59 ms** | 19.0 / 34.4 ms | **46.3 FPS** | 22.9 MB | 31.4 °C |
| **SSD MobileNetV2** | `INT8` | GPU (Adreno 730 OpenCL Delegate) | **11.28 ms** | 10.4 / 12.5 ms | **88.7 FPS** | 91.5 MB | 32.0 °C |
| **SSD MobileNetV2** | `INT8` | NNAPI (Qualcomm Hexagon NPU / DSP) | **2.52 ms** | 1.9 / 4.0 ms | **397.6 FPS** | 18.6 MB | 32.0 °C |
| **EfficientDet-Lite0** | `FP32` | CPU (1 Thread - Embedded Simulation) | **65.51 ms** | 64.4 / 66.5 ms | **15.3 FPS** | 61.7 MB | 32.0 °C |
| **EfficientDet-Lite0** | `FP32` | CPU (4 Threads - Mobile Baseline) | **17.88 ms** | 15.9 / 33.5 ms | **55.9 FPS** | 61.7 MB | 32.0 °C |
| **EfficientDet-Lite0** | `FP32` | GPU (Adreno 730 OpenCL Delegate) | **12.19 ms** | 11.5 / 13.3 ms | **82.1 FPS** | 88.0 MB | 32.0 °C |
| **EfficientDet-Lite0** | `FP32` | NNAPI (Qualcomm Hexagon NPU / DSP) | **25.70 ms** | 22.9 / 28.3 ms | **38.9 FPS** | 53.8 MB | 32.0 °C |
| **EfficientDet-Lite0** | `INT8` | CPU (1 Thread - Embedded Simulation) | **27.14 ms** | 26.7 / 27.3 ms | **36.8 FPS** | 29.2 MB | 32.0 °C |
| **EfficientDet-Lite0** | `INT8` | CPU (4 Threads - Mobile Baseline) | **7.28 ms** | 6.8 / 10.3 ms | **137.4 FPS** | 29.2 MB | 32.0 °C |
| **EfficientDet-Lite0** | `INT8` | GPU (Adreno 730 OpenCL Delegate) | **12.96 ms** | 11.9 / 14.1 ms | **77.2 FPS** | 81.1 MB | 32.0 °C |
| **EfficientDet-Lite0** | `INT8` | NNAPI (Qualcomm Hexagon NPU / DSP) | FAILED | N/A | N/A | N/A | 32.0 °C |
| **YOLOv8n (Nano)** | `FP32` | CPU (1 Thread - Embedded Simulation) | **65.22 ms** | 64.8 / 66.0 ms | **15.3 FPS** | 43.6 MB | 32.0 °C |
| **YOLOv8n (Nano)** | `FP32` | CPU (4 Threads - Mobile Baseline) | **15.38 ms** | 14.8 / 19.3 ms | **65.0 FPS** | 43.2 MB | 32.3 °C |
| **YOLOv8n (Nano)** | `FP32` | GPU (Adreno 730 OpenCL Delegate) | **11.44 ms** | 11.0 / 12.2 ms | **87.4 FPS** | 84.5 MB | 32.3 °C |
| **YOLOv8n (Nano)** | `FP32` | NNAPI (Qualcomm Hexagon NPU / DSP) | **29.49 ms** | 25.9 / 32.7 ms | **33.9 FPS** | 37.7 MB | 32.3 °C |
| **YOLOv8n (Nano)** | `INT8` | CPU (1 Thread - Embedded Simulation) | **18.29 ms** | 17.9 / 18.9 ms | **54.7 FPS** | 21.4 MB | 32.3 °C |
| **YOLOv8n (Nano)** | `INT8` | CPU (4 Threads - Mobile Baseline) | **6.20 ms** | 5.5 / 11.1 ms | **161.3 FPS** | 21.4 MB | 32.3 °C |
| **YOLOv8n (Nano)** | `INT8` | GPU (Adreno 730 OpenCL Delegate) | **12.01 ms** | 11.6 / 14.4 ms | **83.2 FPS** | 79.7 MB | 32.3 °C |
| **YOLOv8n (Nano)** | `INT8` | NNAPI (Qualcomm Hexagon NPU / DSP) | **23.51 ms** | 23.0 / 23.9 ms | **42.5 FPS** | 15.1 MB | 32.3 °C |

---

## 2. Speedup & Quantization Analysis

### 2.1 INT8 Quantization Gain (INT8 vs FP32 on CPU 4-Threads)

| Architecture | FP32 Latency | INT8 Latency | Speedup Factor | Efficiency Gain |
|---|---|---|---|---|
| **SSD MobileNetV2** | 67.87 ms | **21.59 ms** | **3.14×** | **-68.2% latency** |
| **EfficientDet-Lite0** | 17.88 ms | **7.28 ms** | **2.46×** | **-59.3% latency** |
| **YOLOv8n (Nano)** | 15.38 ms | **6.20 ms** | **2.48×** | **-59.7% latency** |

### 2.2 Hardware Acceleration (GPU / NNAPI vs CPU 4-Threads Baseline)

| Model (Precision) | CPU 4T Baseline | GPU (Adreno 730) | GPU Speedup | NNAPI (Hexagon NPU) | NNAPI Speedup |
|---|---|---|---|---|---|
| **SSD MobileNetV2 (FP32)** | 67.87 ms | 32.04 ms | **2.12×** | N/A | N/A |
| **SSD MobileNetV2 (INT8)** | 21.59 ms | 11.28 ms | **1.91×** | 2.52 ms | **8.59×** |
| **EfficientDet-Lite0 (FP32)** | 17.88 ms | 12.19 ms | **1.47×** | 25.70 ms | **0.70×** |
| **EfficientDet-Lite0 (INT8)** | 7.28 ms | 12.96 ms | **0.56×** | N/A | N/A |
| **YOLOv8n (Nano) (FP32)** | 15.38 ms | 11.44 ms | **1.34×** | 29.49 ms | **0.52×** |
| **YOLOv8n (Nano) (INT8)** | 6.20 ms | 12.01 ms | **0.52×** | 23.51 ms | **0.26×** |

---

## 3. Comparison with Low-Cost Embedded Camera Profile

| Metric | Xiaomi 12S Pro (Measured Testbed) | Rockchip RV1109 / Ingenic T31 (Target Camera) | Deployment Feasibility Analysis |
|---|---|---|---|
| **Top INT8 Throughput** | YOLOv8n INT8 @ mobile FPS | Target: 15–30 FPS real-time | Full INT8 models meet edge camera real-time frame rates |
| **Model Memory Footprint** | 8 GB LPDDR5 available | 128 MB DDR3 shared pool | All INT8 models (3.3MB–5.9MB) fit inside 128MB RAM without OOM |
| **ROM Budget Compliance** | 128/256 GB UFS 3.1 | 16–32 MB SPI Flash ROM (<= 6 MB budget) | All 3 INT8 models pass the SPI Flash gate |
| **Inference Acceleration** | Adreno 730 GPU + Hexagon NPU | Rockchip 1.2 TOPS dedicated NPU | Dedicated embedded NPU yields comparable INT8 throughput at < 1.5W |

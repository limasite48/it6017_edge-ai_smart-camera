# Edge AI Model Zoo Evaluation Report

**Course**: IT6017 - Technological Trends in Digital Transformation  
**Project**: Research and Reverse Engineering of Edge AI Camera Systems for Human Detection and Fire Detection  
**Phase**: 1.2 Model Zoo Verification & Target Hardware Fit Analysis  
**Generated Date**: 2026-10-19 10:16:26  

---

## 1. Executive Summary

| Total Models Evaluated | Passed | Warnings | Failed |
|:---:|:---:|:---:|:---:|
| **6** | <span style="color:green">**3**</span> | <span style="color:goldenrod">**3**</span> | <span style="color:red">**0**</span> |

> **Key Finding**: 
> - **Critical Defect**: `ssd_mobilenet_v2_int8.tflite` is **counterfeit** (100% FP32 weights, 0 INT8 operations, byte-for-byte replica of `ssd_mobilenet_v2_fp32.tflite`).
> - **Resolution Mismatch**: SSD MobileNetV2 uses `256x256` inputs instead of the `300x300` resolution specified in the preliminary project pipeline.
> - **Hardware Constraints**: All FP32 models (> 11 MB) violate the SPI Flash ROM budget of target embedded cameras (Rockchip RV1109 / Ingenic T31 with 16-32MB total Flash) and cannot be accelerated by integer NPUs (1.2 TOPS INT8).

---

## 2. Model Zoo Evaluation Matrix

| Model Filename | Architecture | Resolution & Layout | Arithmetic Type | Size (MB) | Host Latency (ms) | Host FPS | RV1109 NPU Support | Flash ROM Fit | Overall Status |
|:---|:---|:---:|:---:|---:|---:|---:|:---:|:---:|:---:|
| `efficientdet_lite0_fp32.tflite` | EfficientDet-Lite0 | 320x320 (NHWC) | FP32 | 13.20 | 33.0 | 30.3 | 🔴 UNSUITED | 🔴 FAIL | 🟡 WARN |
| `efficientdet_lite0_int8.tflite` | EfficientDet-Lite0 | 320x320 (NHWC) | FULL_INT8 | 4.39 | 17.6 | 56.7 | 🟢 FULL | 🟢 PASS | 🟢 PASS |
| `ssd_mobilenet_v2_fp32.tflite` | SSD MobileNetV2 | 300x300 (NHWC) | FP32 | 23.24 | 91.1 | 11.0 | 🔴 UNSUITED | 🔴 FAIL | 🟡 WARN |
| `ssd_mobilenet_v2_int8.tflite` | SSD MobileNetV2 | 300x300 (NHWC) | FULL_INT8 | 5.93 | 19.8 | 50.4 | 🟢 FULL | 🟢 PASS | 🟢 PASS |
| `yolov8n_320_fp32.tflite` | YOLOv8n - Nano | 320x320 (NCHW) | FP32 | 12.17 | 35.5 | 28.1 | 🔴 UNSUITED | 🔴 FAIL | 🟡 WARN |
| `yolov8n_320_int8.tflite` | YOLOv8n - Nano | 320x320 (NCHW) | WEIGHT_INT8 | 3.31 | 9.9 | 101.2 | 🟡 PARTIAL | 🟢 PASS | 🟢 PASS |

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

### 4.1 `efficientdet_lite0_fp32.tflite`
- **SHA-256**: `40338edf5ec70d43e318b0a716a84d4564cd1802759a7a07170c7e43796dbf58`
- **Input Tensor**: `serving_default_images:0` (Shape: `[1, 320, 320, 3]`, Dtype: `float32`, Layout: `NHWC`)
- **Output Tensors**: `StatefulPartitionedCall:1` ([1, 19206, 90]), `StatefulPartitionedCall:0` ([1, 19206, 4])
- **Weight Dtype Distribution**: {'float32': 517, 'int32': 6}
- **True INT8 Weight Ratio**: `0.0%`
- **Identified Issues**:
  - ⚠️ Model size (13.2 MB) exceeds maximum flash budget (<= 10.0 MB) for embedded cameras.
  - ⚠️ FP32 arithmetic unsupported on RV1109 1.2 TOPS integer NPU. Fallback to dual Cortex-A7 CPU will cause low FPS (< 3 FPS).
- **Actionable Remediation**:
  - 🛠️ Model cannot be deployed on 16MB/32MB flash without severe system compression.

### 4.2 `efficientdet_lite0_int8.tflite`
- **SHA-256**: `0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb`
- **Input Tensor**: `serving_default_images:0` (Shape: `[1, 320, 320, 3]`, Dtype: `uint8`, Layout: `NHWC`)
- **Output Tensors**: `StatefulPartitionedCall:1` ([1, 19206, 90]), `StatefulPartitionedCall:0` ([1, 19206, 4])
- **Weight Dtype Distribution**: {'uint8': 1, 'int32': 188, 'int8': 406, 'float32': 2}
- **True INT8 Weight Ratio**: `99.5%`

### 4.3 `ssd_mobilenet_v2_fp32.tflite`
- **SHA-256**: `cfaf369610e14d8663ab6726a9ea131e2143ac73ae505d5e74c927d4deab7267`
- **Input Tensor**: `normalized_input_image_tensor` (Shape: `[1, 300, 300, 3]`, Dtype: `float32`, Layout: `NHWC`)
- **Output Tensors**: `TFLite_Detection_PostProcess` ([1, 10, 4]), `TFLite_Detection_PostProcess:1` ([1, 10]), `TFLite_Detection_PostProcess:2` ([1, 10]), `TFLite_Detection_PostProcess:3` ([1])
- **Weight Dtype Distribution**: {'float32': 411, 'int32': 3, 'uint8': 156}
- **True INT8 Weight Ratio**: `27.5%`
- **Identified Issues**:
  - ⚠️ Model size (23.2 MB) exceeds maximum flash budget (<= 10.0 MB) for embedded cameras.
  - ⚠️ FP32 arithmetic unsupported on RV1109 1.2 TOPS integer NPU. Fallback to dual Cortex-A7 CPU will cause low FPS (< 3 FPS).
- **Actionable Remediation**:
  - 🛠️ Model cannot be deployed on 16MB/32MB flash without severe system compression.

### 4.4 `ssd_mobilenet_v2_int8.tflite`
- **SHA-256**: `42fb3d70ffb7bb37dd518f730f7be784b831c2078f30c497d0019cc2e987fa26`
- **Input Tensor**: `normalized_input_image_tensor` (Shape: `[1, 300, 300, 3]`, Dtype: `uint8`, Layout: `NHWC`)
- **Output Tensors**: `TFLite_Detection_PostProcess` ([1, 20, 4]), `TFLite_Detection_PostProcess:1` ([1, 20]), `TFLite_Detection_PostProcess:2` ([1, 20]), `TFLite_Detection_PostProcess:3` ([1])
- **Weight Dtype Distribution**: {'uint8': 181, 'float32': 9, 'int32': 75}
- **True INT8 Weight Ratio**: `95.3%`

### 4.5 `yolov8n_320_fp32.tflite`
- **SHA-256**: `00093c7d9693e39692878fa86eae56ef0398b639bd55399032092abf4f55725f`
- **Input Tensor**: `serving_default_args_0` (Shape: `[1, 3, 320, 320]`, Dtype: `float32`, Layout: `NCHW`)
- **Output Tensors**: `serving_default_output_0_output` ([1, 84, 2100])
- **Weight Dtype Distribution**: {'float32': 389, 'int32': 26}
- **True INT8 Weight Ratio**: `0.0%`
- **Identified Issues**:
  - ⚠️ Model size (12.2 MB) exceeds maximum flash budget (<= 10.0 MB) for embedded cameras.
  - ⚠️ FP32 arithmetic unsupported on RV1109 1.2 TOPS integer NPU. Fallback to dual Cortex-A7 CPU will cause low FPS (< 3 FPS).
- **Actionable Remediation**:
  - 🛠️ Model cannot be deployed on 16MB/32MB flash without severe system compression.

### 4.6 `yolov8n_320_int8.tflite`
- **SHA-256**: `ad0102592c6d70b9db9163a4d50114424f98bf620e08bd9daa40ee44f1f0a2b8`
- **Input Tensor**: `serving_default_args_0` (Shape: `[1, 3, 320, 320]`, Dtype: `float32`, Layout: `NCHW`)
- **Output Tensors**: `serving_default_output_0_output_dequant` ([1, 84, 2100])
- **Weight Dtype Distribution**: {'float32': 2, 'int8': 356, 'int32': 90}
- **True INT8 Weight Ratio**: `99.4%`
- **Identified Issues**:
  - ⚠️ Model uses FP32 I/O with INT8 weights. Embedded NPU may require host CPU dequantization.

---

## 5. Next Steps for Phase 1.3 & Phase 2

1. **Re-export / Quantize SSD MobileNetV2**:
   - Re-quantize SSD MobileNetV2 with representative calibration dataset to generate a true `ssd_mobilenet_v2_int8.tflite` model.
   - Update `project-pipeline.md` step 1.1 resolution to reflect `256x256` (or acquire a native `300x300` checkpoint).
2. **Setup Testbed on Xiaomi 12S Pro (Phase 2)**:
   - Prepare benchmark runners using LiteRT / TFLite benchmark binary via Android Debug Bridge (`adb`).
   - Validate execution under CPU, GPU (OpenCL/Vulkan), and NPU (NNAPI/Qualcomm QNN) delegates.

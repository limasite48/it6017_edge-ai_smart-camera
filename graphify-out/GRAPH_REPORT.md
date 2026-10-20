# Graph Report - it6017_edge-ai_smart-camera  (2026-10-20)

## Corpus Check
- 24 files · ~207,275 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 27 file(s) not represented in the graph (top: .exe 8, (none) 6, .tflite 6)

## Summary
- 240 nodes · 403 edges · 14 communities (12 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `51d017a5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- LiveCameraInspector
- .evaluate_model
- eval-model.py
- Phase 1 Technical Handout (v0.0.1)
- VideoEvaluator
- prepare_models.py
- verify_video
- it6017-edge-ai-smart-camera
- execute_single_benchmark
- main
- Project Handover Briefing: Phase 2 Finalization & Technical Handout (v0.0.2)
- Hardware & Configuration Audit Report: Xiaomi 12S Pro Testbed
- Xiaomi 12S Pro Mobile Testbed Benchmark Report

## God Nodes (most connected - your core abstractions)
1. `LiveCameraInspector` - 10 edges
2. `VideoEvaluator` - 9 edges
3. `Project Handover Briefing: Phase 2 Finalization & Technical Handout (v0.0.2)` - 9 edges
4. `execute_single_benchmark()` - 8 edges
5. `main()` - 7 edges
6. `prepare_input_tensor()` - 6 edges
7. `ModelEvaluator` - 6 edges
8. `main()` - 6 edges
9. `run_adb()` - 6 edges
10. `prepare_device_environment()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Full INT8 Quantization Efficiency` --semantically_similar_to--> `SPI Flash ROM 6MB Budget Gate`  [INFERRED] [semantically similar]
  reports/model_evaluation_report.md → _antigravity/v0-0-1_handout.md
- `Repository README Document` --references--> `Project Overview Document`  [INFERRED]
  README.md → _antigravity/project-overview.md
- `SSD MobileNetV2 INT8 Defect Remediation` --conceptually_related_to--> `SSD MobileNetV2 Architecture`  [INFERRED]
  reports/model_evaluation_report.md → _antigravity/project-pipeline.md
- `Model Zoo Evaluation Report` --references--> `Strict Model Quality Gate (eval-model.py)`  [EXTRACTED]
  reports/model_evaluation_report.md → _antigravity/v0-0-1_handout.md
- `General Multi-Class Video Benchmark` --references--> `COCO Decoding and Multi-Class NMS`  [EXTRACTED]
  reports/video_all_classes_evaluation_report.md → _antigravity/project-pipeline.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Standardized Edge AI Model Architectures** — _antigravity_project_pipeline_ssd_mobilenet_v2, _antigravity_project_pipeline_efficientdet_lite0, _antigravity_project_pipeline_yolov8n, _antigravity_project_pipeline_model_zoo_matrix [EXTRACTED 1.00]
- **Video Inference Pipeline Preprocessing and Postprocessing** — _antigravity_project_pipeline_letterbox_padding, _antigravity_project_pipeline_coco_decoding_nms, reports_video_evaluation_report_human_benchmark, reports_video_all_classes_evaluation_report_multi_class_benchmark [INFERRED 0.85]
- **Embedded Camera ROM and NPU Hardware Constraints** — _antigravity_project_overview_camera_hardware_baseline, _antigravity_v0_0_1_handout_spi_flash_budget_gate, reports_model_evaluation_report_int8_quantization_advantage [INFERRED 0.95]

## Communities (14 total, 2 thin omitted)

### Community 0 - "LiveCameraInspector"
Cohesion: 0.10
Nodes (20): Detection, EfficientDetAnchorGenerator, letterbox_image(), LiveCameraInspector, main(), Any, Interpreter, ndarray (+12 more)

### Community 1 - ".evaluate_model"
Cohesion: 0.09
Nodes (25): calculate_sha256(), detect_layout_and_resolution(), display_rich_dashboard(), export_json_report(), export_markdown_report(), extract_person_score(), main(), ModelEvaluationResult (+17 more)

### Community 2 - "eval-model.py"
Cohesion: 0.16
Nodes (28): ai_edge_litert_interpreter, argparse, cv2, dataclasses, datetime, json, numpy, os (+20 more)

### Community 3 - "Phase 1 Technical Handout (v0.0.1)"
Cohesion: 0.11
Nodes (25): Embedded Smart Camera Hardware Profile (Rockchip RV1109 / Ingenic T31), Project Overview Document, Sub-team 3 Experimentation and Evaluation, Xiaomi 12S Pro Experimental Testbed, COCO Decoding and Multi-Class NMS, Project Pipeline Roadmap Document, EfficientDet-Lite0 Architecture, Aspect-Ratio Preserving Letterbox Preprocessing (+17 more)

### Community 4 - "VideoEvaluator"
Cohesion: 0.13
Nodes (17): EfficientDetAnchorGenerator, generate_markdown_report(), letterbox_image(), main(), Any, Interpreter, ndarray, Path (+9 more)

### Community 5 - "prepare_models.py"
Cohesion: 0.13
Nodes (21): hashlib, io, archive_existing_models(), calculate_sha256(), convert_ssd_mobilenet_v2_fp32(), download_file(), main(), prepare_yolov8n_models() (+13 more)

### Community 6 - "verify_video"
Cohesion: 0.33
Nodes (7): download_video_with_progress(), main(), Any, Path, Verify video can be decoded by OpenCV and extract metadata., Download video with interactive Rich progress bar or verify user-provided file., verify_video()

### Community 9 - "execute_single_benchmark"
Cohesion: 0.13
Nodes (21): enforce_thermal_guard(), execute_single_benchmark(), export_benchmark_reports(), find_adb(), get_battery_temp(), main(), parse_benchmark_stdout(), prepare_device_environment() (+13 more)

### Community 10 - "main"
Cohesion: 0.14
Nodes (17): audit_device(), check_device_status(), export_reports(), find_adb(), main(), print_rich_summary(), Any, Path (+9 more)

### Community 11 - "Project Handover Briefing: Phase 2 Finalization & Technical Handout (v0.0.2)"
Cohesion: 0.14
Nodes (13): 1. Executive Summary & Scope, 2. Testbed Hardware Audit (Xiaomi 12S Pro), 3. Toolchain & On-Device Architecture, 4. Full Benchmark Matrix Results (6 Models × 4 Delegates), 5.1 Qualcomm Hexagon NPU Apex Performance, 5.2 CPU vs. GPU Acceleration Crossover on Ultra-Lightweight Models, 5.3 Embedded Camera (Rockchip RV1109 / Ingenic T31) Viability, 5. Core Hardware, Algorithmic & Deployment Insights (+5 more)

### Community 12 - "Hardware & Configuration Audit Report: Xiaomi 12S Pro Testbed"
Cohesion: 0.25
Nodes (7): 1. Device Identification & Operating System, 2. SoC & Processing Topology, 3. GPU, NPU & AI Execution Delegates, 4. Memory, Storage & Environment Baseline, 5. Architectural Comparison: Testbed vs Target Smart Camera, CPU Cluster Frequencies:, Hardware & Configuration Audit Report: Xiaomi 12S Pro Testbed

### Community 13 - "Xiaomi 12S Pro Mobile Testbed Benchmark Report"
Cohesion: 0.29
Nodes (6): 1. Full Benchmark Matrix (6 Models × 4 Delegates), 2.1 INT8 Quantization Gain (INT8 vs FP32 on CPU 4-Threads), 2.2 Hardware Acceleration (GPU / NNAPI vs CPU 4-Threads Baseline), 2. Speedup & Quantization Analysis, 3. Comparison with Low-Cost Embedded Camera Profile, Xiaomi 12S Pro Mobile Testbed Benchmark Report

## Knowledge Gaps
- **25 isolated node(s):** `it6017-edge-ai-smart-camera`, `Sub-Team Scope Clarification`, `2. Testbed Hardware Audit (Xiaomi 12S Pro)`, `3. Toolchain & On-Device Architecture`, `4. Full Benchmark Matrix Results (6 Models × 4 Delegates)` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 100 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LiveCameraInspector` connect `LiveCameraInspector` to `eval-model.py`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `VideoEvaluator` connect `VideoEvaluator` to `eval-model.py`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Why does `ModelEvaluator` connect `.evaluate_model` to `eval-model.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **What connects `it6017-edge-ai-smart-camera`, `Sub-Team Scope Clarification`, `2. Testbed Hardware Audit (Xiaomi 12S Pro)` to the rest of the system?**
  _25 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `LiveCameraInspector` be split into smaller, more focused modules?**
  _Cohesion score 0.10098522167487685 - nodes in this community are weakly interconnected._
- **Should `.evaluate_model` be split into smaller, more focused modules?**
  _Cohesion score 0.08994708994708994 - nodes in this community are weakly interconnected._
- **Should `Phase 1 Technical Handout (v0.0.1)` be split into smaller, more focused modules?**
  _Cohesion score 0.11 - nodes in this community are weakly interconnected._
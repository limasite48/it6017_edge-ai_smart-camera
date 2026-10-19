# Graph Report - it6017_edge-ai_smart-camera  (2026-10-19)

## Corpus Check
- Corpus is ~25,187 words - fits in a single context window. You may not need a graph.

## Summary
- 167 nodes · 283 edges · 9 communities (7 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Live Camera Interactive Testbed
- Static Model Verification & Audit
- Core AI & Runtime Dependencies
- Edge AI Architecture & Handover
- Video Benchmark & Letterbox Engine
- Model Zoo Acquisition & Quantization
- Video Dataset Preparation
- Project Packaging & Metadata

## God Nodes (most connected - your core abstractions)
1. `LiveCameraInspector` - 10 edges
2. `VideoEvaluator` - 9 edges
3. `main()` - 7 edges
4. `prepare_input_tensor()` - 6 edges
5. `ModelEvaluator` - 6 edges
6. `Phase 1 Technical Handout (v0.0.1)` - 6 edges
7. `ModelEvaluationResult` - 5 edges
8. `export_markdown_report()` - 5 edges
9. `export_json_report()` - 5 edges
10. `main()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Full INT8 Quantization Efficiency` --semantically_similar_to--> `SPI Flash ROM 6MB Budget Gate`  [INFERRED] [semantically similar]
  reports/model_evaluation_report.md → _antigravity/v0-0-1_handout.md
- `Repository README Document` --references--> `Project Overview Document`  [INFERRED]
  README.md → _antigravity/project-overview.md
- `SSD MobileNetV2 INT8 Defect Remediation` --conceptually_related_to--> `SSD MobileNetV2 Architecture`  [INFERRED]
  reports/model_evaluation_report.md → _antigravity/project-pipeline.md
- `General Multi-Class Video Benchmark` --references--> `Aspect-Ratio Preserving Letterbox Preprocessing`  [EXTRACTED]
  reports/video_all_classes_evaluation_report.md → _antigravity/project-pipeline.md
- `Human Surveillance Video Benchmark` --references--> `Aspect-Ratio Preserving Letterbox Preprocessing`  [EXTRACTED]
  reports/video_evaluation_report.md → _antigravity/project-pipeline.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Standardized Edge AI Model Architectures** — _antigravity_project_pipeline_ssd_mobilenet_v2, _antigravity_project_pipeline_efficientdet_lite0, _antigravity_project_pipeline_yolov8n, _antigravity_project_pipeline_model_zoo_matrix [EXTRACTED 1.00]
- **Embedded Camera ROM and NPU Hardware Constraints** — _antigravity_project_overview_camera_hardware_baseline, _antigravity_v0_0_1_handout_spi_flash_budget_gate, reports_model_evaluation_report_int8_quantization_advantage [INFERRED 0.95]
- **Video Inference Pipeline Preprocessing and Postprocessing** — _antigravity_project_pipeline_letterbox_padding, _antigravity_project_pipeline_coco_decoding_nms, reports_video_evaluation_report_human_benchmark, reports_video_all_classes_evaluation_report_multi_class_benchmark [INFERRED 0.85]

## Communities (9 total, 2 thin omitted)

### Community 0 - "Live Camera Interactive Testbed"
Cohesion: 0.10
Nodes (20): Detection, EfficientDetAnchorGenerator, letterbox_image(), LiveCameraInspector, main(), Any, Interpreter, ndarray (+12 more)

### Community 1 - "Static Model Verification & Audit"
Cohesion: 0.09
Nodes (25): calculate_sha256(), detect_layout_and_resolution(), display_rich_dashboard(), export_json_report(), export_markdown_report(), extract_person_score(), main(), ModelEvaluationResult (+17 more)

### Community 2 - "Core AI & Runtime Dependencies"
Cohesion: 0.17
Nodes (22): ai_edge_litert_interpreter, argparse, cv2, dataclasses, json, numpy, pathlib, psutil (+14 more)

### Community 3 - "Edge AI Architecture & Handover"
Cohesion: 0.11
Nodes (25): Embedded Smart Camera Hardware Profile (Rockchip RV1109 / Ingenic T31), Project Overview Document, Sub-team 3 Experimentation and Evaluation, Xiaomi 12S Pro Experimental Testbed, COCO Decoding and Multi-Class NMS, Project Pipeline Roadmap Document, EfficientDet-Lite0 Architecture, Aspect-Ratio Preserving Letterbox Preprocessing (+17 more)

### Community 4 - "Video Benchmark & Letterbox Engine"
Cohesion: 0.13
Nodes (17): EfficientDetAnchorGenerator, generate_markdown_report(), letterbox_image(), main(), Any, Interpreter, ndarray, Path (+9 more)

### Community 5 - "Model Zoo Acquisition & Quantization"
Cohesion: 0.12
Nodes (23): hashlib, io, os, archive_existing_models(), calculate_sha256(), convert_ssd_mobilenet_v2_fp32(), download_file(), main() (+15 more)

### Community 6 - "Video Dataset Preparation"
Cohesion: 0.33
Nodes (7): download_video_with_progress(), main(), Any, Path, Verify video can be decoded by OpenCV and extract metadata., Download video with interactive Rich progress bar or verify user-provided file., verify_video()

## Knowledge Gaps
- **6 isolated node(s):** `it6017-edge-ai-smart-camera`, `Repository README Document`, `EfficientDet-Lite0 Architecture`, `YOLOv8n Nano Architecture`, `Video Multi-Class Evaluation Report` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 61 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LiveCameraInspector` connect `Live Camera Interactive Testbed` to `Core AI & Runtime Dependencies`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `VideoEvaluator` connect `Video Benchmark & Letterbox Engine` to `Core AI & Runtime Dependencies`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Why does `ModelEvaluator` connect `Static Model Verification & Audit` to `Core AI & Runtime Dependencies`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **What connects `it6017-edge-ai-smart-camera`, `Repository README Document`, `EfficientDet-Lite0 Architecture` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Live Camera Interactive Testbed` be split into smaller, more focused modules?**
  _Cohesion score 0.10098522167487685 - nodes in this community are weakly interconnected._
- **Should `Static Model Verification & Audit` be split into smaller, more focused modules?**
  _Cohesion score 0.08994708994708994 - nodes in this community are weakly interconnected._
- **Should `Edge AI Architecture & Handover` be split into smaller, more focused modules?**
  _Cohesion score 0.11 - nodes in this community are weakly interconnected._
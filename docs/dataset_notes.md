# Dataset Notes

## Overview

| Dataset | Role | Format | Platforms | Use in This Project |
|---------|------|--------|-----------|-------------------|
| Multimodal-Mind2Web | Train + Eval | Screenshots + HTML + actions | Web | Primary training and generalization benchmark |
| ScreenSpot-v2 | Eval | Screenshot + instruction → bbox | Desktop, Mobile, Web | Primary clean grounding test set |
| VisualWebBench | Supplementary Eval | Web page screenshots + tasks | Web | Broader transfer analysis |
| ScreenSpot-Pro | Optional Hard Eval | High-res professional GUIs | Desktop (professional) | Stress test on dense interfaces |

## Multimodal-Mind2Web

**Reference**: Deng et al., "Mind2Web: Towards a Generalist Agent for the Web", NeurIPS 2023.

**HuggingFace ID**: `osunlp/Multimodal-Mind2Web`

**Splits**: `train`, `test_task`, `test_website`, `test_domain`

**Statistics**:
- Train: ~7,775 actions from ~1,009 tasks
- test_task: Same websites, different tasks
- test_website: Different websites within same domains
- test_domain: Entirely new domains

**Role in this project**: Main supervised training data (Stage A) and primary generalization benchmark.

### Raw HF Fields → Unified Schema Mapping

Each row represents a **single action step** in a multi-step web task.

| HF Raw Field | Type | → GroundingSample Field | Notes |
|---|---|---|---|
| `action_uid` | `str` | `sample_id` (as `mind2web_{split}_{action_uid}`) | Unique per action |
| `confirmed_task` | `str` | `instruction` | Task-level instruction (same for all steps in a task) |
| `operation` | JSON `str` | `action_type` | Parsed: `{"op":"CLICK"}` → `ActionType.CLICK` |
| `operation.value` | `str` | `metadata["typed_value"]` | Non-empty for TYPE actions |
| `pos_candidates[0].attributes.bounding_box_rect` | `str` | `target_bbox` | **XYWH → XYXY conversion** |
| *(computed)* | | `click_point` | Center of `target_bbox` |
| `pos_candidates[0].backend_node_id` | `str` | `target_element_id` | DOM node ID |
| `screenshot` | PIL Image | `image_path` (saved to disk) | JPEG, variable size (e.g. 1280×5429) |
| `website` | `str` | `website` | e.g. "united" |
| `domain` | `str` | `domain` | e.g. "Travel" |
| `subdomain` | `str` | `metadata["subdomain"]` | e.g. "Airlines" |
| `pos_candidates` | `list[str]` | `dom_candidates` (positive part) | JSON strings with bbox |
| `neg_candidates` | `list[str]` | `dom_candidates` (negative part, capped) | Up to `max_candidates` total |
| `annotation_id` | `str` | `metadata["annotation_id"]` | Session ID |
| `target_action_index` | `str` | `metadata["target_action_index"]` | Step index in multi-step task |
| `target_action_reprs` | `str` | `metadata["target_action_reprs"]` | e.g. "[heading] CAR -> CLICK" |
| `action_reprs` | `list[str]` | *(not stored)* | All steps in the task |
| `cleaned_html` | `str` | *(not stored)* | Too large for per-sample storage |
| `raw_html` | `str` | *(not stored)* | Too large |
| *(hardcoded)* | | `platform` = `"web"` | Always web for Mind2Web |

### Coordinate System

- **Raw format**: `bounding_box_rect` = `"x, y, width, height"` (XYWH, absolute pixels)
- **Converted to**: `BBox(x1, y1, x2, y2)` where `x2 = x + width`, `y2 = y + height`
- All coordinates are in absolute pixel space relative to the screenshot
- Screenshots are variable size (width typically 1280, height can be very tall for scrolled pages)

### Operation Type Mapping

| Raw `op` | → `ActionType` |
|----------|---------------|
| `CLICK` | `click` |
| `TYPE` | `type` |
| `SELECT` | `select` |
| `HOVER` | `hover` |
| Other | `None` (logged as warning) |

### Field Availability

| Field | Availability | Notes |
|-------|-------------|-------|
| `instruction` | Always present | |
| `action_type` | Always present (CLICK/TYPE/SELECT are common) | |
| `target_bbox` | Present when `pos_candidates` has valid bbox | Very high coverage |
| `click_point` | Derived from `target_bbox` center | Available when bbox is available |
| `target_element_id` | Present when `pos_candidates` exist | `backend_node_id` |
| `screenshot` | Always present as PIL Image | |
| `website` | Always present | |
| `domain` | Always present | |
| `ocr_text` | **Not available** | Not in Mind2Web |
| `dom_candidates` | Typically 1 positive + hundreds of negatives | Capped to `max_candidates` |

### Known Issues / Caveats

1. **Screenshots can be very tall** (5000+ pixels) — these are full-page captures, not viewport-only
2. **`confirmed_task` is the same for all actions in a task** — it's the task-level instruction, not a per-step instruction
3. **`click_point` is derived** (bbox center), not annotated — real click might be elsewhere in the element
4. **`pos_candidates` usually has exactly 1 element** — occasionally there may be multiple
5. **`neg_candidates` can have 500+ elements** — we cap to `max_candidates` (default 32)

## ScreenSpot-v2

**Reference**: ScreenSpot-v2 dataset release, 2025 (update to SeeClick ScreenSpot).

**What it provides**:
- Instruction-grounding pairs with bounding box annotations
- Covers desktop, mobile, and web interfaces
- Corrected annotation quality compared to ScreenSpot v1

**Role in this project**: Primary held-out evaluation benchmark. Tests grounding quality on a clean, well-annotated set that spans multiple platforms.

**Why v2?** The original ScreenSpot had known annotation issues. v2 provides corrected labels.

## VisualWebBench

**Reference**: Wang et al., "VisualWebBench: How Far Have Multimodal LLMs Evolved in Web Page Understanding and Grounding?", 2024.

**What it provides**:
- Multiple web understanding subtasks
- Relevant subtasks: element grounding, action grounding, action prediction
- Also includes: element OCR, webpage understanding (less relevant here)

**Role in this project**: Supplementary evaluation to test whether grounding improvements also transfer to broader web understanding tasks.

**Subtasks to use**:
- Element grounding — directly relevant
- Action grounding — directly relevant
- Action prediction — relevant

## ScreenSpot-Pro

**Reference**: Li et al., "ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use", 2025.

**What it provides**:
- High-resolution screenshots from professional software (IDEs, CAD tools, etc.)
- Dense, complex interfaces with many small elements
- A harder stress test for grounding models

**Role in this project**: Optional hard evaluation. Only run if compute and time permit. Expected to show larger performance drops, especially for visual-only models.

## Data Directory Convention

```
data/
├── raw/                    # Original downloaded data
│   ├── mind2web/
│   │   ├── train.jsonl
│   │   ├── test_task.jsonl
│   │   ├── test_website.jsonl
│   │   ├── test_domain.jsonl
│   │   └── screenshots/
│   ├── screenspot_v2/
│   │   ├── test.jsonl
│   │   └── images/
│   ├── visualwebbench/
│   └── screenspot_pro/
├── interim/                # Intermediate processing artifacts
├── processed/              # Model-ready data (tokenized, cached)
└── manifests/              # Split manifests, sample lists
```

## Data Preparation Checklist

- [ ] Download Multimodal-Mind2Web via HuggingFace
- [ ] Convert to canonical JSONL format
- [ ] Verify screenshot paths and bbox annotations
- [ ] Download ScreenSpot-v2
- [ ] Verify bbox format (x1, y1, x2, y2 vs. other)
- [ ] (Optional) Download VisualWebBench relevant subsets
- [ ] (Optional) Download ScreenSpot-Pro
- [ ] Create sample manifests for quick iteration

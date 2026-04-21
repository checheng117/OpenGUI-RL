# Data and Benchmark Access

OpenGUI-RL does not redistribute benchmark datasets, raw screenshots, cached benchmark records, or model checkpoints. Users should obtain each dataset from its official source and follow its license and terms.

## Benchmarks Used

| Benchmark | Role in the project | Public release status |
| --- | --- | --- |
| Mind2Web | Main training and split-wise generalization benchmark for candidate-aware GUI grounding. | Not redistributed. Users must obtain access from the official source. |
| ScreenSpot-v2 | Primary held-out GUI grounding benchmark for point-native and dual-path transfer inference. | Not redistributed. Configs can load from the upstream Hugging Face dataset when available. |
| VisualWebBench | Supplementary transfer benchmark for protocol-sensitive transfer analysis. | Not redistributed. Configs can load from the upstream dataset when available. |
| ScreenSpot-Pro | Mentioned in the proposal as an optional stress test. | Not completed in the final reported scope. |

## Expected Local Layout

The repository keeps only placeholder directories under `data/`:

```text
data/
├── raw/          # user-provided original benchmark payloads
├── interim/      # temporary conversion products
├── processed/    # model-ready local records and screenshots
└── manifests/    # lightweight local manifests
```

Large files in these directories are ignored by git. The tracked `.gitkeep` files only preserve the directory structure.

## Minimal Record Schema

A GUI grounding example is represented conceptually as:

```json
{
  "sample_id": "example_0001",
  "screenshot_path": "path/to/screenshot.png",
  "instruction": "Click the Submit button.",
  "candidates": [
    {
      "candidate_slot": 0,
      "text": "Submit",
      "bbox": [120, 340, 220, 380],
      "source": "ocr_or_dom"
    }
  ],
  "target": {
    "candidate_slot": 0,
    "click_point": [170, 360],
    "bbox": [120, 340, 220, 380],
    "action_type": "click"
  }
}
```

See `data_examples/sample_gui_grounding_record.jsonl` for a tiny synthetic example that documents the expected interface without using benchmark data.

## OCR / DOM Cues

OCR and DOM are external structured observations, not separate trained encoders inside the model. The candidate-aware setting serializes these cues as compact text / metadata context for the VLM policy:

```text
screenshot + instruction + OCR/DOM candidate list -> Qwen2.5-VL + LoRA -> structured GUI action
```

This distinction matters for transfer: candidate-aware grounding works when the benchmark exposes semantically meaningful candidates, but it does not transfer unchanged to anonymous option-box protocols.

## What Is Publicly Releasable Here

The public repository includes:

- code and configs,
- public documentation and summary artifacts,
- small summary figures and tables,
- synthetic examples,
- documentation for expected dataset interfaces.

The public repository excludes:

- raw benchmark screenshots,
- processed benchmark records,
- cached benchmark prediction dumps,
- model checkpoints and LoRA weights,
- local training logs and experiment caches.

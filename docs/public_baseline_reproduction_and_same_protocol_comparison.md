# Public Baseline Reproduction and Same-Protocol Comparison

## Main Conclusion

- Reproduced public baseline: plain public `Qwen/Qwen2.5-VL-3B-Instruct`
- Same-protocol repo comparison result: **our current refined method is below the reproduced plain public baseline on ScreenSpot-v2 point accuracy**
- Absolute point-accuracy gap: `0.7099 - 0.7563 = -0.0464`
- Honest interpretation: the large jump from the old repo baseline to the current refined result is **mainly coordinate-frame bug-fix recovery**, not a demonstrated win over a strong public plain-Qwen baseline

## What Changed

- Added a minimal public-baseline adapter at `src/gui_grounding/models/qwen2_vl_public_point_baseline.py`
- Added adapter selection support to `scripts/run_eval_screenspot_v2.py`
- Added a dedicated baseline config at `configs/eval/screenspot_v2_public_qwen2_5_vl_3b_point_baseline.yaml`
- Ran the full `1272 / 1272` ScreenSpot-v2 test evaluation for the reproduced public baseline
- Saved direct comparison artifacts against the current refined run:
  - `outputs/screenspot_v2_public_qwen2_5_vl_3b_point_baseline/predictions.jsonl`
  - `outputs/screenspot_v2_public_qwen2_5_vl_3b_point_baseline/evaluation_summary.json`
  - `outputs/screenspot_v2_public_qwen2_5_vl_3b_point_baseline/subgroup_metrics.json`
  - `outputs/screenspot_v2_public_qwen2_5_vl_3b_point_baseline/comparison_vs_ours.json`
  - `outputs/screenspot_v2_public_qwen2_5_vl_3b_point_baseline/comparison_vs_ours.md`

## Protocol Comparability Audit

### Repo Metric Definition

Grounded in `scripts/run_eval_screenspot_v2.py`:

- `point_accuracy`: predicted `click_point` lies inside the ground-truth bbox
- `iou@0.5`: predicted bbox IoU with ground-truth bbox is at least `0.5`
- `mean_iou`: mean bbox IoU
- `action_type_valid_rate`: predicted action type is one of the repo-valid labels
- `parseable_output_rate`: structured payload parsed successfully

Important implication:

- The repo headline metric for ScreenSpot-v2 is a **point-hit success metric**
- The repo IoU metrics are **additional** and are **not** what the public ScreenSpot-v2 summary tables report

### Public Metric Definition

Grounded in the public `GUI-Actor` ScreenSpot-v2 evaluation code and model card:

- Public ScreenSpot-v2 evaluation computes `hit_top1` by checking whether the predicted point lies inside the target bbox
- That is the closest public metric to the repo’s `point_accuracy`
- The public evaluation code also computes an overlap-style box proxy, but the well-known `80.9 / 91.0 / 92.4` numbers are public benchmark summary numbers, not repo IoU numbers

Therefore:

- **Repo `point_accuracy` is substantially comparable to the public ScreenSpot-v2 point-hit metric**
- **Repo `iou@0.5` and `mean_iou` are not directly comparable to the public `80.9 / 91.0 / 92.4` table**

### Dataset / Packaging Mismatch Note

- Repo evaluation dataset: `lscpku/ScreenSpot-v2`
- Public GUI-Actor evaluation dataset: `HongxinLi/ScreenSpot_v2`
- Both expose a `1272`-sample `test` split, but they are different Hugging Face repos and use different bbox packaging conventions
- The repo dataset adapter converts absolute `xywh` annotations to absolute `xyxy`
- The public GUI-Actor evaluation normalizes bbox coordinates to `[0, 1]`

Practical consequence:

- The comparison is **same-protocol inside this repo**
- It is **not** a claim that we exactly replicated the public model-card number end-to-end with the exact same data packaging and official eval script

### Coordinate-Frame Audit

This turned out to be the core comparability issue.

- A naive original-frame public-point smoke run produced `0 / 10` point hits
- Inspecting the raw outputs showed that plain Qwen was producing plausible points, but in the resized model-view coordinate frame rather than the original screenshot frame
- The reproduced public baseline therefore uses an **explicit** `model_resized` coordinate interpretation in the baseline adapter, then maps predictions back to the original screenshot before scoring

This was not applied silently:

- The baseline config explicitly sets `coordinate_frame: model_resized`
- The parsed payload records include `_resolved_coordinate_frame: model_resized`

Interpretation:

- Without correcting the coordinate interpretation, plain Qwen is artificially under-scored
- This strongly supports the conclusion that the repo’s earlier `0.0849` result was dominated by coordinate mismatch

## Reproduced Public Baseline

### Target

- Backbone: `Qwen/Qwen2.5-VL-3B-Instruct`
- Adapter: plain public point-style baseline
- Prompt style: Qwen-style `point_2d` JSON
- Coordinate interpretation for fair scoring: `model_resized`
- No reranker
- No verifier
- No DPO / GRPO / surrogate optimization
- No GUI-Actor fine-tuning

### Full Baseline Result Under Repo Protocol

Artifact: `outputs/screenspot_v2_public_qwen2_5_vl_3b_point_baseline/evaluation_summary.json`

| Metric | Reproduced Public Baseline |
| --- | ---: |
| Evaluated samples | 1272 |
| Point accuracy | 0.7563 |
| IoU@0.5 | 0.0519 |
| Mean IoU | 0.1327 |
| Action type validity | 0.2980 |
| Parseable output rate | 0.9937 |
| Valid bbox rate | 0.9772 |
| Valid click point rate | 0.9772 |

Action-type note:

- The low strict `action_type_valid_rate` is mostly a parser-canonicalization artifact
- The plain baseline emitted `left_click` `881` times, which the repo’s strict validity metric counts as invalid
- If obvious click synonyms are canonicalized, the baseline action-type validity is approximately `0.9906`
- I did **not** overwrite the saved same-protocol metric; the official comparison below uses the repo’s strict metric as saved

### Public Reference Check

- Public reference target mentioned in the task: approximately `80.9`
- Reproduced same-protocol repo point accuracy: `75.63`
- Gap versus public reference: `-5.27` points

Interpretation:

- This reproduction is in the same **range** as the public reference, but it is **not an exact replication** of the public model-card score
- Remaining difference is plausibly due to dataset packaging differences, prompt/runtime differences, processor version differences, and the fact that this reproduction runs inside the repo evaluator rather than the public GUI-Actor script

## Direct Comparison Against Our Current Refined Method

### Our Current Refined Method

Artifact: `outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/evaluation_summary.json`

| Metric | Our Refined Method |
| --- | ---: |
| Evaluated samples | 1272 |
| Point accuracy | 0.7099 |
| IoU@0.5 | 0.1682 |
| Mean IoU | 0.2404 |
| Action type validity | 1.0000 |
| Parseable output rate | 1.0000 |
| Valid bbox rate | 1.0000 |
| Valid click point rate | 1.0000 |

### Same-Protocol Overall Comparison

Artifact: `outputs/screenspot_v2_public_qwen2_5_vl_3b_point_baseline/comparison_vs_ours.json`

| Metric | Public Baseline | Ours | Delta (Ours - Baseline) |
| --- | ---: | ---: | ---: |
| Evaluated samples | 1272 | 1272 | 0 |
| Point accuracy | 0.7563 | 0.7099 | -0.0464 |
| IoU@0.5 | 0.0519 | 0.1682 | +0.1164 |
| Mean IoU | 0.1327 | 0.2404 | +0.1077 |
| Action type validity | 0.2980 | 1.0000 | +0.7020 |
| Parseable output rate | 0.9937 | 1.0000 | +0.0063 |

### Subgroup Point Accuracy Comparison

| Group | Public Baseline | Ours | Delta (Ours - Baseline) |
| --- | ---: | ---: | ---: |
| desktop | 0.7275 | 0.6976 | -0.0299 |
| mobile | 0.7944 | 0.7445 | -0.0499 |
| web | 0.7346 | 0.6796 | -0.0549 |
| icon | 0.6300 | 0.5614 | -0.0686 |
| text | 0.8538 | 0.8245 | -0.0292 |

Observed pattern:

- The reproduced plain public baseline is **higher than ours on point accuracy in every reported platform subgroup**
- Our refined method is **higher on IoU metrics**, which is expected because it is explicitly optimized to emit structured bbox proposals rather than only a point-style answer

## Honest Interpretation

### Did We Beat the Reproduced Public Baseline?

No.

- Under the repo’s same evaluation protocol, our current refined method is **below** the reproduced plain public Qwen baseline on the most public-facing comparable metric, `point_accuracy`
- Absolute gap: `-0.0464`

### Is the Big Gain Mainly Bug-Fix Recovery?

Mostly yes.

Comparison to the old repo baseline:

| Run | Point Accuracy |
| --- | ---: |
| Old repo baseline (`outputs/screenspot_v2_eval_qwen2_5_vl_3b`) | 0.0849 |
| Reproduced public plain-Qwen baseline | 0.7563 |
| Our refined method | 0.7099 |

Key implication:

- The move from `0.0849` to `0.7099` is real and important, but the reproduced plain public baseline at `0.7563` shows that most of that recovery comes from fixing coordinate interpretation and getting plain Qwen back into the correct performance band

### Does the Current Refined Method Still Contribute Something Meaningful?

Yes, but not as a demonstrated public-baseline win on ScreenSpot-v2 point accuracy.

Meaningful remaining contribution:

- stable structured outputs
- perfect parseability on the saved refined run
- perfect strict action-type validity on the saved refined run
- materially stronger bbox metrics than the plain point-style public baseline

What it does **not** support:

- a claim that the current refined method is above a strong public plain-Qwen baseline on ScreenSpot-v2
- any superiority claim over public strong baselines like `GUI-Actor-3B` or `GUI-Actor-3B + Verifier`

## Final Answer

- Public baseline reproduced: plain `Qwen/Qwen2.5-VL-3B-Instruct`
- Protocol fully comparable: **not fully**; `point_accuracy` is meaningfully comparable to the public ScreenSpot-v2 hit metric, but public model-card numbers are not identical to the repo protocol because dataset packaging, bbox format, and prompt/runtime differ
- Reproduced public baseline metrics: `1272` samples, point accuracy `0.7563`, IoU@0.5 `0.0519`, mean IoU `0.1327`, strict action-type validity `0.2980`, parseable output rate `0.9937`
- Our current refined metrics under the same repo protocol: `1272` samples, point accuracy `0.7099`, IoU@0.5 `0.1682`, mean IoU `0.2404`, action-type validity `1.0000`, parseable output rate `1.0000`
- Are we above the reproduced public baseline: **no**
- Best honest interpretation: the current result is a strong recovery from the repo’s earlier coordinate-frame mismatch and produces better structured bbox outputs, but it does **not** currently beat the reproduced plain public Qwen baseline on the public-facing ScreenSpot-v2 point metric

## Public Sources Used

- `GUI-Actor` model card: `https://huggingface.co/microsoft/GUI-Actor-3B-Qwen2.5-VL`
- `GUI-Actor` ScreenSpot-v2 eval code: `https://github.com/microsoft/GUI-Actor` and `https://raw.githubusercontent.com/microsoft/GUI-Actor/main/eval/screenSpot_v2.py`
- Public ScreenSpot-v2 dataset packaging used by GUI-Actor: `https://huggingface.co/datasets/HongxinLi/ScreenSpot_v2`
- Repo dataset packaging used here: `https://huggingface.co/datasets/lscpku/ScreenSpot-v2`

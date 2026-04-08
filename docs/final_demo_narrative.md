# Lightweight Demo Narrative

This project does not package a new interactive demo system. Instead, it reuses existing qualitative artifacts that are already saved in the repository and are cheap to show during a final talk or screen recording.

## Suggested 2-Minute Demo Flow

1. Start with the problem statement:
   - input = screenshot + natural-language instruction
   - output = structured GUI action with `bbox_proposal`, `click_point`, and `action_type`
2. Show one saved qualitative example where the model highlights a likely target region on a real webpage screenshot.
3. Show a second example with a different page layout to reinforce cross-website behavior.
4. Transition immediately to the benchmark takeaway:
   - the final project is not an anecdotal demo claim
   - the main evidence is the full ScreenSpot-v2 rerun after coordinate-frame refinement

## Representative Example 1

- Artifact:
  - `outputs/qwen_multisample_validation_qwen2_5/visualizations/mind2web_train_6c7a7082-2897-41c7-9688-4b0f3d778cdb.png`
- Instruction:
  - `rent a car in Brooklyn - Central, NY on from April 9 to April 15.`
- Parsed action payload:
  - `action_type: type`
  - `predicted_element_id: From*`
  - `predicted_click_point: [163, 357]`
  - `predicted_bbox: [100, 357, 330, 384]`
- Why it is useful:
  - shows the Qwen-first path producing a structured action on a realistic travel website screenshot

## Representative Example 2

- Artifact:
  - `outputs/qwen_multisample_validation_qwen2_5/visualizations/mind2web_train_c7548fe6-29eb-4ffb-a431-24ad7f535f5c.png`
- Instruction:
  - `Find the address and store hours for the Armageddon Shop record store in Boston.`
- Parsed action payload:
  - `action_type: type`
  - `predicted_element_id: search-bar`
  - `predicted_click_point: [357, 3009]`
  - `predicted_bbox: [180, 3009, 450, 3038]`
- Why it is useful:
  - shows a very different page geometry and still makes the structured-output contract visible

## Optional Runtime-Unblock Artifact

- Artifact:
  - `outputs/single_inference_qwen_unblock/unblock_qwen2_5_20260401_161151.png`
- Use:
  - only if you want one slide showing that real Qwen single-sample inference was working before the held-out benchmark phase

## Demo Framing Guidance

- Do not present the qualitative examples as the main proof.
- Use them only to help the audience understand the action format.
- Anchor the real claim in the final ScreenSpot-v2 numbers:
  - point accuracy `0.0849 -> 0.7099`
  - held-out samples `1272/1272`

"""Gradio demo for GUI grounding visualization.

Upload a screenshot and type an instruction to see the predicted
bounding box and click point overlaid on the image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from gui_grounding.utils.logger import get_logger
from gui_grounding.utils.visualization import draw_prediction

logger = get_logger(__name__)

DEMO_DESCRIPTION = """\
## Cross-Website GUI Grounding Demo

Upload a webpage screenshot, type a natural-language instruction, and see
the model's predicted bounding box and click point.

> **Note**: This is currently running in scaffold mode with dummy predictions.
> Connect a real model checkpoint to get actual results.
"""


def predict_grounding(
    image: Optional[Image.Image],
    instruction: str,
) -> Optional[Image.Image]:
    """Run grounding prediction and return annotated image.

    TODO(stage-2): Replace dummy prediction with real model inference.
    """
    if image is None:
        return None

    if not instruction.strip():
        return image

    logger.info("Demo prediction for instruction: '%s'", instruction)

    w, h = image.size
    dummy_bbox = (w * 0.3, h * 0.3, w * 0.7, h * 0.7)
    dummy_click = (w * 0.5, h * 0.5)

    result_image = draw_prediction(
        image,
        pred_bbox=dummy_bbox,
        pred_point=dummy_click,
        action_type="click",
    )

    return result_image


def build_demo():
    """Construct and return the Gradio demo interface."""
    try:
        import gradio as gr
    except ImportError:
        logger.error("Gradio is not installed. Run: pip install gradio")
        raise

    with gr.Blocks(title="GUI Grounding Demo") as demo:
        gr.Markdown(DEMO_DESCRIPTION)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Screenshot")
                instruction_input = gr.Textbox(
                    label="Instruction",
                    placeholder='e.g., "Click the login button"',
                )
                run_btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Prediction")

        run_btn.click(
            fn=predict_grounding,
            inputs=[input_image, instruction_input],
            outputs=output_image,
        )

    return demo


def launch_demo(share: bool = False, port: int = 7860) -> None:
    """Build and launch the Gradio demo."""
    demo = build_demo()
    demo.launch(share=share, server_port=port)


if __name__ == "__main__":
    launch_demo()

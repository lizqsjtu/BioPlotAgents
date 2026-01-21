# src/deepagents_plot_agent.py
from typing import Dict, Any, List
import os

from .graph_plot import build_plot_graph
from .tools_langchain import BioDataPreviewTool


class DeepAgentsPlotPipelineAgent:
    """
    Top-level plotting agent.

    It wraps the LangGraph workflow and adds an outer agentic loop:
    - Multiple attempts (iterations)
    - Uses execution feedback (errors, missing images) to adjust prompts
      and encourage alternative visualization types / mappings.
    """

    def __init__(self):
        os.makedirs("./outputs/plots", exist_ok=True)
        self.graph_app = build_plot_graph()

    def run(
        self,
        user_task: str,
        data_path: str,
        data_metadata: Dict[str, Any],
        example_images: List[str],
        figure_caption: str,
        user_style_desc: str = "Please imitate the layout and color style of the reference figure.",
        backend: str = "matplotlib",
        max_attempts: int = 3,
        desired_plot_types: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Agentic outer loop.
        """

        preview_tool = BioDataPreviewTool()
        data_info = preview_tool._run(data_path=data_path)

        meta: Dict[str, Any] = dict(data_metadata) if isinstance(data_metadata, dict) else {}
        meta["schema"] = data_info.get("schema", [])

        last_state: Dict[str, Any] = {}
        previous_attempts: List[Dict[str, Any]] = []
        desired_types = desired_plot_types or []

        for attempt in range(1, max_attempts + 1):
            print(f"[AGENT] Attempt {attempt} / {max_attempts}")

            init_state: Dict[str, Any] = {
                "user_task": user_task,
                "data_path": data_path,
                "data_preview": data_info["preview"],
                "data_metadata": meta,
                "example_images": example_images,
                "figure_caption": figure_caption,
                "user_style_desc": user_style_desc,
                "backend": backend,
                "previous_attempts": previous_attempts,
                "desired_plot_types": desired_types,
            }

            state = self.graph_app.invoke(init_state)
            last_state = state

            generated_images = state.get("generated_images", []) or []
            exec_result = state.get("exec_result", {}) or {}

            attempt_record = {
                "attempt": attempt,
                "generated_images": generated_images,
                "exec_success": exec_result.get("success", None),
                "exec_error": exec_result.get("error", ""),
                "quality_eval": state.get("quality_eval", ""),
            }
            previous_attempts.append(attempt_record)

            if generated_images:
                print(f"[AGENT] Success on attempt {attempt}, images generated.")
                break
            else:
                print(f"[AGENT] No images on attempt {attempt}, will try to adjust if attempts remain.")

        last_state["attempt_history"] = previous_attempts
        return last_state

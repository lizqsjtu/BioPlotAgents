# src/run_direct.py
import argparse
import json
import os
from typing import Dict, Any, List

from .deepagents_plot_agent import DeepAgentsPlotPipelineAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct run of multi-agent plotting pipeline.")
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Path to reference figure (PNG).",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file (CSV/TSV).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Maximum number of agent attempts.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="Page index (reserved for PDF mode, currently unused).",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="generic",
        help="Optional data type metadata (e.g. generic / rna-seq-de / scrna / variant).",
    )
    parser.add_argument(
        "--plot_types",
        type=str,
        default="",
        help="Optional comma-separated list of preferred plot types "
             "(e.g. scatterplot,heatmap,boxplot).",
    )
    return parser.parse_args()


def build_data_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "data_type": args.data_type,
    }


def main() -> None:
    args = parse_args()

    outputs_dir = "/workspace/outputs" if os.path.exists("/workspace") else "./outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[INFO] Using outputs directory: {outputs_dir}")
    print(f"[INFO] Reference: {args.ref}")
    print(f"[INFO] Data: {args.data}")
    print(f"[INFO] Iters: {args.iters}, Page: {args.page}")
    print(f"[INFO] User task: Generate a scientific plot consistent with the reference style.")
    print(f"[INFO] Data type: {args.data_type}")

    desired_plot_types: List[str] = []
    if args.plot_types.strip():
        desired_plot_types = [t.strip() for t in args.plot_types.split(",") if t.strip()]

    agent = DeepAgentsPlotPipelineAgent()

    user_task = "Please generate a scientific figure consistent with the reference style."
    data_metadata = build_data_metadata(args)
    example_images = [args.ref]
    figure_caption = ""  # could be extended to read from a sidecar file
    user_style_desc = "Please imitate the layout and overall visual style of the reference."

    print("[INFO] Running plotting pipeline ...")
    result = agent.run(
        user_task=user_task,
        data_path=args.data,
        data_metadata=data_metadata,
        example_images=example_images,
        figure_caption=figure_caption,
        user_style_desc=user_style_desc,
        backend="matplotlib",
        max_attempts=args.iters,
        desired_plot_types=desired_plot_types,
    )

    best_spec_path = os.path.join(outputs_dir, "best_spec.json")
    best_spec_obj: Dict[str, Any] = {
        "layout_spec": result.get("layout_spec", ""),
        "bio_plot_mapping": result.get("bio_plot_mapping", ""),
        "generated_images": result.get("generated_images", []),
        "user_task": user_task,
        "data_path": args.data,
        "ref_path": args.ref,
        "iters": args.iters,
        "page": args.page,
        "quality_eval": result.get("quality_eval", ""),
        "exec_result": result.get("exec_result", {}),
        "attempt_history": result.get("attempt_history", []),
        "generated_code": result.get("generated_code", ""),
    }

    with open(best_spec_path, "w", encoding="utf-8") as f:
        json.dump(best_spec_obj, f, indent=2)
    print(f"[INFO] Saved best_spec to: {best_spec_path}")

    if not result.get("generated_images"):
        print(f"[WARN] No images were detected in {plots_dir}. "
              "Check exec_result.error and generated_code in best_spec.json for issues.")
    else:
        print(f"[INFO] Generated images: {result.get('generated_images')}")

    print("[INFO] You can now run:")
    print(f"  ls -lh {outputs_dir}")
    print(f"  cat {best_spec_path}")


if __name__ == "__main__":
    main()

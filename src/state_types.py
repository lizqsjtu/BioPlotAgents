# src/state_types.py
from typing import TypedDict, List, Dict, Any


class PlotState(TypedDict, total=False):
    # ===== Inputs =====
    # Natural language description of the user task
    user_task: str
    # Path to the data file (CSV/TSV)
    data_path: str
    # Data preview (first few rows as CSV string)
    data_preview: str
    # Metadata, including schema and optional data_type, etc.
    data_metadata: Dict[str, Any]
    # List of example image paths (PNG/JPG)
    example_images: List[str]
    # Figure caption or page text for the reference figure
    figure_caption: str
    # User description of desired style
    user_style_desc: str
    # Plotting backend, e.g. "matplotlib"
    backend: str

    # Optional user preferences for plot types, e.g. ["heatmap", "boxplot"]
    desired_plot_types: List[str]

    # History of previous attempts (for agentic adjustment)
    # Each element may contain attempt index, errors, quality eval, etc.
    previous_attempts: List[Dict[str, Any]]

    # ===== Intermediate state =====
    # Layout/style specification as JSON-like string
    layout_spec: str
    # Semantic plot mapping as JSON string
    bio_plot_mapping: str
    # Generated Python plotting code
    generated_code: str

    # ===== Outputs =====
    # List of generated image file paths
    generated_images: List[str]
    # Result of code execution (success flag, stdout, error, work_dir)
    exec_result: Dict[str, Any]
    # LLM-based quality evaluation of the result
    quality_eval: str
    # History of all attempts made by the outer agent loop
    attempt_history: List[Dict[str, Any]]

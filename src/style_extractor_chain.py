# src/style_extractor_chain.py
from typing import Dict, Any
import json

from .llm_langchain import get_local_qwen_model


def _summarize_image_stats(image_stats: Dict[str, Any]) -> str:
    """
    Compress image_stats into a short natural language description
    to avoid sending a huge dict to the LLM.
    """
    if not image_stats:
        return "No image statistics were provided."

    width = image_stats.get("width")
    height = image_stats.get("height")
    rows = image_stats.get("approx_rows")
    cols = image_stats.get("approx_cols")
    colors = image_stats.get("color_clusters_hex", [])[:3]
    has_bar = image_stats.get("maybe_has_colorbar", False)

    parts = []
    if width and height:
        parts.append(f"The image resolution is approximately {width}x{height} pixels.")
    if rows and cols:
        parts.append(f"The estimated subplot layout is about {rows} rows by {cols} columns.")
    if colors:
        parts.append("Representative colors (up to 3) are: " + ", ".join(colors) + ".")
    parts.append(f"Right-side colorbar detected (approximate): {has_bar}.")

    return "\n".join(parts)


def _build_style_prompt(
    image_stats: Dict[str, Any],
    figure_caption: str,
    user_style_desc: str = "",
) -> str:
    """
    Build a text prompt for the style/layout extraction agent.
    """
    image_stats_text = _summarize_image_stats(image_stats)

    prompt = (
        "You are an expert in scientific figure layout and style.\n"
        "You will infer a reusable layout and style configuration (as JSON) based on:\n"
        "- Basic image statistics (size, approximate panel grid, representative colors)\n"
        "- The figure caption or surrounding text\n"
        "- The user's informal style description\n\n"
        "IMAGE STATS (summary):\n"
        f"{image_stats_text}\n\n"
        "FIGURE CAPTION / PAGE TEXT (truncated to 1000 characters):\n"
        f"{figure_caption[:1000]}\n\n"
        "USER STYLE DESCRIPTION (truncated to 500 characters):\n"
        f"{user_style_desc[:500]}\n\n"
        "Please infer:\n"
        "1. Whether the figure is multi-panel, and approximate grid (rows, cols).\n"
        "2. The panel types at a high level (scatter, boxplot, violin, bar, heatmap, etc.).\n"
        "3. Global style hints: font family, font sizes, line widths, marker sizes, color palette.\n"
        "4. Legend placement and orientation.\n"
        "5. Whether there are annotations like significance stars or reference lines.\n\n"
        "IMPORTANT:\n"
        "- For this task, limit the layout to at most 4 panels arranged on a 2x2 grid.\n"
        "- Use grid_pos values only from [[0,0], [0,1], [1,0], [1,1]].\n\n"
        "Output a single JSON object with the following keys:\n"
        "- plot_type: string\n"
        "- canvas: {\"width_px\": int, \"height_px\": int, \"dpi\": int}\n"
        "- panels: list of objects with {\"id\", \"type\", \"grid_pos\", \"notes\"}\n"
        "- color: {\"mode\", \"palette_name\", \"n_colors\", \"example_colors_hex\"}\n"
        "- legend: {\"position\", \"orientation\", \"title_present\"}\n"
        "- annotation: {\"has_significance_stars\", \"has_reference_lines\", \"others\"}\n"
        "- style_constraints: {\"font_family\", \"font_size_main\", \"line_width\", \"marker_size\"}\n\n"
        "Return ONLY JSON. Do not wrap it in Markdown fences or add extra text."
    )
    return prompt


def _truncate_layout_to_2x2(layout_spec_text: str) -> str:
    """
    Try to parse a layout_spec JSON string and ensure:
    - At most 4 panels
    - grid_pos is mapped to a 2x2 grid
    If parsing fails, return the original text.
    """
    try:
        obj = json.loads(layout_spec_text)
    except Exception:
        return layout_spec_text

    panels = obj.get("panels", [])
    if not isinstance(panels, list):
        return layout_spec_text

    # Keep at most 4 panels
    panels = panels[:4]
    # Reassign grid positions to a 2x2 grid
    grid_positions = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for idx, p in enumerate(panels):
        if idx < len(grid_positions):
            p["grid_pos"] = grid_positions[idx]
    obj["panels"] = panels

    try:
        return json.dumps(obj, indent=2)
    except Exception:
        return layout_spec_text


def extract_style(
    image_stats: Dict[str, Any],
    figure_caption: str,
    user_style_desc: str = "",
) -> str:
    """
    Build the style prompt and call the local vLLM via ChatOpenAI.

    Returns a JSON-like string, truncated to a 2x2 layout.
    """
    llm = get_local_qwen_model()
    prompt = _build_style_prompt(
        image_stats=image_stats,
        figure_caption=figure_caption,
        user_style_desc=user_style_desc,
    )

    resp = llm.invoke(prompt)
    if hasattr(resp, "content"):
        raw = resp.content
    else:
        raw = str(resp)

    # Try to ensure max 4 panels, 2x2 grid
    truncated = _truncate_layout_to_2x2(raw)
    return truncated

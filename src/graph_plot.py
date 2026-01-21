# src/graph_plot.py
from typing import Dict, Any, List
from glob import glob
import os
import re

from langgraph.graph import StateGraph, END

from .state_types import PlotState
from .llm_langchain import get_local_qwen_model
from .image_style_tools import ImageBasicStatsTool
from .style_extractor_chain import extract_style
from .tools_langchain import ExecuteCodeTool

# Initialize local vLLM wrapped as LangChain ChatOpenAI
llm = get_local_qwen_model()


def _sanitize_generated_code(raw: str) -> str:
    """
    Clean LLM-generated code string and inject saving logic:

    - Remove Markdown code fences.
    - Remove obvious explanation lines (e.g. "Here is the code:").
    - Remove non-comment lines that contain Chinese characters.
    - Ensure there is a main() function and a guard.
    - Append a small wrapper to save combined_panels.png and panel_1..4.png
      *inside* main(), after the user's original plotting code.
    """

    if not raw:
        return raw

    # 1) Strip Markdown fences and explanation lines, remove non-comment Chinese
    lines = raw.splitlines()
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        lower = stripped.lower()
        if lower.startswith("here is the code"):
            continue
        if any("\u4e00" <= ch <= "\u9fff" for ch in stripped) and not stripped.startswith("#"):
            continue
        cleaned.append(line)
    code = "\n".join(cleaned)

    # 2) Ensure we have imports for os and matplotlib.pyplot as plt
    if "import os" not in code:
        code = "import os\n" + code
    if "import matplotlib.pyplot as plt" not in code:
        code = "import matplotlib.pyplot as plt\n" + code

    # 3) Ensure there is a main() function and a guard at the end
    if "def main(" not in code:
        # If model forgot main, create a trivial one that at least creates a figure
        stub = (
            "\n\ndef main():\n"
            "    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=96)\n"
            "    os.makedirs('plots', exist_ok=True)\n"
            "    fig.savefig(os.path.join('plots', 'combined_panels.png'))\n"
        )
        code = code.rstrip() + stub

    if "if __name__" not in code:
        code = code.rstrip() + "\n\nif __name__ == '__main__':\n    main()\n"

    # 4) Inject per-panel saving code at the end of main(), AFTER user's logic
    #
    # We will:
    # - locate 'def main('
    # - locate its end by searching for 'if __name__ =='
    # - insert our block just before the guard
    main_start = code.find("def main(")
    if main_start == -1:
        # Fallback: nothing more we can safely do
        return code

    guard_index = code.find("if __name__ == '__main__':")
    if guard_index == -1:
        guard_index = len(code)

    main_block = code[main_start:guard_index]
    before_main = code[:main_start]
    after_main = code[guard_index:]

    # Our injection assumes:
    # - the user's main() has already created 'fig' and 'axes'
    # - if not, the try/except will swallow errors without breaking combined save
    injection = """
    # Auto-injected saving logic for combined and individual panels
    try:
        os.makedirs('plots', exist_ok=True)
        try:
            fig.tight_layout()
        except Exception:
            pass
        # Save combined figure (if not already saved)
        try:
            fig.savefig(os.path.join('plots', 'combined_panels.png'))
        except Exception:
            pass
        # Save each panel as a separate image
        try:
            panel_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for idx, (r, c) in enumerate(panel_indices, start=1):
                panel_path = os.path.join('plots', f'panel_{idx}.png')
                try:
                    extent = axes[r, c].get_window_extent().transformed(
                        fig.dpi_scale_trans.inverted()
                    )
                    fig.savefig(panel_path, bbox_inches=extent)
                except Exception:
                    # Ignore errors for individual panels
                    pass
        except Exception:
            pass
    except Exception:
        # Do not crash if fig/axes are missing
        pass
"""

    # Insert our injection just before the end of main()
    # We simply append to the main block
    main_block = main_block.rstrip() + injection + "\n"

    # Reassemble full code
    code = before_main + main_block + after_main
    return code


def _strip_markdown_fence(text: str) -> str:
    """
    Remove leading and trailing Markdown code fences from a text block,
    often used around JSON like ```json ... ```.
    """
    if not text:
        return text
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


# ===== Node 1: Style extraction (Layout / Style Agent) =====

def style_extract_node(state: PlotState) -> PlotState:
    """
    Extract basic image statistics from a reference example figure,
    then call LLM to generate a layout/style specification (layout_spec).
    """

    example_images = state.get("example_images", [])
    if not example_images:
        image_stats = {
            "image_path": None,
            "width": None,
            "height": None,
            "approx_rows": 1,
            "approx_cols": 1,
            "color_clusters_hex": [],
            "maybe_has_colorbar": False,
        }
    else:
        img_path = example_images[0]
        stats_tool = ImageBasicStatsTool()
        image_stats = stats_tool._run(image_path=img_path)

    figure_caption = state.get("figure_caption", "")
    user_style_desc = state.get("user_style_desc", "")

    layout_spec_raw = extract_style(
        image_stats=image_stats,
        figure_caption=figure_caption,
        user_style_desc=user_style_desc,
    )
    layout_spec_clean = _strip_markdown_fence(layout_spec_raw)
    state["layout_spec"] = layout_spec_clean
    return state


# ===== Node 2: Domain/Visualization mapping Agent (seaborn-aware) =====

def bio_domain_node(state: PlotState) -> PlotState:
    """
    Neutral visualization domain agent with seaborn awareness.
    """

    data_preview = state.get("data_preview", "")
    data_meta = state.get("data_metadata", {}) or {}
    layout_spec = state.get("layout_spec", "")
    task = state.get("user_task", "")
    previous_attempts = state.get("previous_attempts", []) or []
    desired_plot_types = state.get("desired_plot_types", []) or []

    schema = data_meta.get("schema", [])
    columns = [c.get("name") for c in schema]
    cols_str = ", ".join(columns) if columns else "(unknown columns)"

    prompt = (
        "You are a general-purpose scientific data visualization expert, familiar with seaborn's example gallery.\n"
        "You will design an appropriate set of plots for a 2x2 multi-panel layout using seaborn-style plot types.\n\n"
        "DATA INFORMATION:\n"
        f"- Available columns (you MUST only use these, do NOT invent new names): {cols_str}\n"
        f"- Column schema (JSON): {schema}\n"
        "DATA PREVIEW (first rows as CSV):\n"
        f"{data_preview}\n\n"
        "LAYOUT SPEC (may include JSON-like content):\n"
        f"{layout_spec[:2000]}\n\n"
        "USER TASK:\n"
        f"{task}\n\n"
        "USER PREFERRED PLOT TYPES (if any, prioritize these if possible):\n"
        f"{desired_plot_types}\n\n"
        "PREVIOUS ATTEMPTS (execution errors and issues; avoid repeating the same mistakes):\n"
        f"{previous_attempts}\n\n"
        "Seaborn plot families you know (examples from seaborn example gallery):\n"
        "- Relational: scatterplot, lineplot\n"
        "- Distribution: histplot, kdeplot, ecdfplot\n"
        "- Categorical: boxplot, violinplot, stripplot, swarmplot, barplot, pointplot\n"
        "- Regression: regplot\n"
        "- Matrix: heatmap\n\n"
        "Instructions:\n"
        "1. Do NOT assume any specific scientific domain. Treat this as generic tabular data.\n"
        "2. Identify numeric columns vs categorical columns, based on schema and preview.\n"
        "3. For each of the up to 4 panels in the layout, choose a seaborn-style plot type that is: "
        "(a) compatible with the data, (b) consistent with the original layout idea, and (c) if possible, matches user preferred types.\n"
        "4. Consider previous_attempts: if there was a column name error, avoid using that column again. "
        "If a certain plot type clearly failed, consider an alternative type from the same family or another reasonable family.\n"
        "5. For each panel, specify:\n"
        "   - id: matching the layout panels (e.g. panel_1, panel_2, ...).\n"
        "   - chosen_type: one seaborn-style plot type name, e.g. 'scatterplot', 'boxplot', 'violinplot', 'heatmap', etc.\n"
        "   - x: a valid column name or null.\n"
        "   - y: a valid column name or null.\n"
        "   - hue: optional grouping column, or null.\n"
        "   - facet_by: optional column name for splitting data into sub-panels, or null.\n"
        "   - agg: if needed (for bar/heatmap), e.g. 'count' or 'mean'.\n"
        "   - notes: short justification.\n"
        "6. Use only column names from the provided list; do NOT invent new ones.\n\n"
        "Output JSON ONLY, with this structure:\n"
        "{\n"
        "  \"panels\": [\n"
        "    {\n"
        "      \"id\": \"panel_1\",\n"
        "      \"chosen_type\": \"scatterplot\",\n"
        "      \"x\": \"<col_or_null>\",\n"
        "      \"y\": \"<col_or_null>\",\n"
        "      \"hue\": \"<col_or_null>\",\n"
        "      \"facet_by\": \"<col_or_null>\",\n"
        "      \"agg\": \"<agg_or_null>\",\n"
        "      \"notes\": \"...\"\n"
        "    },\n"
        "    ...\n"
        "  ],\n"
        "  \"global_recommendations\": {\n"
        "     \"numeric_columns\": [...],\n"
        "     \"categorical_columns\": [...],\n"
        "     \"suggested_layout_intent\": \"...\",\n"
        "     \"seaborn_families_used\": [...]\n"
        "  }\n"
        "}\n"
        "No extra text."
    )

    resp = llm.invoke(prompt)
    content = resp.content if hasattr(resp, "content") else str(resp)
    state["bio_plot_mapping"] = content
    return state


# ===== Node 3: Code generation Agent =====

def code_node(state: PlotState) -> PlotState:
    """
    Code Agent: generate Python plotting script using seaborn/matplotlib.
    """

    layout_spec = state.get("layout_spec", "")
    bio_map = state.get("bio_plot_mapping", "")
    data_path = state.get("data_path", "")

    data_meta = state.get("data_metadata", {}) or {}
    schema = data_meta.get("schema", [])
    columns = [c.get("name") for c in schema]
    cols_str = ", ".join(columns)

    previous_attempts = state.get("previous_attempts", []) or []

    prompt = (
        "You are a Python data visualization engineer familiar with seaborn's example gallery. "
        "Generate a PURE Python script (no Markdown, no Chinese text) that creates a 2x2 grid of plots "
        "based on a generic tabular dataset.\n\n"
        f"Data file path: '{data_path}'\n\n"
        "Available columns in the dataset (you MUST only use these, do NOT invent new names):\n"
        f"{cols_str}\n\n"
        "Layout specification (JSON-like string, at most 4 panels):\n"
        f"{layout_spec[:1500]}\n\n"
        "Bio plot mapping JSON (panel definitions with chosen_type, x, y, hue, facet_by, agg, notes):\n"
        f"{bio_map[:2000]}\n\n"
        "Previous attempts summary (execution errors etc. Avoid repeating same mistakes):\n"
        f"{previous_attempts}\n\n"
        "Script requirements:\n"
        "1. Use ONLY Python code as plain text. Do NOT include Markdown or any Chinese text.\n"
        "2. Use short English comments starting with '#'.\n"
        "3. Import os, pandas as pd, seaborn as sns, and matplotlib.pyplot as plt.\n"
        "4. Read the CSV/TSV into a DataFrame named 'df' using: pd.read_csv(file_path, sep=None, engine=\"python\").\n"
        "5. Create a figure with a 2x2 grid: fig, axes = plt.subplots(2, 2, figsize=(1421/96, 1132/96), dpi=96).\n"
        "   Only use axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]. Do NOT index beyond these.\n"
        "6. Use seaborn functions according to chosen_type, e.g. scatterplot, boxplot, violinplot, heatmap.\n"
        "7. For axis titles, use ONLY simple ASCII strings like 'Panel 1', 'Panel 2', etc. "
        "Do NOT chain calls on the result of set_title (just call set_title once).\n"
        "8. At the end of main(), call fig.tight_layout() and then save the combined figure as:\n"
        "   os.makedirs('plots', exist_ok=True)\n"
        "   fig.savefig(os.path.join('plots', 'combined_panels.png'))\n"
        "9. Define a main() function that does all the work, and add at the end:\n"
        "if __name__ == '__main__':\n"
        "    main()\n\n"
        "Now output ONLY the Python code script as plain text. Do NOT include any explanations or Markdown fences."
    )

    resp = llm.invoke(prompt)
    content = resp.content if hasattr(resp, "content") else str(resp)
    state["generated_code"] = content
    return state


# ===== Node 4: Execution and Evaluation Agent =====

def exec_eval_node(state: PlotState) -> PlotState:
    """
    Execute the generated plotting code, collect generated images,
    and perform a simple LLM-based quality evaluation using the logs.
    """

    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    raw_code = state.get("generated_code", "")
    code = _sanitize_generated_code(raw_code)

    before = set(glob(os.path.join(output_dir, "plots", "*.png")))
    exec_tool = ExecuteCodeTool(work_dir=output_dir)
    exec_result = exec_tool._run(code=code)
    after = set(glob(os.path.join(output_dir, "plots", "*.png")))
    new_files: List[str] = sorted(list(after - before))

    state["exec_result"] = exec_result
    state["generated_images"] = new_files

    task = state.get("user_task", "")
    layout_spec = state.get("layout_spec", "")
    bio_map = state.get("bio_plot_mapping", "")

    eval_prompt = (
        "You are a scientific plotting reviewer. Evaluate whether the following Python code and its execution "
        "satisfy the task.\n\n"
        f"- Task (truncated): {task[:500]}\n"
        f"- Layout spec (truncated): {layout_spec[:1500]}\n"
        f"- Bio mapping (truncated): {bio_map[:1500]}\n"
        "- Code (truncated to first 4000 characters):\n"
        "```python\n"
        f"{code[:4000]}\n"
        "```\n"
        "- Execution log (truncated):\n"
        "stdout:\n"
        f"{exec_result.get('stdout', '')[:2000]}\n\n"
        "error:\n"
        f"{exec_result.get('error', '')[:2000]}\n\n"
        "Output JSON with:\n"
        "- score (1-5)\n"
        "- issues: list of issues found\n"
        "- suggestions: list of improvement suggestions\n"
        "No extra text."
    )

    eval_resp = llm.invoke(eval_prompt)
    eval_content = eval_resp.content if hasattr(eval_resp, "content") else str(eval_resp)
    state["quality_eval"] = eval_content
    return state


# ===== Build LangGraph workflow =====

def build_plot_graph():
    """
    Build a multi-node StateGraph: layout -> bio_domain -> code_gen -> exec_eval
    """
    graph = StateGraph(PlotState)

    graph.add_node("layout", style_extract_node)
    graph.add_node("bio_domain", bio_domain_node)
    graph.add_node("code_gen", code_node)
    graph.add_node("exec_eval", exec_eval_node)

    graph.set_entry_point("layout")
    graph.add_edge("layout", "bio_domain")
    graph.add_edge("bio_domain", "code_gen")
    graph.add_edge("code_gen", "exec_eval")
    graph.add_edge("exec_eval", END)

    app = graph.compile()
    return app

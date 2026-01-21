# BioPlotAgents

BioPlotAgents is a small, fully local, multi-agent plotting system for bioinformatics-style data.

Given:

- A reference figure (`ref.png`) from a paper or report
- A tabular dataset (`data.csv`)

the system uses a chain of specialized agents to:

1. Perceive the layout and style of the reference figure  
2. Infer suitable seaborn/matplotlib plot types and field mappings from the data  
3. Generate Python plotting code using a local language model (via vLLM)  
4. Execute and evaluate the code, iterating until valid PNGs are produced  

It automatically produces:

- A combined multi-panel plot: `combined_panels.png`
- One PNG per panel: `panel_1.png`, `panel_2.png`, `panel_3.png`, `panel_4.png`

The design is inspired by the [BioAgents](https://www.nature.com/articles/s41598-025-25919-z) multi-agent system and the general principles of agentic AI systems described in [“How to Build Agentic AI Systems from Scratch?”](https://medium.com/@ali_hamza/how-to-build-agentic-ai-systems-from-scratch-22c33999df91).

---

## Features

- **Agentic architecture**  
  - Style Agent: infers layout, panel grid, basic color/style hints from `ref.png`  
  - Domain/Mapping Agent: chooses seaborn plot families and column mappings based on data preview  
  - Code Agent: generates executable Python plotting code (seaborn + matplotlib), constrained to use only real columns  
  - Execution & Evaluation Agent: runs code in a sandbox, captures errors, checks for images, and scores quality  

- **Local-model friendly**  
  - Uses vLLM with an OpenAI-compatible API (e.g. `qwen2.5-coder-7b`)  
  - No calls to external cloud LLMs required  

- **Bioinformatics-aware but domain-neutral**  
  - Works with generic tabular data (e.g. RNA-seq summaries, imaging quantification, clinical variables)  
  - Infers numeric vs categorical columns and picks scatter/box/violin/heatmap etc. accordingly  

- **Self-adjusting loop**  
  - Outer agent loop retries multiple times when code generation or execution fails  
  - Each attempt passes previous errors into the reasoning context, encouraging better revisions  

---

## Repository Structure

```text
BioPlotAgents/
├─ src/
│  ├─ llm_langchain.py          # Local vLLM (ChatOpenAI-compatible) wrapper
│  ├─ state_types.py            # Typed state shared across agents
│  ├─ image_style_tools.py      # Basic image stats (size, grid, colors, colorbar)
│  ├─ style_extractor_chain.py  # Style/Layout Agent prompt + vLLM call
│  ├─ tools_langchain.py        # Data preview + code execution tools
│  ├─ pdf_figure_tools.py       # (Optional) extract PNG + text from PDF pages
│  ├─ graph_plot.py             # LangGraph definition: layout → domain → code → exec
│  ├─ deepagents_plot_agent.py  # Top-level agent with iterative attempts
│  ├─ run_direct.py             # CLI entry point (used by plot.sh)
│  └─ ...
├─ inputs/
│  ├─ ref.png                   # Example reference figure
│  └─ data.csv                  # Example tabular dataset
├─ outputs/
│  ├─ best_spec.json            # Latest run metadata (layout, mapping, code, logs)
│  └─ plots/
│     ├─ combined_panels.png    # Combined multi-panel plot
│     ├─ panel_1.png            # Individual panel 1
│     ├─ panel_2.png
│     ├─ panel_3.png
│     └─ panel_4.png
├─ start.sh                     # Example entry script (Singularity container)
└─ plot.sh                      # Example plotting script (calls run_direct.py)

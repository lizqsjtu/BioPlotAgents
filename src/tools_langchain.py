# src/tools_langchain.py
from typing import Dict, Any
from langchain.tools import BaseTool
from pydantic import PrivateAttr
import pandas as pd
import os
import io
import contextlib
import traceback
import matplotlib

matplotlib.use("Agg")


class BioDataPreviewTool(BaseTool):
    """
    Read a CSV/TSV file and return:
    - a preview (first rows as CSV string)
    - schema info (column name and dtype)
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="bio_data_preview_tool",
            description=(
                "Given a CSV/TSV file path, return preview (head) and schema info."
            ),
            **kwargs,
        )

    def _run(self, data_path: str) -> Dict[str, Any]:
        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)
        df = pd.read_csv(data_path, sep=None, engine="python")
        preview = df.head(10).to_csv(index=False)
        schema = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]
        return {"preview": preview, "schema": schema}

    async def _arun(self, data_path: str) -> Dict[str, Any]:
        raise NotImplementedError


class ExecuteCodeTool(BaseTool):
    """
    Simple code execution tool.

    It executes arbitrary Python code inside a working directory.
    Use only in controlled environments.
    """

    _work_dir: str = PrivateAttr()

    def __init__(self, work_dir: str = "./outputs", **kwargs):
        super().__init__(
            name="execute_code_tool",
            description=(
                "Execute Python code in a given working directory and capture outputs."
            ),
            **kwargs,
        )
        self._work_dir = work_dir
        os.makedirs(self._work_dir, exist_ok=True)

    def _run(self, code: str) -> Dict[str, Any]:
        stdout_buf = io.StringIO()
        success = True
        err = ""
        try:
            with contextlib.redirect_stdout(stdout_buf):
                old = os.getcwd()
                os.chdir(self._work_dir)
                try:
                    exec(code, {"__name__": "__main__"})
                finally:
                    os.chdir(old)
        except Exception:
            success = False
            err = traceback.format_exc()
        return {
            "success": success,
            "stdout": stdout_buf.getvalue(),
            "error": err,
            "work_dir": self._work_dir,
        }

    async def _arun(self, code: str) -> Dict[str, Any]:
        raise NotImplementedError

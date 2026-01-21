# src/pdf_figure_tools.py
from typing import Dict, Any
from langchain.tools import BaseTool
from pydantic import PrivateAttr
import os
from pdf2image import convert_from_path
from PyPDF2 import PdfReader


class PdfFigureExtractorTool(BaseTool):
    """
    Given a PDF file and a page index, render that page as a PNG,
    and extract the page text as a potential caption.
    """

    _output_dir: str = PrivateAttr()

    def __init__(self, output_dir: str = "./outputs/figures_from_pdf", **kwargs):
        super().__init__(
            name="pdf_figure_extractor_tool",
            description=(
                "Given a PDF file and page index, render that page as PNG and "
                "extract page text as potential caption."
            ),
            **kwargs,
        )
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

    def _run(self, pdf_path: str, page_index: int = 0) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)

        images = convert_from_path(
            pdf_path,
            dpi=300,
            first_page=page_index + 1,
            last_page=page_index + 1,
        )
        if not images:
            raise RuntimeError("No image rendered from PDF")

        img = images[0]
        out_path = os.path.join(self._output_dir, f"page_{page_index + 1}.png")
        img.save(out_path)

        reader = PdfReader(pdf_path)
        page = reader.pages[page_index]
        text = page.extract_text() or ""

        return {
            "image_path": out_path,
            "page_text": text,
        }

    async def _arun(self, pdf_path: str, page_index: int = 0) -> Dict[str, Any]:
        raise NotImplementedError

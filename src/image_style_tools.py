# src/image_style_tools.py
from typing import Dict, Any
from langchain.tools import BaseTool
from pydantic import PrivateAttr
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import os


class ImageBasicStatsTool(BaseTool):
    """
    Basic image analysis:
    - width/height
    - approximate subplot row/col count
    - color clusters
    - rough right-side colorbar detection

    This is used to give the LLM a rough sense of the style and structure
    of a reference figure.
    """

    _dummy: bool = PrivateAttr(default=False)

    def __init__(self, **kwargs):
        super().__init__(
            name="image_basic_stats_tool",
            description=(
                "Given a PNG/JPG image path, compute basic stats: width/height, "
                "approximate subplot layout (rows/cols), approximate color clusters, "
                "and a rough detection of a colorbar on the right side."
            ),
            **kwargs,
        )

    def _run(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        arr = np.array(img)
        h, w, _ = arr.shape

        # 1) Color clustering on a random sample of pixels
        sample_size = min(20000, h * w)
        flat = arr.reshape(-1, 3)
        if sample_size > 0:
            idx = np.random.choice(flat.shape[0], size=sample_size, replace=False)
            sample = flat[idx]
        else:
            sample = flat

        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(sample)
        centers = kmeans.cluster_centers_.astype(int)
        colors_hex = [
            "#{:02x}{:02x}{:02x}".format(*c.tolist()) for c in centers
        ]

        # 2) Rough subplot layout estimation via bright rows/cols
        gray = np.mean(arr, axis=2)
        row_mean = gray.mean(axis=1)
        col_mean = gray.mean(axis=0)

        row_threshold = float(np.percentile(row_mean, 90))
        col_threshold = float(np.percentile(col_mean, 90))

        def count_segments(mask: np.ndarray) -> int:
            count = 0
            in_seg = False
            for v in mask:
                if v and not in_seg:
                    in_seg = True
                elif not v and in_seg:
                    in_seg = False
                    count += 1
            if in_seg:
                count += 1
            return count

        row_empty_mask = row_mean > row_threshold
        col_empty_mask = col_mean > col_threshold
        n_row_gaps = count_segments(row_empty_mask)
        n_col_gaps = count_segments(col_empty_mask)

        approx_rows = min(4, max(1, n_row_gaps + 1))
        approx_cols = min(4, max(1, n_col_gaps + 1))

        # 3) Very simple right-side colorbar detection
        bar_detected = False
        right_strip = arr[:, int(0.85 * w):, :]
        rs = right_strip.reshape(-1, 3)
        if rs.shape[0] > 0:
            rs_sample = rs[
                np.random.choice(
                    rs.shape[0],
                    size=min(5000, rs.shape[0]),
                    replace=False,
                )
            ]
            uniq = np.unique(rs_sample, axis=0)
            uniq_ratio = uniq.shape[0] / rs_sample.shape[0]
            if uniq_ratio < 0.1:
                bar_detected = True

        return {
            "image_path": image_path,
            "width": int(width),
            "height": int(height),
            "approx_rows": int(approx_rows),
            "approx_cols": int(approx_cols),
            "color_clusters_hex": colors_hex,
            "maybe_has_colorbar": bool(bar_detected),
        }

    async def _arun(self, image_path: str) -> Dict[str, Any]:
        raise NotImplementedError

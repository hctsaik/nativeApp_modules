"""Process layer for module_006 (YOLO Image Viewer) — 純邏輯，無 Streamlit（契約）。

新版的 module_006：Input 只給『主要 / 對比 資料夾』，實際的瀏覽 / 疊圖 / IoU 比對
全部在 Output 頁（cvviewer_app，完整移植自 CV_Viewer）。本層只做輕量驗證：
確認主要資料夾存在、數一下有幾張影像，把資料夾路徑交給 Output。
"""
from __future__ import annotations

import os
from pathlib import Path

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def count_images(folder: str) -> int:
    """數資料夾內的影像張數；支援 YOLO 切分佈局（無直接影像但有 images/ 子夾 → 數 images/）。"""
    p = Path(folder)
    for root in (p, p / "images"):
        if not root.is_dir():
            continue
        try:
            n = sum(1 for f in root.iterdir()
                    if f.is_file() and f.suffix.lower() in _IMG_EXT)
        except OSError:
            n = 0
        if n:
            return n
    return 0


def execute_logic(params: dict) -> dict:
    main_folder = str(params.get("main_folder", "")).strip()
    compare_folder = str(params.get("compare_folder", "")).strip()

    if not main_folder:
        return {"error": "no_folder", "main_folder": "", "compare_folder": "", "n_images": 0}
    if not os.path.isdir(main_folder):
        return {"error": "bad_folder", "main_folder": main_folder,
                "compare_folder": compare_folder, "n_images": 0}

    n_images = count_images(main_folder)
    return {
        "main_folder": main_folder,
        "compare_folder": compare_folder if os.path.isdir(compare_folder) else "",
        "n_images": n_images,
        "error": "" if n_images else "no_images",
    }

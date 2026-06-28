"""Output layer for module_006 (YOLO Image Viewer).

完整的 viewer（縮圖牆 / OpenSeadragon 縮放畫布 / 上下張導覽 / 信心門檻 / 類別下拉 /
🔀 比較模式）由 cvviewer_app.render() 提供——忠實移植自獨立 app『CV_Viewer』。
本檔只負責三件事：
  1) 把本模組目錄放上 sys.path，讓自帶的支援模組（overlay / imageset / yolo / viewer …）
     可被 cvviewer_app `import`（_load_from_file 不會自動加目錄）；
  2) 讀 Input 頁寫入的資料夾、處理錯誤狀態；
  3) 呼叫 cvviewer_app.render(主要資料夾, 對比資料夾)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def render_output(result: dict) -> None:
    main_folder = (result.get("main_folder") or "").strip()
    err = result.get("error")

    if err == "no_folder" or not main_folder:
        st.warning("尚未選擇資料夾，請在 Input 頁籤選『主要資料夾』後按 ▶ 執行。")
        return
    if err == "bad_folder":
        st.error(f"找不到資料夾：{main_folder}")
        return

    # no_images 等其餘狀態交給 viewer 自己處理（它對空資料夾會顯示提示）。
    compare_folder = (result.get("compare_folder") or "").strip()
    import cvviewer_app  # noqa: PLC0415 — sibling module, importable via _HERE on sys.path
    cvviewer_app.render(main_folder, compare_folder)

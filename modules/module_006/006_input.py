"""Input layer for module_006 (YOLO Image Viewer) — 只選『主要 / 對比』兩個資料夾。

其餘檢視（縮圖牆 / 縮放畫布 / 導覽 / 信心門檻 / 類別 / 比較模式）全部在 Output 頁
（cvviewer_app.render，忠實移植自獨立 app CV_Viewer）。選好資料夾→按 ▶ 執行→切到 Output。
"""
from __future__ import annotations

import subprocess
import sys

import streamlit as st


def _browse_folder() -> str:
    """開原生資料夾選擇對話框（本機桌面）；headless/無顯示則回空字串、不崩。
    走獨立子程序的 tkinter askdirectory（與 module_026 同範式，避開 worker thread 限制）。"""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import tkinter as tk; from tkinter import filedialog; "
             "root=tk.Tk(); root.withdraw(); root.wm_attributes('-topmost',True); "
             "p=filedialog.askdirectory(title='選擇資料夾'); "
             "root.destroy(); print(p or '',end='')"],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _folder_field(label: str, key: str, help_txt: str) -> str:
    """資料夾輸入欄 + 原生『📁』選擇鈕（可打字、也可用選的）。回目前路徑字串。
    瀏覽鈕在 text_input 實例化『之前』寫 session_state[key]（合法）再 rerun → 回填欄位。"""
    st.session_state.setdefault(key, "")
    col_in, col_btn = st.columns([6, 1], vertical_alignment="bottom")
    if col_btn.button("📁", key=f"{key}__browse", help=help_txt):
        picked = _browse_folder()
        if picked:
            st.session_state[key] = picked
            st.rerun()
    return col_in.text_input(label, key=key)


def render_input() -> dict:
    st.subheader(":material/folder_open: 資料來源")
    st.caption("選好資料夾後按 ▶ 執行，再切到 **Output** 分頁瀏覽（縮圖牆 / 縮放畫布 / 比較模式）。")

    main_folder = _folder_field(
        "主要資料夾（影像 + Model A 標註；自動偵測 images/、labels/ 子夾）",
        "m006_main_folder", "選擇主要影像資料夾（= Model A 的影像 + 標註）")
    compare_folder = _folder_field(
        "第二個 model 結果資料夾 B（比較模式；留空 = 不比較）",
        "m006_compare_folder", "選擇 Model B 的標註資料夾（對同一包影像）")

    return {
        "main_folder": main_folder.strip(),
        "compare_folder": compare_folder.strip(),
    }

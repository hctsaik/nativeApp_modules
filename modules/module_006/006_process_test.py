"""Unit tests for module_006 process layer (folder-based YOLO viewer).

新版 module_006：Input 只給資料夾，process 只做輕量驗證 + 數影像。
完整 viewer（縮圖牆 / 縮放畫布 / 比較）在 Output（cvviewer_app，由 E2E 驗收）。

    py -3.11 -m pytest modules/module_006/006_process_test.py -q
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PF = Path(__file__).parent / "006_process.py"
_spec = importlib.util.spec_from_file_location("module_006_process", _PF)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
execute_logic = _mod.execute_logic
count_images = _mod.count_images


def _touch_imgs(folder: Path, names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"x")  # count_images 只看副檔名/是否為檔，不解碼


# ── count_images ──────────────────────────────────────────
def test_count_images_direct(tmp_path):
    _touch_imgs(tmp_path, ["a.png", "b.jpg", "c.TIF", "notes.txt", "d.bmp"])
    assert count_images(str(tmp_path)) == 4  # png/jpg/tif/bmp（.txt 不算）


def test_count_images_yolo_split_images_subdir(tmp_path):
    # 無直接影像但有 images/ 子夾（典型 <split>/images）→ 數 images/
    _touch_imgs(tmp_path / "images", ["x.png", "y.jpeg"])
    (tmp_path / "labels").mkdir()
    assert count_images(str(tmp_path)) == 2


def test_count_images_empty(tmp_path):
    (tmp_path / "labels").mkdir()
    assert count_images(str(tmp_path)) == 0


# ── execute_logic ─────────────────────────────────────────
def test_execute_no_folder():
    r = execute_logic({"main_folder": "", "compare_folder": ""})
    assert r["error"] == "no_folder" and r["n_images"] == 0


def test_execute_bad_folder():
    r = execute_logic({"main_folder": r"Z:\nope\nope-12345"})
    assert r["error"] == "bad_folder"


def test_execute_valid_with_images(tmp_path):
    _touch_imgs(tmp_path, ["1.png", "2.png"])
    r = execute_logic({"main_folder": str(tmp_path)})
    assert r["error"] == "" and r["n_images"] == 2
    assert r["main_folder"] == str(tmp_path)
    assert r["compare_folder"] == ""


def test_execute_valid_no_images(tmp_path):
    (tmp_path / "labels").mkdir()
    r = execute_logic({"main_folder": str(tmp_path)})
    assert r["error"] == "no_images" and r["n_images"] == 0


def test_execute_compare_folder_kept_only_if_exists(tmp_path):
    _touch_imgs(tmp_path, ["1.png"])
    cmp_dir = tmp_path.parent / "cmp"
    cmp_dir.mkdir()
    r = execute_logic({"main_folder": str(tmp_path), "compare_folder": str(cmp_dir)})
    assert r["compare_folder"] == str(cmp_dir)
    r2 = execute_logic({"main_folder": str(tmp_path), "compare_folder": r"Z:\nope-cmp"})
    assert r2["compare_folder"] == ""


def test_result_is_json_serializable(tmp_path):
    import json
    _touch_imgs(tmp_path, ["1.png"])
    json.dumps(execute_logic({"main_folder": str(tmp_path)}))


# ── 契約：process 不得 import streamlit ───────────────────
def test_no_streamlit_import():
    src = _PF.read_text(encoding="utf-8")
    assert "import streamlit" not in src and "from streamlit" not in src

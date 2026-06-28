"""sidecar 模組(M2 / Tier B)。

每張影像一份同目錄、同檔名主幹的 sidecar JSON(`<name>.cvr.json`),
持久化人工判讀狀態,絕不修改原始影像檔。

契約見 3_Architect_Design/06_sidecar.md。除 default()/sidecar_path()/load()/save()
外的函式皆為純函式(copy-on-write,不就地改、不碰時鐘/亂數/檔案系統)。

僅依賴 Python 標準庫。
"""
import copy
import json
import os
import tempfile

# 設計 §2.1 合法的 review_status 列舉
_LEGAL_STATUS = ("none", "need_review", "done")


def default() -> dict:
    """回傳一份全新、獨立的預設 sidecar dict(§2.1 八欄位、固定初值)。"""
    return {
        "review_status": "none",
        "tags": [],
        "verdict": "unset",
        "comment": "",
        "bookmarked": False,
        "rois": [],
        "reviewer": "",
        "timestamp": "",
    }


def sidecar_path(image_path) -> str:
    """同目錄、檔名主幹不變、副檔名換成(或附加).cvr.json,回傳 str。"""
    p = os.fspath(image_path)
    root, _ext = os.path.splitext(p)
    return root + ".cvr.json"


def load(image_path) -> dict:
    """讀取 sidecar;不存在 / 空檔 / 壞 JSON 都回 default()。
    合法但缺欄位時以 default() 補齊缺欄位,既有值保留。
    """
    path = sidecar_path(image_path)
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except (FileNotFoundError, OSError):
        return default()

    if raw.strip() == "":
        return default()

    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return default()

    if not isinstance(parsed, dict):
        return default()

    result = default()
    for key in result:
        if key in parsed:
            result[key] = parsed[key]
    return result


def save(image_path, data) -> None:
    """原子寫入:先寫同目錄暫存檔,flush+fsync 後 os.replace 換名。"""
    path = sidecar_path(image_path)
    directory = os.path.dirname(path) or "."

    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=os.path.basename(path) + ".", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # 失敗時清掉殘留暫存檔,避免污染目錄
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def set_status(data, status) -> dict:
    """設定 review_status;非法值丟 ValueError(及早失敗,不就地改輸入)。"""
    if status not in _LEGAL_STATUS:
        raise ValueError(
            "invalid review_status: %r (must be one of %r)" % (status, _LEGAL_STATUS)
        )
    new = copy.deepcopy(data)
    new["review_status"] = status
    return new


def toggle_bookmark(data) -> dict:
    """反轉 bookmarked(copy-on-write)。"""
    new = copy.deepcopy(data)
    new["bookmarked"] = not new["bookmarked"]
    return new


def add_tag(data, tag) -> dict:
    """加入 tag,去重且保序(copy-on-write)。"""
    new = copy.deepcopy(data)
    if tag not in new["tags"]:
        new["tags"].append(tag)
    return new


def remove_tag(data, tag) -> dict:
    """移除 tag;不存在視為 no-op(copy-on-write,不丟錯)。"""
    new = copy.deepcopy(data)
    if tag in new["tags"]:
        new["tags"] = [t for t in new["tags"] if t != tag]
    return new


def add_roi(data, bbox, label="", verdict="unset", comment="") -> dict:
    """追加一個 ROI;bbox 存成長度 4 的 list(copy-on-write)。"""
    new = copy.deepcopy(data)
    new["rois"].append({
        "bbox": [int(v) for v in bbox],
        "label": label,
        "verdict": verdict,
        "comment": comment,
    })
    return new

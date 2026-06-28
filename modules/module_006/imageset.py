"""imageset (Tier B, M1b) — 把資料夾變成可自然排序/重排/進度/位置持久化的影像清單。

依 3_Architect_Design/03_imageset.md 契約實作。只用標準庫。
"""
import json
import os
import re

_DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

_NUM_RE = re.compile(r"(\d+)")


def _natural_key(name):
    """自然排序鍵:連續數字當整數比較,其餘字元 lower() 後逐字比較。"""
    parts = _NUM_RE.split(str(name))
    key = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # 奇數段為數字(split 以捕獲群組切出)
            key.append((1, int(part), ""))
        else:
            if part == "":
                continue
            key.append((0, 0, part.lower()))
    return key


def _norm_folder(folder):
    """資料夾路徑正規化:abspath + normcase,使不同寫法對應同一 key。"""
    return os.path.normcase(os.path.abspath(folder))


def scan(folder, exts=_DEFAULT_EXTS):
    """掃描資料夾,回傳影像記錄清單(依檔名自然排序給 0-based index)。"""
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)

    exts_lower = tuple(e.lower() for e in exts)
    records = []
    for entry in os.listdir(folder):
        full = os.path.join(folder, entry)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext not in exts_lower:
            continue
        abspath = os.path.abspath(full)
        records.append({
            "path": abspath,
            "name": os.path.basename(abspath),
            "index": 0,
            "mtime": float(os.path.getmtime(full)),
            "size": int(os.path.getsize(full)),
        })

    records.sort(key=lambda r: _natural_key(r["name"]))
    for i, r in enumerate(records):
        r["index"] = i
    return records


def sort_records(records, key="name"):
    """回新 list 並重編 index;key in {"name","time","size"}。"""
    if key not in ("name", "time", "size"):
        raise ValueError("invalid key: {!r}".format(key))

    # 複製每筆 dict,避免修改傳入物件
    copied = [dict(r) for r in records]

    if key == "name":
        sort_key = lambda r: _natural_key(r["name"])
    elif key == "time":
        sort_key = lambda r: (r["mtime"], _natural_key(r["name"]))
    else:  # size
        sort_key = lambda r: (r["size"], _natural_key(r["name"]))

    copied.sort(key=sort_key)
    for i, r in enumerate(copied):
        r["index"] = i
    return copied


def progress(index, total):
    """顯示 "i / n"(1-based);total<=0 → ValueError。"""
    if total <= 0:
        raise ValueError("total must be > 0, got {!r}".format(total))
    return "{} / {}".format(index + 1, total)


class Position:
    """記住每資料夾上次 index,以 json 狀態檔持久化。"""

    def __init__(self, state_path):
        self.state_path = state_path

    def _load(self):
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.loads(f.read())
            if isinstance(data, dict):
                return data
            return {}
        except (OSError, ValueError):
            return {}

    def get(self, folder, default=0):
        data = self._load()
        k = _norm_folder(folder)
        if k in data:
            try:
                return int(data[k])
            except (ValueError, TypeError):
                return default
        return default

    def set(self, folder, index):
        data = self._load()
        data[_norm_folder(folder)] = int(index)
        parent = os.path.dirname(self.state_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data))

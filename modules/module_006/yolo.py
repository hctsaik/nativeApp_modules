"""yolo 模組實作(/pg 產物;依 3_Architect_Design/07_yolo.md 契約)。

把單張影像的 YOLO 偵測 JSON(多種來源 schema)載入並正規化成跨模組共用的
`list[Detection]`(絕對像素 `[x,y,w,h]` bbox),容錯第一:任何髒資料只跳過該筆
或回空清單,絕不拋例外。

對外承諾(§6):
- `load` 永不拋例外;回傳形狀恆為 `list[Detection]`。
- 每筆 `Detection` 恰三鍵 `bbox`/`cls`/`conf`:
  bbox 為四個 int、conf 為 [0,1] 內 float、cls 為 str。
- 僅依賴 Python 標準庫;不修改原圖、不寫任何檔。

依賴:json / os / pathlib(標準庫)。
"""
import json
import os

# 來源容器頂層 dict 時,依序找第一個「存在且為 list」的鍵(§2.2)。
_CONTAINER_KEYS = ("detections", "predictions", "objects")

# 類別名別名,依序取第一個存在者(§2.3)。
_CLASS_KEYS = ("cls", "class", "name", "label")

# 信心別名,依序取第一個存在者(§2.3)。
_CONF_KEYS = ("conf", "confidence", "score")


def _is_number(v):
    # bool 是 int 子類別;座標/信心語義上不該接受 True/False 當數值。
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _as_len4_numbers(v):
    """v 為「長度 4 的數值序列」→ 回 [float,float,float,float];否則回 None。

    字串(即使長度 4)不算數值序列;含任何非數值元素 → None。
    """
    if not isinstance(v, (list, tuple)):
        return None
    if len(v) != 4:
        return None
    nums = []
    for el in v:
        if not _is_number(el):
            return None
        nums.append(float(el))
    return nums


def _box_from(obj, img_w, img_h):
    """依優先序 bbox > xyxy > xywhn 找第一個「存在且為長度 4 數值序列」的框鍵,
    換算成絕對像素 [x,y,w,h](float)。

    - 找不到任何有效框鍵 → None。
    - 命中 xywhn 但缺 img_w/img_h(None 或 ≤0)→ None(該筆無法換算)。

    注意:依契約 §2.3「找第一個存在且為長度 4 的數值序列」,某框鍵存在但無效
    (長度錯/含非數值)時,跳過它、繼續找下一個鍵。
    """
    # 1) bbox:絕對 xywh,直接採用。
    if "bbox" in obj:
        nums = _as_len4_numbers(obj["bbox"])
        if nums is not None:
            x, y, w, h = nums
            return [x, y, w, h]

    # 2) xyxy:絕對對角點 → xywh(min/abs 正規化)。
    if "xyxy" in obj:
        nums = _as_len4_numbers(obj["xyxy"])
        if nums is not None:
            x1, y1, x2, y2 = nums
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            return [x, y, w, h]

    # 3) xywhn:正規化中心點 + 正規化寬高 → 絕對 xywh(需有效尺寸)。
    if "xywhn" in obj:
        nums = _as_len4_numbers(obj["xywhn"])
        if nums is not None:
            if not _is_number(img_w) or not _is_number(img_h):
                return None
            if img_w <= 0 or img_h <= 0:
                return None
            xc, yc, wn, hn = nums
            w = wn * img_w
            h = hn * img_h
            x = xc * img_w - w / 2
            y = yc * img_h - h / 2
            return [x, y, w, h]

    return None


def _class_of(obj):
    """類別名:依序 cls/class/name/label 取第一個存在者 → str;皆缺 → ""。"""
    for key in _CLASS_KEYS:
        if key in obj:
            return str(obj[key])
    return ""


def _conf_of(obj):
    """信心:依序 conf/confidence/score 取第一個存在者 → float clamp[0,1];
    皆缺 / 轉 float 失敗 → 1.0。"""
    for key in _CONF_KEYS:
        if key in obj:
            try:
                c = float(obj[key])
            except (TypeError, ValueError):
                return 1.0
            if c < 0.0:
                return 0.0
            if c > 1.0:
                return 1.0
            return c
    return 1.0


def normalize_one(obj, img_w=None, img_h=None):
    """把單一原始偵測 dict 正規化成一筆 Detection;無法正規化 → None。

    純函式、不做 I/O、不拋例外、不 mutate 輸入。load() 內部對每筆呼叫它。
    """
    if not isinstance(obj, dict):
        return None

    box = _box_from(obj, img_w, img_h)
    if box is None:
        return None

    bbox = [int(v) for v in box]  # int() 向 0 截斷(§4.e)
    return {"bbox": bbox, "cls": _class_of(obj), "conf": _conf_of(obj)}


def _extract_array(data):
    """從已解析的頂層 JSON 抽出「偵測陣列」(§2.2)。

    - 頂層即 list → 它本身。
    - 頂層為 dict → 依序找第一個存在且為 list 的容器鍵;皆無 → []。
    - 其餘型別 → []。
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in _CONTAINER_KEYS:
            v = data.get(key)
            if isinstance(v, list):
                return v
        return []
    return []


def _load_yolo_txt(path, img_w, img_h, names):
    """YOLO 標註 .txt:每行 `cls_id cx cy w h [conf]`(空白分隔,座標正規化 0~1)。
    需有效 img_w/img_h 才能換算絕對像素(否則回 []);壞行跳過;永不拋例外。
    names:可選 id→類別名清單(缺或越界 → cls=str(id))。輸出與 .json 同形 Detection。"""
    if not _is_number(img_w) or not _is_number(img_h) or img_w <= 0 or img_h <= 0:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    names = names if isinstance(names, (list, tuple)) else None
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, wn, hn = (float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4]))
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
        except (TypeError, ValueError):
            continue
        w = wn * img_w
        h = hn * img_h
        x = cx * img_w - w / 2.0
        y = cy * img_h - h / 2.0
        conf = 0.0 if conf < 0.0 else (1.0 if conf > 1.0 else conf)
        cls = names[cid] if (names and 0 <= cid < len(names)) else str(cid)
        out.append({"bbox": [int(x), int(y), int(w), int(h)], "cls": str(cls), "conf": conf})
    return out


def load(label_path, img_w=None, img_h=None, names=None):
    """讀單張影像的偵測檔 → 正規化成 list[Detection]。

    支援兩種格式(依副檔名):
      - `.json`:多來源 schema(bbox/xyxy/xywhn;cls/name…;conf/score…),容錯正規化。
      - `.txt`:YOLO 格式,每行 `cls_id cx cy w h [conf]`(正規化座標),需 img_w/img_h 換算;
        `names` 提供 id→類別名。
    label_path: str | pathlib.Path。檔不存在 / 壞檔 / 空 → []。
    img_w, img_h: 正整數或 None;換算 xywhn / YOLO txt 時必需。
    names: 可選 list[str](僅 .txt 使用)。永不拋例外。
    """
    path = os.fspath(label_path)  # 支援 str 與 pathlib.Path

    if not os.path.isfile(path):
        return []

    if path.lower().endswith(".txt"):
        return _load_yolo_txt(path, img_w, img_h, names)

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return []

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        # 空字串 / 壞 JSON → JSONDecodeError(ValueError 子類別)→ []
        return []

    array = _extract_array(data)

    result = []
    for raw in array:
        det = normalize_one(raw, img_w, img_h)
        if det is not None:
            result.append(det)
    return result

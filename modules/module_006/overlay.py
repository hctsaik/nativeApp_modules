"""overlay 模組實作(PG 上採樣 / 生成層)。

依契約:3_Architect_Design/08_overlay.md(AC1..AC36 + §3 資料流 + §4 邊界)。
純邏輯、零 I/O、零 GUI(Tier A)。僅依賴 numpy 與 Python 標準庫。

嚴禁 import yolo / sidecar / tagging(本模組只「消費」Detection dict 形狀,
不產生、不持久化;以資料形狀解耦,可獨立平行驗收)。

座標慣例:array[row=y, col=x, channel],RGB uint8,(H, W, 3)。
邊框語義為半開區間(§3.3b):框覆蓋像素為 x <= col < x+w 且 y <= row < y+h。
"""

import numpy as np

# class → 顏色 的內建對照(契約值,釘死;§2.4)
CLASS_COLORS = {
    "defect": (255, 0, 0),
    "scratch": (0, 255, 0),
    "dent": (0, 0, 255),
}

# 未命中 CLASS_COLORS、且未顯式給 color 時的回退色(紅;§2.4)
DEFAULT_COLOR = (255, 0, 0)

# 標籤帶高度(實作自定的小常數;文字字形/精確像素不列為釘值 AC,§3.4)
LABEL_H = 2


def filter_detections(dets, conf_threshold=0.0, classes=None):
    """保留 conf >= conf_threshold 且 (classes is None 或 cls in classes) 者。

    保序、不 mutate;回新 list,元素為原 dict 物件的參照(淺層,不複製 dict)。
    conf 缺鍵 → 視同 0.0;cls 缺鍵 → 視同 None(§3.1 / §4c / §4d)。
    """
    kept = []
    for det in dets:
        conf_ok = float(det.get("conf", 0.0)) >= conf_threshold
        cls_ok = (classes is None) or (det.get("cls", None) in classes)
        if conf_ok and cls_ok:
            kept.append(det)
    return kept


def color_for(det, color=None):
    """決定一筆 det 的繪製顏色(§3.2)。

    color 顯式給 → 回 tuple(color)(覆蓋對照);
    否則 → CLASS_COLORS.get(det.get("cls"), DEFAULT_COLOR)。
    """
    if color is not None:
        return tuple(color)
    return CLASS_COLORS.get(det.get("cls"), DEFAULT_COLOR)


def _draw_box(out, x, y, w, h, color, thickness):
    """在 out 上畫單一 bbox 的邊框(向框內加厚),夾界、半開區間。

    框覆蓋半開區間:x <= col < x+w 且 y <= row < y+h(§3.3b)。
    粗細 t:像素到最近外邊的距離 k(0 起)滿足 k < t 才塗(向內加厚);
    t >= min(w,h) → 整框實心填滿;t <= 0 → 不畫(§3.3d)。
    完全在影像外的部分不寫、不拋例外(§3.3c)。
    """
    if w <= 0 or h <= 0:
        return
    if thickness <= 0:
        return

    H, W = out.shape[0], out.shape[1]

    # 框的半開覆蓋範圍(影像座標),含外框最外圈與最內圈
    x0, x1 = x, x + w - 1            # 左 / 右 外邊欄(col)
    y0, y1 = y, y + h - 1            # 上 / 下 外邊列(row)

    # 夾到影像界內的迭代範圍(超界部分不寫)
    col_lo = max(x0, 0)
    col_hi = min(x1, W - 1)
    row_lo = max(y0, 0)
    row_hi = min(y1, H - 1)
    if col_lo > col_hi or row_lo > row_hi:
        return  # 框完全在影像外

    color_arr = np.array(color, dtype=out.dtype)

    for row in range(row_lo, row_hi + 1):
        for col in range(col_lo, col_hi + 1):
            # 該像素到最近外邊的距離(以未夾界的外框邊定義,確保加厚向框內一致)
            k = min(col - x0, x1 - col, row - y0, y1 - row)
            if k < thickness:
                out[row, col, :] = color_arr


def _draw_label(out, x, y, w, h, text, color):
    """畫標籤帶(框上緣外側的小色帶),夾界、不拋例外(§3.4)。

    標籤可驗收性僅約束:不影響 draw_label=False 的 bbox 像素、不拋例外、
    形狀/型別不變、空 dets 仍逐像素等於輸入。文字字形不列為釘值 AC。
    這裡畫框上緣外側 row 從 y-LABEL_H 起的色帶(界外夾掉),且避開 bbox 邊像素。
    """
    if w <= 0 or h <= 0:
        return
    H, W = out.shape[0], out.shape[1]
    color_arr = np.array(color, dtype=out.dtype)

    # 標籤帶:上緣外側,row in [y-LABEL_H, y-1](嚴格在框上邊 row=y 之外)
    row_start = y - LABEL_H
    row_end = y - 1
    col_start = x
    col_end = x + w - 1  # 與 bbox 同寬,但只佔上緣外側列,不碰 bbox 邊像素

    for row in range(row_start, row_end + 1):
        if row < 0 or row >= H:
            continue
        for col in range(col_start, col_end + 1):
            if col < 0 or col >= W:
                continue
            out[row, col, :] = color_arr


def draw(array, dets, color=None, thickness=1, conf_threshold=0.0,
         classes=None, draw_label=False):
    """在影像上畫 bbox(+可選 class/conf 標籤),回新陣列(輸入不被 mutate)。

    流程(§3.3):
      1. out = array.copy()(輸入永不 mutate)。
      2. kept = filter_detections(dets, conf_threshold, classes)。
      3. 對 kept 依輸入順序畫 bbox(後畫者覆蓋先畫者的重疊像素);座標夾界。
      4. draw_label=True 時另畫文字標籤帶(策略見 §3.4)。
      5. 回 out。
    """
    out = array.copy()
    kept = filter_detections(dets, conf_threshold, classes)

    for det in kept:
        bbox = det["bbox"]
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])
        c = color_for(det, color)
        _draw_box(out, x, y, w, h, c, thickness)

    if draw_label:
        for det in kept:
            bbox = det["bbox"]
            x = int(bbox[0])
            y = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
            c = color_for(det, color)
            text = "{} {:.2f}".format(det.get("cls", ""), float(det.get("conf", 0.0)))
            _draw_label(out, x, y, w, h, text, c)

    return out

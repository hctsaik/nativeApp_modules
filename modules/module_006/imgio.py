"""imgio — Tier B (M1b) 影像讀寫 + 顯示映射模組。

契約見 3_Architect_Design/02_imgio.md。
僅用 numpy / Pillow / tifffile / cv2(可選)/ 標準庫;零 GUI、零網路。
"""
import base64
import io
import os

import numpy as np
import tifffile
from PIL import Image

_TIFF_EXTS = {".tif", ".tiff"}
_PIL_EXTS = {".png", ".jpg", ".jpeg"}
_SUPPORTED_EXTS = _TIFF_EXTS | _PIL_EXTS


def _normalize_array(arr):
    """把讀回的陣列正規化成契約允許的形狀/dtype，否則 ValueError。

    回傳 (array, bit_depth, channels)。
    """
    a = np.asarray(arr)

    # 去掉多餘的單一通道維 (HxWx1 → HxW)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]

    if a.ndim == 2:
        if a.dtype == np.uint8:
            return a, 8, 1
        if a.dtype == np.uint16:
            return a, 16, 1
        raise ValueError(
            "unsupported grayscale dtype: %r (MVP 僅支援 uint8/uint16)" % a.dtype)

    if a.ndim == 3:
        ch = a.shape[2]
        # RGBA → 安全去 alpha（不透明則轉 RGB；否則範圍外）
        if ch == 4:
            if a.dtype != np.uint8:
                raise ValueError("unsupported RGBA dtype: %r" % a.dtype)
            alpha = a[:, :, 3]
            if np.all(alpha == 255):
                a = np.ascontiguousarray(a[:, :, :3])
                ch = 3
            else:
                raise ValueError("RGBA with non-opaque alpha 不在 MVP 契約內")
        if ch == 3:
            if a.dtype != np.uint8:
                raise ValueError(
                    "unsupported RGB dtype: %r (MVP 僅支援 8-bit RGB)" % a.dtype)
            return np.ascontiguousarray(a), 8, 3
        raise ValueError("unsupported channel count: %d" % ch)

    raise ValueError("unsupported array ndim: %d" % a.ndim)


def load(path):
    """讀檔成 {array,width,height,bit_depth,channels}。"""
    p = os.fspath(path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)

    ext = os.path.splitext(p)[1].lower()
    if ext not in _SUPPORTED_EXTS:
        raise ValueError("unsupported extension: %r" % ext)

    try:
        if ext in _TIFF_EXTS:
            raw = tifffile.imread(p)
        else:
            with Image.open(p) as im:
                im.load()
                raw = np.asarray(im)
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:  # 解碼失敗一律歸為非影像 → ValueError
        raise ValueError("failed to decode image %r: %s" % (p, e))

    array, bit_depth, channels = _normalize_array(raw)
    return {
        "array": array,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "bit_depth": bit_depth,
        "channels": channels,
    }


def to_display_rgb(array, lo=None, hi=None):
    """window/level 顯示映射 → HxWx3 uint8。"""
    a = np.asarray(array)

    # 已是 HxWx3 uint8 RGB → 原樣回傳（忽略 lo/hi）
    if a.ndim == 3 and a.shape[2] == 3 and a.dtype == np.uint8:
        return a

    if a.ndim != 2:
        raise ValueError("to_display_rgb 期待 HxW 灰階或 HxWx3 uint8 RGB")

    if lo is None and hi is None:
        amin = a.min() if a.size else 0
        amax = a.max() if a.size else 0
        if amax == amin:
            gray = np.zeros(a.shape, dtype=np.uint8)
        else:
            scaled = (a.astype(np.float64) - float(amin)) * 255.0 / (
                float(amax) - float(amin))
            gray = np.clip(np.floor(scaled + 0.5), 0, 255).astype(np.uint8)
    else:
        if lo is None or hi is None:
            raise ValueError("lo/hi 必須同時為 None 或同時給定")
        lo = float(lo)
        hi = float(hi)
        if lo > hi:
            raise ValueError("lo > hi 非法")
        af = a.astype(np.float64)
        if hi == lo:
            # 階梯：value < lo → 0，value >= lo → 255
            gray = np.where(af < lo, 0, 255).astype(np.uint8)
        else:
            scaled = (af - lo) * 255.0 / (hi - lo)
            gray = np.clip(np.floor(scaled + 0.5), 0, 255).astype(np.uint8)

    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


def value_at(array, x, y):
    """取原始真值（x=col, y=row）。超界 → IndexError。"""
    a = np.asarray(array)
    height = a.shape[0]
    width = a.shape[1]
    xi = int(x)
    yi = int(y)
    if xi < 0 or yi < 0 or xi >= width or yi >= height:
        raise IndexError("coordinate (x=%r, y=%r) out of bounds" % (x, y))

    if a.ndim == 2:
        return int(a[yi, xi])
    if a.ndim == 3 and a.shape[2] == 3:
        px = a[yi, xi]
        return (int(px[0]), int(px[1]), int(px[2]))
    raise ValueError("value_at 不支援此 array 形狀")


def crop(array, x, y, w, h):
    """安全 clamp 裁切，不丟例外。"""
    a = np.asarray(array)
    height = a.shape[0]
    width = a.shape[1]

    x0 = min(max(int(x), 0), width)
    y0 = min(max(int(y), 0), height)
    x1 = min(max(int(x) + int(w), 0), width)
    y1 = min(max(int(y) + int(h), 0), height)

    if x1 < x0:
        x1 = x0
    if y1 < y0:
        y1 = y0

    return a[y0:y1, x0:x1]


def _validate_uint8_image(rgb_uint8):
    a = np.asarray(rgb_uint8)
    if a.dtype != np.uint8:
        raise ValueError("expected uint8 image, got dtype %r" % a.dtype)
    if a.ndim == 2:
        return a, "L"
    if a.ndim == 3 and a.shape[2] == 3:
        return a, "RGB"
    raise ValueError("expected HxW 或 HxWx3 uint8, got shape %r" % (a.shape,))


def to_png_bytes(rgb_uint8):
    """HxWx3(或 HxW)uint8 → PNG 位元組。"""
    a, mode = _validate_uint8_image(rgb_uint8)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a), mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def to_data_url(rgb_uint8):
    """→ data:image/png;base64,<b64>。"""
    png = to_png_bytes(rgb_uint8)
    b64 = base64.b64encode(png).decode("ascii")
    return "data:image/png;base64," + b64


def thumbnail(array, max_px=256):
    """auto window/level 後等比例縮放，長邊 <= max_px（只縮不放）→ HxWx3 uint8。"""
    a = np.asarray(array)
    if a.ndim < 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("thumbnail 收到退化輸入（某邊為 0）")

    disp = to_display_rgb(a)  # HxWx3 uint8
    h, w = disp.shape[0], disp.shape[1]
    long_edge = max(h, w)

    if long_edge <= max_px:
        return disp

    scale = max_px / float(long_edge)
    if w >= h:
        new_w = max_px
        new_h = int(round(h * scale))
    else:
        new_h = max_px
        new_w = int(round(w * scale))
    new_w = max(new_w, 1)
    new_h = max(new_h, 1)

    img = Image.fromarray(np.ascontiguousarray(disp), mode="RGB")
    img = img.resize((new_w, new_h), Image.BILINEAR)
    return np.asarray(img)

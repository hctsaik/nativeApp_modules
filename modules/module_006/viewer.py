"""OpenSeadragon viewer 的 Streamlit 包裝(宣告式元件,無 React build)。

回傳最近一次事件 dict,或 None:
  {"type":"click", "x":int, "y":int, "zoom":float, "n":int}
  {"type":"roi",   "bbox":[x,y,w,h], "n":int}

注意:這是 Tier B 整合件(GUI),由真實 E2E / 人工驗收,不在純單元測試範圍。
影像以 data URL 或 http URL 餵入(由 imageio.to_data_url 產生);ROI 以 [{"bbox":[x,y,w,h],"label":str}] 畫出。
大圖可改餵 tiles(dzitiles 風格的金字塔,瓦片為 data URL)走 OSD 自訂 tile source,不送全解析度單圖。
"""
from pathlib import Path

import streamlit.components.v1 as components

_DIR = Path(__file__).parent / "viewer_component"
_component = components.declare_component("cv_viewer", path=str(_DIR))


def osd_viewer(image_url: str = None, rois=None, tiles=None, height: int = 600,
               key: str = "cv_viewer",
               meta: dict = None, dets=None, max_zoom_pixel_ratio: float = 24.0,
               restore_zoom: float = None, restore_center=None,
               nav_keys: bool = False, auto_height: bool = False):
    """image_url 與 tiles 二擇一:tiles 非 None 時走金字塔瓦片模式(大圖)。
    tiles = {"width","height","tile_size","overlap","max_level",
             "tiles": {level(int): {"col_row": <data url>}}}。

    meta:dict|None  HUD 頂列要顯示的影像中繼資料(形狀見設計 §2.1a),不影響取樣;
                    meta=None → HUD 退回 M5 舊版座標列(向後相容)。
    dets:list|None  『過濾後』的偵測 list(形狀=Detection;app 端已套 conf/class 篩選),
                    供 HUD『游標落在框內顯示 cls·conf』;data url 已燒框,dets 只供 HUD 文字命中。
    max_zoom_pixel_ratio:float  傳給 OSD 的 maxZoomPixelRatio(放大上限;契約預設 24.0)。

    M7 新增四參數(皆可選、預設保留 M6 行為,見設計 §2.1):
    restore_zoom:float|None    切張後要還原的 viewport zoom;None → 新圖預設 fit(M6 行為)。
                               注意:固定 key 下元件不 remount、OSD 物件存活,M7a 切張保存是
                               『純 client 端、image-swap 前自存當下 viewport 再 open 後還原』,
                               不需 Python round-trip;此參數供 M7b 鍵盤 nav 經 args 帶回時使用。
    restore_center:list|None   要還原的 viewport center [x,y];None → 不還原 pan。
    nav_keys:bool              M7b 鍵盤熱鍵開關;M7a 不啟用鍵盤(本切片不接 keydown)。
    auto_height:bool           True → 元件端量 window.innerHeight 動態 setFrameHeight(height 當下限),
                               讓 viewer 最大化(設計 §1.1 / M7a-AC7);False → 用傳入 height(M6 行為)。
    回傳值形狀不變(仍是最近一次事件 dict 或 None;hover 不回傳事件、不 round-trip)。"""
    return _component(image=image_url, rois=rois or [], tiles=tiles,
                      height=height, key=key,
                      meta=meta, dets=dets or [],
                      max_zoom_pixel_ratio=max_zoom_pixel_ratio,
                      restore_zoom=restore_zoom, restore_center=restore_center,
                      nav_keys=bool(nav_keys), auto_height=bool(auto_height),
                      default=None)

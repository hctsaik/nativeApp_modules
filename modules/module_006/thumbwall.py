"""縮圖牆的 Streamlit 包裝(宣告式元件,無 React build)。

設計:3_Architect_Design/19_viewer_ux.md §2.2 / §2.2a。
渲染一排可點縮圖(整張 <img> 可點)+ 角標(label)+ mark(⭐/✓)+ 偵測數徽章(nd>0)。
回傳被點的縮圖 index(int)或 None(本 run 無點擊)。

items: list[dict],每筆 = {
    "img":   str,   # 縮圖 data URL(已含框,由 app 用 imgio.to_data_url 產生)
    "label": str,   # 角標文字,如 "3"
    "mark":  str,   # ⭐/✓ 標記(可空)
    "nd":    int,   # 偵測框數(0 → 不顯示徽章)
}
selected: 目前選中的 index(高亮邊框用)。

回傳值(JS → Python):{"index": int, "n": int}(n 單調遞增、去重,語義同 viewer)。
app 端讀 index 設 ss.idx;用 n>last_thumb_n 判斷是否本次新點擊,避免 rerun 重複觸發。
"""
from pathlib import Path

import streamlit.components.v1 as components

_DIR = Path(__file__).parent / "thumbwall_component"
_component = components.declare_component("cv_thumbwall", path=str(_DIR))


def thumbwall(items, selected: int = 0, height: int = 620, key: str = "thumbwall",
              horizontal: bool = False):
    """渲染可點縮圖牆。回傳被點的縮圖 index(int)或 None(本 run 無點擊)。

    horizontal=True → 橫向可捲縮圖條(`overflow-x:auto` 原生水平捲軸,整列渲染全部縮圖、
    可快速捲到任一張);label 建議放檔名。預設 False = 縱向縮圖牆(向後相容,主縮圖牆/viewer_ux 不變)。"""
    ev = _component(items=items or [], selected=int(selected), height=int(height),
                    horizontal=bool(horizontal), key=key, default=None)
    if isinstance(ev, dict) and "index" in ev:
        idx = ev.get("index")
        n = ev.get("n", 0)
        last = thumbwall._last_n.get(key, 0)
        if n > last:
            thumbwall._last_n[key] = n
            try:
                return int(idx)
            except (TypeError, ValueError):
                return None
    return None


# 每個 key 的最近事件序號(去重用;模組層級狀態,避免 rerun 重複觸發)
thumbwall._last_n = {}

"""CV Review Workbench — 獨立 Streamlit 原型(M1…M7a)。
跑法:streamlit run 5_PG_Develop/app.py

M7a Viewer-First 版面重設計(設計 3_Architect_Design/20_viewer_workbench_redesign.md):
  - 頂列 Command Bar(全寬一行):上一張/下一張 + 跳頁 + 檔名/偵測N框 + 召喚入口鈕
  - Stage 三欄:左=縮圖牆(可收 0 寬)| 中=主 viewer + viewer-footer | 右=判定 Rail(可收 0 寬)
  - viewer-footer 薄條常駐:顯示偵測框 toggle + 信心門檻 slider + 顯示 k/n 框 + 最後點擊
  - 🧰工具台:framecompare/simhash/DZI/missedq/embcluster/匯出 全收進單一 expander 內 st.tabs
             (未展開 → 重函式 0 呼叫,設計 §效能 PerfB 基礎)
  - sidebar:篩選/搜尋/資料源/智慧排序 維持(預設收合)
  - 固定 key viewer(key="cv_viewer")修 remount;zoom/pan 跨切張保存(純 client 端)
  - P1 隱藏探針 data-render-ms / data-reruns / data-thumb-recalc / data-tool-calls

模組:imageset / imgio / sidecar / tagging / viewer / yolo / overlay / framecompare /
      filtersort / casepkg / framediff / missedq / simhash / embcluster / cocoio / dzitiles / htmlreport。
"""
import json
import os
import sys
import time
import tempfile
from pathlib import Path

# streamlit run 不會載 conftest,需自行把本目錄放到 sys.path 最前(讓 imgio 等本地模組優先)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st
from PIL import Image

_STRETCH = "stretch"  # st.* width 參數(取代已棄用的 width=_STRETCH)

import filtersort
import imageset
import imgio
import labelloc
import missedq
import modeldiff
import overlay
import sidecar
import tagging
import yolo
from thumbwall import thumbwall
from viewer import osd_viewer

STATE = os.path.join(os.environ.get("CIM_LOG_DIR", tempfile.gettempdir()), "module_006_cvr_state.json")

# ★ M7a §效能量測:本 run server 端 render 起點(結尾寫進 P1 探針 data-render-ms)

# 注意:sidebar 預設『展開』(不設 initial_sidebar_state="collapsed")—— 既有 M6 回歸測試
# test_sidebar_has_no_duplicate_overlay_toggle 會 wait_for(sidebar 可見)後才斷言『無重複開關』,
# 收合會使該 wait_for 逾時。設計 §1.1 的『收合 sidebar』為可選優化,不得犧牲既有契約測試;
# 使用者仍可手動收合(篩選/資料源/排序留在 sidebar,M7a 不改其行為)。

# Streamlit checkbox 的原生 <input> 預設 width/height=0、opacity=0(視覺由樣式化方塊取代),
# 導致精準點擊(含自動化點擊)落不到實際 input 上。給它一個真實的透明命中區(疊在視覺方塊上),
# 不改變外觀,只讓「點到框」更可靠。純展示層調整,不動任何功能行為。
@st.cache_resource
def _position():
    return imageset.Position(STATE)


@st.cache_data(show_spinner=False)
def _records(folder, sort_key):
    # imageset 的檔案系統排序(name/time/size);M3 的智慧排序另由 filtersort 處理
    return imageset.sort_records(imageset.scan(folder), key=sort_key)


@st.cache_data(show_spinner=False)
def _array(path):
    return imgio.load(path)["array"]


@st.cache_data(show_spinner=False)
def _meta(path):
    d = imgio.load(path)
    return d["width"], d["height"], d["bit_depth"], d["channels"]


@st.cache_data(show_spinner=False)
def _display_rgb(path):
    return imgio.to_display_rgb(_array(path))


# 類別 → 顏色:穩定調色盤(涵蓋任意類別名);主 viewer 元件與縮圖『共用同一函式』→ 兩處顏色一致。
# (取代 overlay.CLASS_COLORS 只有 3 類、其餘退紅 → User 真實 10 類資料才有分辨度且前後一致。)
_PALETTE = [(0, 200, 0), (0, 160, 255), (255, 90, 0), (255, 0, 170), (170, 0, 255),
            (255, 190, 0), (0, 210, 200), (255, 70, 70), (130, 200, 0), (0, 110, 255),
            (200, 120, 255), (255, 150, 40)]


def _cls_color(cls):
    """類別名 → 穩定 RGB(同名同色;主 viewer 與縮圖共用,顏色一致)。"""
    s = str(cls or "")
    return _PALETTE[sum(ord(c) for c in s) % len(_PALETTE)] if s else (255, 0, 0)


@st.cache_data(show_spinner=False)
def _thumb_cached(path, max_px, dets_key, show, conf_thr, classes_key):
    """實際算縮圖(+可選疊框)的 cache 函式;所有參數皆可雜湊。
    dets_key/classes_key 為投影成 tuple 的可雜湊鍵。
    ★ M7a §效能(P3):每次真算(cache miss 進到這裡)即一次 thumb_recalc。"""
    thumb = imgio.thumbnail(_array(path), max_px=max_px)
    if not show or not dets_key:
        return imgio.to_png_bytes(thumb)
    th, tw = thumb.shape[0], thumb.shape[1]
    ow, oh = _meta(path)[0], _meta(path)[1]
    sx = tw / float(ow) if ow else 1.0
    sy = th / float(oh) if oh else 1.0
    scaled = []
    for (cls, conf, bx, by, bw, bh) in dets_key:
        scaled.append({"cls": cls, "conf": conf,
                       "bbox": [round(bx * sx), round(by * sy),
                                max(1, round(bw * sx)), max(1, round(bh * sy))]})
    classes = list(classes_key) if classes_key else None
    kept_t = overlay.filter_detections(scaled, conf_threshold=conf_thr, classes=classes)
    # 依類別調色盤逐色畫(與主 viewer 元件 _cls_color 同色 → 縮圖/大圖顏色一致;overlay.draw 每次回新陣列可鏈式)。
    drawn = thumb
    by_color = {}
    for det in kept_t:
        by_color.setdefault(_cls_color(det.get("cls")), []).append(det)
    for col, group in by_color.items():
        drawn = overlay.draw(drawn, group, color=col, thickness=1, draw_label=False)
    return imgio.to_png_bytes(drawn)


def _dets_key(dets):
    """把 list[dict] 偵測投影成可雜湊 tuple,當 cache key(避免 unhashable 崩潰)。"""
    out = []
    for d in (dets or []):
        b = d.get("bbox", [0, 0, 0, 0])
        out.append((d.get("cls"), round(float(d.get("conf", 0.0)), 3),
                    int(b[0]), int(b[1]), int(b[2]), int(b[3])))
    return tuple(out)


def _thumb(path, max_px=130, dets=None, show=False, conf_thr=0.0, classes=None):
    """show=False 或 dets 空 → 純縮圖(向後相容);否則縮圖座標系疊『過濾後』框。
    回傳 PNG bytes。dets 投影成可雜湊 tuple 進 cache(避免 unhashable list 破 cache)。
    ★ M7a §效能(P3):以 cache 命中與否估 thumb_recalc(miss → +1)。"""
    dets_key = _dets_key(dets) if (show and dets) else ()
    classes_key = tuple(classes) if classes else ()
    args = (path, max_px, dets_key, bool(show and dets), conf_thr, classes_key)
    # cache miss 估算:同 args 第一次見 → 視為真算(+1 thumb_recalc),之後視為命中。
    # 與 st.cache_data(同 args→同結果)語義一致,供 §效能 PerfC「改門檻不重建整牆」量測。
    if args not in _thumb_seen:
        _counters["thumb_recalc"] += 1
        _thumb_seen[args] = True
    return _thumb_cached(*args)


# 進程級「已算過的縮圖 cache key」集合(估 thumb_recalc:同 key 二次視為命中,不再 +1)。
# 與 st.cache_data 命中語義一致(同 args → 同結果),供 §效能 PerfC「改門檻不重建整牆」量測。
_thumb_seen = {}

# 效能計數器:模組層級全域,讓模組層級的 _thumb() 可累加(原 app.py 即模組層級;
# 移植拆 render() 後仍須保留全域,否則 _thumb 會 NameError)。render() 每次進來 reset。
_counters = {"thumb_recalc": 0, "tool_calls": 0}


@st.cache_data(show_spinner=False)
def _main_url_cached(path, dets_key, show, conf_thr, classes_key):
    """主 viewer 影像 data URL(可選燒框)的 cache:keyed by (path, dets, show, conf, classes)。
    ★ M7a:把『overlay.draw 全解析度 + PNG 編碼 + base64』結果快取,讓信心門檻連改時相同 conf
    命中、不同 conf 也只算一次 → 縮短 rerun(footer slider 連按即時反映,M7a-AC5)。"""
    base = _display_rgb(path)
    if show and dets_key:
        dets = [{"cls": c, "conf": cf, "bbox": [bx, by, bw, bh]}
                for (c, cf, bx, by, bw, bh) in dets_key]
        classes = list(classes_key) if classes_key else None
        disp = overlay.draw(base, dets, thickness=2, conf_threshold=conf_thr,
                            classes=classes, draw_label=True)
    else:
        disp = base
    return imgio.to_data_url(disp)


def _main_url(path, dets, show, conf_thr, classes):
    dets_key = _dets_key(dets) if (show and dets) else ()
    classes_key = tuple(classes) if classes else ()
    return _main_url_cached(path, dets_key, bool(show and dets), conf_thr, classes_key)


@st.cache_data(show_spinner=False)
def _display_url(path):
    """主 viewer 用『純顯示影像』data URL(不燒框);偵測框改由元件畫成向量框+標籤(B)。"""
    return imgio.to_data_url(_display_rgb(path))


def _label_path(label_dir, image_path):
    """偵測檔慣例:<label_dir>/<stem>.json 優先,否則 .txt(YOLO);兩者皆無 → 回 .json 路徑(yolo.load 容錯回 [])。"""
    stem = Path(image_path).stem
    pj = str(Path(label_dir) / f"{stem}.json")
    if os.path.isfile(pj):
        return pj
    pt = str(Path(label_dir) / f"{stem}.txt")
    return pt if os.path.isfile(pt) else pj


def _label_exists(label_dir, image_path):
    """該圖在此標註夾是否有輸出(.json 或 .txt)—— 供 modeldiff 區分『缺檔』vs『有檔但 0 框』。"""
    stem = Path(image_path).stem
    return (os.path.isfile(Path(label_dir) / f"{stem}.json")
            or os.path.isfile(Path(label_dir) / f"{stem}.txt"))


@st.cache_data(show_spinner=False)
def _detections(pred_folder, image_path, w, h, names_key=()):
    """載入單張圖偵測(.json 或 YOLO .txt;容錯,檔不存在/壞檔 → [])。names_key=類別名 tuple(YOLO txt id→名)。"""
    return yolo.load(_label_path(pred_folder, image_path), img_w=w, img_h=h,
                     names=list(names_key) or None)


@st.cache_data(show_spinner=False)
def _names_from_yaml(path):
    """從 data.yaml 取 names(list 或 {id:name} dict)→ list[str];失敗 → []。"""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, list):
            return [str(x) for x in names]
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    except Exception:
        pass
    return []


@st.cache_data(show_spinner=False)
def _class_names(start_dir):
    """沿目錄上行(≤6 層)找 data.yaml(names:)或 classes.txt → list[str](YOLO .txt 的 id→名);找不到 → []。"""
    d = os.path.abspath(start_dir)
    for _ in range(6):
        for fn in ("data.yaml", "data.yml", "dataset.yaml"):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                ns = _names_from_yaml(p)
                if ns:
                    return ns
        for fn in ("classes.txt", "obj.names"):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        ns = [ln.strip() for ln in f if ln.strip()]
                    if ns:
                        return ns
                except OSError:
                    pass
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return []


@st.cache_data(show_spinner=False)
def _resolve_pred(folder, stems_key):
    """自動判定偵測標註目錄(labels/ 子夾 vs 同層);labelloc 純函式 → 同 key 同結果,cache 安全。
    stems_key = tuple(stems)(可雜湊)。回絕對路徑或 None(folder 非資料夾)。"""
    return labelloc.resolve_label_dir(folder, list(stems_key))


def _resize_to(arr, h, w):
    """把 RGB uint8 陣列縮放到 (h, w),供 framecompare 同形運算。"""
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    return np.asarray(Image.fromarray(arr).resize((w, h)), dtype=np.uint8)


@st.cache_data(show_spinner=False)
def _feat_vec(path):
    """簡易特徵向量(8×8 灰階 flatten,64 維)供 embcluster 示範;非 DINO/SAM。"""
    g = np.asarray(Image.fromarray(_display_rgb(path)).convert("L").resize((8, 8)), dtype=np.float64)
    return g.flatten().tolist()


@st.cache_data(show_spinner=False)
def _dzi_tiles(path):
    """把圖建成 DZI 金字塔,瓦片轉 data url 供 OSD 自訂 tile source(只在可視區被取)。"""
    import base64
    bt = dzitiles.build_tiles(_display_rgb(path), tile_size=254, overlap=1)
    tiles = {lvl: {k: "data:image/png;base64," + base64.b64encode(v).decode("ascii")
                   for k, v in d.items()} for lvl, d in bt["tiles"].items()}
    return {"width": bt["width"], "height": bt["height"], "tile_size": bt["tile_size"],
            "overlap": bt["overlap"], "max_level": bt["max_level"], "num_levels": bt["num_levels"],
            "tiles": tiles}



def render(folder, model_b_override=""):
    """Render the full CV_Viewer workbench for the given folders (Output page).
    folder = 主要資料夾(影像 + Model A 標註); model_b_override = 對比資料夾(Model B;空=不比較)。"""
    global _counters  # 每 run reset 模組層級計數器(_thumb 會讀它)
    _T0 = time.perf_counter()
    model_b_override = (model_b_override or "").strip()
    st.markdown(
        "<style>[data-testid='stCheckbox'] input[type='checkbox']{"
        "width:1em !important;height:1em !important;opacity:0 !important;cursor:pointer;}"
        "/* User 回饋:拿掉頂部空白 —— 收掉 Streamlit 預設 header 列 + 緊湊化 block-container 上緣 */"
        "header[data-testid='stHeader']{height:0 !important;min-height:0 !important;}"
        "[data-testid='stDecoration']{display:none !important;}"
        "[data-testid='stToolbar']{display:none !important;}"
        "/* 頂列工具列專業排版:標籤/按鈕不換行(避免『顯示偵測\\n框』『可撤 0\\n筆』醜換行)*/"
        "[data-testid='stCheckbox'] label{white-space:nowrap !important;}"
        ".stButton button{white-space:nowrap !important;}"
        "/* User 回饋:標題上方一大塊空白 = 主內容容器預設上 padding(新版 Streamlit 用 .block-container/"
        "   stMainBlockContainer,舊選擇器 section.main 不一定命中)→ 用 class + testid 雙保險收到接近 0 */"
        ".block-container,[data-testid='stMainBlockContainer'],section.main div.block-container"
        "{padding-top:0.4rem !important;padding-bottom:0.5rem !important;}"
        "/* 隱藏只放 <style> 的空 markdown 容器(本身佔一行高,造成標題上方留白)*/"
        "[data-testid='stMarkdownContainer']:has(> style){display:none !important;}"
        "/* 標題上緣不再額外留白 */"
        ".block-container > div:first-child{margin-top:0 !important;}"
        "</style>",
        unsafe_allow_html=True,
    )

    ss = st.session_state

    # ============================== M7a session_state 旗標(全域持久,跨圖不重置)==============================
    ss.setdefault("thumb_collapsed", False)  # 縮圖牆收合(0 寬)
    ss.setdefault("tool_open", False)        # 🧰工具台 expander 展開旗標(供 §效能 PerfB:未開→重函式 0 呼叫)
    ss.setdefault("compare_on", False)       # 🔀 比較模式(雙區塊:各自選圖 + 差異/混合;設計 23_compare.md)
    ss.setdefault("undo_stack", [])          # ★ M7b 跨圖 undo 軌跡({path,idx,field,old,new},LIFO,留近 5;設計 §3.6)
    ss.setdefault("reruns", 0)               # 累積 rerun 計數(P1 探針 data-reruns)
    ss.reruns += 1
    # 本 run 的效能計數器(P3:重函式呼叫計數,寫進 P1 探針)。每 run 歸零後累加。
    _counters = {"thumb_recalc": 0, "tool_calls": 0}

    # ★ M7b:flush 上一個 run(緊接 st.rerun() 而無法當場顯示)留下的 toast 訊息。
    _pending_toast = ss.pop("_pending_toast", None)
    if _pending_toast:
        st.toast(_pending_toast)


    # ============================== 快取資料層 ==============================
    fs_sort = "name"
    f_tags, f_tag_mode, f_verdict, f_reviewed = [], "any", "(全部)", "(全部)"
    f_book, f_text, f_class = False, "", ""
    smart, smart_rev, use_queue = "(關閉)", False, False


    if not os.path.isdir(folder):
        st.error(f"找不到資料夾:{folder}")
        st.stop()

    # YOLO 切分佈局:選到的夾若無『直接影像』但有 images/ 子夾(典型 <split>/images + <split>/labels)→ 改掃 images/。
    img_dir = folder
    if not _records(folder, fs_sort) and os.path.isdir(os.path.join(folder, "images")):
        img_dir = os.path.join(folder, "images")
    records = _records(img_dir, fs_sort)
    if not records:
        st.warning("這個資料夾沒有支援的影像(支援 .png/.jpg/.tif…;YOLO 資料集請選含 images/ 的那層,或 images/ 本身)。")
        st.stop()

    # 類別名:沿目錄上行找 data.yaml(names:)/classes.txt → YOLO .txt 的 id→名;找不到則框上顯示數字 id。
    class_names = _class_names(img_dir)
    _names_key = tuple(class_names)

    # ---- Model A 標註目錄 ----
    # YOLO 切分:影像在 <X>/images → 標註在 sibling <X>/labels;否則 labelloc(同層 / labels 子夾)。
    _pred_stems = tuple(Path(r["name"]).stem for r in records)
    _img_norm = os.path.normpath(img_dir)
    _sibling_labels = os.path.join(os.path.dirname(_img_norm), "labels")
    if os.path.basename(_img_norm).lower() == "images" and os.path.isdir(_sibling_labels):
        pred_folder = _sibling_labels
        _pred_caption = "Model A 標註於 sibling labels/(YOLO 切分)"
    else:
        _resolved_pred = _resolve_pred(img_dir, _pred_stems)
        pred_folder = _resolved_pred or img_dir
        if _resolved_pred and os.path.normcase(os.path.abspath(_resolved_pred)) != \
                os.path.normcase(os.path.abspath(img_dir)):
            _pred_caption = f"Model A 標註於子資料夾 {os.path.basename(_resolved_pred)}/"
        else:
            _pred_caption = "Model A 標註於同影像資料夾"
    if class_names:
        _pred_caption += f" · 類別名 {len(class_names)} 個已載入"
    st.sidebar.caption(_pred_caption)

    # 比較模式第二個 model 的標註夾(須與 model A 對同一影像集);留空 = 不進行雙 model 比對。
    model_b_folder = model_b_override.strip() or None


    # ============================== 組裝 items(含 sidecar / detections) ==============================
    def _item_for(r):
        w, h, _bit, _ch = _meta(r["path"])
        sc = sidecar.load(r["path"])
        dets = _detections(pred_folder, r["path"], w, h, _names_key)
        a_present = _label_exists(pred_folder, r["path"])
        # 第二個 model(比較模式):同一張圖讀 model B 的偵測 + 記錄『B 是否有輸出此圖』。
        # 缺檔 vs 有檔但 0 框 → modeldiff 分成 missing_b / both_empty(防打錯路徑假冒覆蓋差異)。
        dets_b, b_present = [], False
        if model_b_folder:
            dets_b = _detections(model_b_folder, r["path"], w, h, _names_key)
            b_present = _label_exists(model_b_folder, r["path"])
        return {"name": r["name"], "path": r["path"], "time": float(r.get("mtime", 0.0)),
                "sidecar": sc, "detections": dets, "detections_b": dets_b,
                "a_present": a_present, "b_present": b_present, "record": r, "w": w, "h": h}


    need_class = bool(f_class.strip())  # 類別篩選需要載入每張圖的偵測
    items = [_item_for(r) for r in records]

    # ---- 套用篩選 ----
    query = {}
    if f_tags:
        query["tags"] = f_tags
        query["mode"] = f_tag_mode
    if f_verdict != "(全部)":
        query["verdict"] = f_verdict
    if f_reviewed == "已 review":
        query["reviewed"] = True
    elif f_reviewed == "未 review":
        query["reviewed"] = False
    if f_book:
        query["bookmarked"] = True
    if f_text.strip():
        query["text"] = f_text.strip()


    def _passes(it):
        if query and not tagging.matches(it["sidecar"], query):
            return False
        if need_class:
            cls_set = {d["cls"] for d in it["detections"]}
            if f_class.strip() not in cls_set:
                return False
        return True


    shown_items = [it for it in items if _passes(it)]
    if not shown_items:
        st.warning("沒有符合篩選條件的影像。")
        st.stop()

    # ---- 排序:User 主排序(by 檔名 / by 信心,可見控制,單張與比較共用)+ 進階(智慧排序鍵 / Review Queue,sidebar)----
    # 複用 filtersort.sort_items(key="name"/"conf";conf=該圖最大偵測信心,tie-break name)。
    _sort_mode = ss.get("sort_mode", "檔名")
    if use_queue:
        shown_items = filtersort.review_queue(shown_items)
    elif smart != "(關閉)":
        shown_items = filtersort.sort_items(shown_items, smart, reverse=smart_rev)
    elif _sort_mode == "信心(高→低)":
        shown_items = filtersort.sort_items(shown_items, "conf", reverse=True)
    else:
        shown_items = filtersort.sort_items(shown_items, "name")


    # ============================== 索引 + 位置記憶 ==============================
    pos = _position()
    nav_sig = (folder, len(shown_items))
    if ss.get("nav_sig") != nav_sig:
        ss.nav_sig = nav_sig
        ss.idx = min(pos.get(folder, 0), len(shown_items) - 1)
        ss.last_n = 0
        ss.last_click = None
    ss.idx = max(0, min(ss.get("idx", 0), len(shown_items) - 1))
    total = len(shown_items)

    # 偵測框總開關 + 信心門檻:M7a 由 viewer-footer 控制(footer 在三欄之內、於下方渲染)。
    # 縮圖牆(左欄)在 footer 之前渲染,需先有值 → 用 footer widget 的 session_state key 當單一真相
    # (Streamlit widget 以其 key 跨 rerun 持久;使用者改 footer → 自動 rerun → 本輪讀到新值即時反映)。
    # User:移除『顯示偵測框』開關(無實質意義)→ 偵測框恆顯示,改由『信心門檻』過濾。
    # 保留 show_overlay 名稱供既有讀取點(縮圖牆 show=show_overlay 等),恆 True。
    show_overlay = True
    conf_thr = ss.get("footer_conf_thr", 0.25)

    # 漏檢 queue(恆算極輕:只看 sidecar/detections,不碰重函式)→ 供 Command Bar 徽章 N
    _mq = missedq.missed_queue(shown_items)
    _name2idx = {it["name"]: i for i, it in enumerate(shown_items)}
    _path2idx = {it["path"]: i for i, it in enumerate(shown_items)}  # ★ M7b undo 跳轉用


    # ============================== M7b 鍵盤工作流:nav 事件處理 + 改值即時存 + 跨圖 undo ==============================
    _STATUS_CYCLE = ["none", "need_review", "done"]
    # verdict/狀態 仍同步到工具台「✏️ 標記」分頁的 widget(那些 widget 在本檔較後渲染);
    # bookmark 已移到『頂列 button』(在 nav 處理之前渲染,且 button 無持久 widget-state)→ 不走 widget 同步,
    # 只讀寫 sidecar 當單一真相(避免『widget 實例化後再改 session_state』的 StreamlitAPIException)。
    _WIDGET_PREFIX = {"verdict": "vd", "review_status": "stt"}


    def _sync_rail_widget(idx, field, value):
        """把改動同步到 Rail widget 的 session_state key(單一真相:熱鍵改值後 Rail 顯示一致;AC10)。
        在 Rail widget 於本 run 被建立『之前』設定才合法;nav 處理在中欄、Rail 在右欄之後 + 立即 rerun,故安全。"""
        wp = _WIDGET_PREFIX.get(field)
        if wp:
            ss[f"{wp}_{idx}"] = value


    def _push_change(path, idx, field, old, new):
        """改值即時寫 sidecar + 推入 undo_stack(留近 5)+ 同步 Rail widget。old==new 不記(避免空轉)。"""
        if old == new:
            return
        sc = sidecar.load(path)
        d = sidecar.default()
        d.update(sc)
        d[field] = new
        sidecar.save(path, d)
        ss.undo_stack.append({"path": path, "idx": idx, "field": field, "old": old, "new": new})
        del ss.undo_stack[:-5]  # 只留近 5 筆(設計 §3.6)
        _sync_rail_widget(idx, field, new)


    def _handle_nav(ev, cur, total):
        """處理 viewer 元件鍵盤 nav 事件(設計 §3.4 鍵位表 / §3.5 autosave / §3.6 undo)。"""
        action = ev.get("action")
        value = ev.get("value")
        if action == "next":
            ss.idx = min(total - 1, ss.idx + 1)
            st.rerun()
        elif action == "prev":
            ss.idx = max(0, ss.idx - 1)
            st.rerun()
        elif action == "verdict":
            sc = sidecar.load(cur["path"])
            if value in tagging.VERDICTS:
                _push_change(cur["path"], ss.idx, "verdict", sc.get("verdict", "unset"), value)
            st.rerun()
        elif action == "status":
            sc = sidecar.load(cur["path"])
            old = sc.get("review_status", "none")
            nxt = (_STATUS_CYCLE[(_STATUS_CYCLE.index(old) + 1) % len(_STATUS_CYCLE)]
                   if old in _STATUS_CYCLE else "need_review")
            _push_change(cur["path"], ss.idx, "review_status", old, nxt)
            st.rerun()
        elif action == "bookmark":
            sc = sidecar.load(cur["path"])
            old = bool(sc.get("bookmarked", False))
            _push_change(cur["path"], ss.idx, "bookmarked", old, (not old))
            st.rerun()
        elif action == "undo":
            if ss.undo_stack:
                e = ss.undo_stack.pop()
                tgt = max(0, min(total - 1, _path2idx.get(e["path"], e["idx"])))
                sc = sidecar.load(e["path"])
                d = sidecar.default()
                d.update(sc)
                d[e["field"]] = e["old"]
                sidecar.save(e["path"], d)
                _sync_rail_widget(tgt, e["field"], e["old"])
                ss.idx = tgt  # 撤銷會跳轉(設計 §3.6,刻意)
                # toast 用 pending 模式:本 run 緊接 st.rerun() 會在 toast 提交前撕掉本 run,
                # 故把訊息存進 session_state,下一個『正常完成』的 run 開頭再 st.toast 顯示。
                ss["_pending_toast"] = f"已撤銷:{Path(e['path']).name} {e['field']} → {e['old']}"
            else:
                ss["_pending_toast"] = "無可撤銷"
            st.rerun()


    # ============================== 頂列 Command Bar(導覽 + 控制)==============================
    # 控制只剩:導覽(⟵ ⟶ 跳頁)+ ⭐書籤 + 信心門檻 slider + Object 類別 下拉。
    # User 本輪回饋:移除信心門檻旁 [−][＋] 鈕、移除「顯示k/n·可撤N」文字;新增『Object 類別』下拉(放信心門檻旁,單選/全部)。
    # 偵測框恆顯示,由『信心門檻 + Object 類別』過濾(顯示框數/可撤筆數改由 P1 探針 data-shown-k/n、data-undo-n 機器讀)。
    _USER_MANUAL = """### 📖 YOLO Image Viewer — 使用手冊

    **載入資料(左側『資料來源』)**
    - **影像資料夾**:選含影像的夾;支援 YOLO 切分佈局(自動偵測 `images/` 子夾)。Model A 標註自動找同層或 `labels/` 子夾。
    - **第二個 model 資料夾 B**:比較模式用(留空=不比較);兩夾須對同一包影像。
    - 兩欄都可按 **📁** 用選的(原生資料夾對話框)。
    - 標註支援 **YOLO `.txt`** 與 **`.json`**;類別名自動讀 `data.yaml` / `classes.txt`。

    **檢視 / 導覽**
    - 上一張 / 下一張 / 跳頁;鍵盤 **←/→** 切張。
    - 主圖:**滾輪縮放**、拖曳平移、**Shift+拖** 框 ROI、**點擊** 取像素值。
    - 偵測框**恆顯示**(類別色 + `類別 信心` 標籤);**信心門檻** 過濾;**Object 類別** 下拉選一種或全部。
    - **排序**(縮圖牆上方):by 檔名 / by 信心(高→低)。

    **標記(鍵盤)**
    - **1 / 2 / 3** = 判定 true_defect / false_alarm / reflection
    - **r** = 切換 review 狀態 · **b** / 空白 = 書籤 · **u** = 復原

    **比較模式(🔀 雙 model 覆蓋比對)**
    - 覆蓋率儀表板 + 分歧 triage 佇列 + 下鑽雙色疊框(**A 藍 / B 橘**)。
    - 控制:看哪種差異(只A / 只B / 不一致 / 缺檔…)、信心範圍(雙界)、Object 類別、排序。
    - 『B 缺檔』= model B 沒輸出該圖(與『有檔但 0 框』區分,避免假覆蓋差異)。
    """
    # 用 st.dialog(modal)而非 popover:dialog 內容只在『開啟時』才進 DOM,關閉時不存在 →
    # 手冊內提到的控制名(如『信心門檻』粗體)不會被 E2E 的 get_by_text 誤命中為隱藏元素(實測 popover 會)。
    @st.dialog("📖 YOLO Image Viewer — 使用手冊", width="large")
    def _show_manual():
        st.markdown(_USER_MANUAL)


    # 標題 + 緊鄰的小型『❓ 使用手冊』文字鈕(type='tertiary' = 無框、像一個小字,不撐滿欄寬;
    # 標題欄取窄比例讓小字緊貼標題右側)。
    _tcol = st.columns([0.26, 0.74], gap="small", vertical_alignment="center")
    _tcol[0].markdown("##### 🖼️ YOLO Image Viewer")
    if _tcol[1].button("❓ 使用手冊", type="tertiary", key="manual_btn"):
        _show_manual()
    # Object 下拉可選類別 = 全部 shown_items 偵測的 cls 聯集(穩定;跨切張不變)。
    _all_classes = sorted({d.get("cls", "") for it in shown_items
                           for d in it["detections"] if d.get("cls")})
    # 信心門檻 slider 欄需夠寬(過窄會使鍵盤 ArrowRight 微調不可靠 — 見 ROADMAP 2026-06-26)→ 給 3.2。
    bar = st.columns([0.8, 0.8, 0.45, 0.4, 3.2, 1.35], vertical_alignment="center")
    if bar[0].button("⟵ 上一張", width=_STRETCH):
        ss.idx = max(0, ss.idx - 1)
        st.rerun()
    if bar[1].button("下一張 ⟶", width=_STRETCH):
        ss.idx = min(total - 1, ss.idx + 1)
        st.rerun()
    if total > 1:
        jump = bar[2].number_input("跳到第幾張", 1, total, ss.idx + 1, label_visibility="collapsed")
        if jump - 1 != ss.idx:
            ss.idx = jump - 1
            st.rerun()
    pos.set(folder, ss.idx)
    cur = shown_items[ss.idx]
    _cur_sc = sidecar.load(cur["path"])
    # ⭐ Bookmark(頂列 toggle button,讀寫 sidecar);與鍵盤 b/空白鍵走同一 _push_change 路徑(入 undo)。
    _bk_on = bool(_cur_sc.get("bookmarked", False))
    if bar[3].button("⭐" if _bk_on else "☆", key="bk_btn", width=_STRETCH,
                     help="Bookmark(熱鍵 b / 空白鍵)"):
        _push_change(cur["path"], ss.idx, "bookmarked", _bk_on, (not _bk_on))
        st.rerun()
    # 信心門檻 slider(User:移除旁邊 [−][＋] 鈕,只留滑桿)。
    bar[4].slider("信心門檻", 0.0, 1.0, conf_thr, 0.01, key="footer_conf_thr")
    # Object 類別 下拉(User:放信心門檻旁;選『全部』或單一類別 → 只畫該類別框)。
    # 跨資料集切換時,清掉已不在選項內的舊選值(widget 實例化前改 state 才合法)。
    if ss.get("cls_filter") not in (["全部"] + _all_classes):
        ss["cls_filter"] = "全部"
    _cls_sel = bar[5].selectbox("Object 類別", ["全部"] + _all_classes, key="cls_filter")
    overlay_classes = None if _cls_sel == "全部" else [_cls_sel]
    # kept(過濾後偵測):偵測框恆顯示,由信心門檻 + Object 類別過濾;主 viewer dets 與 P1 探針 data-shown-k 共用。
    kept = overlay.filter_detections(cur["detections"], conf_threshold=conf_thr,
                                     classes=overlay_classes)


    # ============================== 🔀 比較模式(雙 model 覆蓋比對;設計 24_modeldiff.md + 23_compare.md §8)==============================
    # User 第三輪裁決:A/B = 兩個 model 對【同一包影像】的結果;filter 濾【整個影像集】(triage,非單圖框);
    # IoU 框級配對(matched/only-A/only-B);主視圖 = 覆蓋率儀表板 + 分歧 triage 佇列(橫向縮圖條)+ 下鑽雙色疊框。
    # 取代舊『兩張圖 diff/混合』(那是先前對需求的誤解;framecompare/framediff 模組保留、未在此用)。
    _CMP_STATUS = {
        "missing_a": ("🟥", "A 缺檔"), "missing_b": ("🟥", "B 缺檔"), "missing_both": ("🟥", "兩者缺檔"),
        "a_only": ("🔵", "只 A 逮到(B 漏)"), "b_only": ("🟠", "只 B 逮到(A 漏)"),
        "disagree": ("🟡", "兩者各有漏"), "agree": ("🟢", "一致"), "both_empty": ("⚪", "兩者皆無框"),
    }
    _CMP_MODES = [("disagree", "有分歧(只A/只B/各有漏)"), ("a_only", "只 A 逮到(B 漏)"),
                  ("b_only", "只 B 逮到(A 漏)"), ("missing", "B 缺檔(打錯路徑/沒輸出)"),
                  ("agree", "兩者一致"), ("all", "全部")]


    def _cmp_filter(dets, lo, hi, classes):
        """conf 雙界 + 類別過濾(沿 §7.2 app inline 慣例;不擴 overlay 單下界契約)。"""
        return [d for d in dets if lo <= float(d.get("conf", 0.0)) <= hi
                and (not classes or d.get("cls") in classes)]


    def _render_compare():
        """雙 model 覆蓋比對(設計 24_modeldiff.md):覆蓋率儀表板 + 分歧 triage 佇列(橫向縮圖條)+ 下鑽雙色疊框。
        A=model A(主來源,藍)/ B=model B(第二夾,橘);信心/類別/分歧 filter 對【整個影像集】做 triage。"""
        import base64 as _b64
        if not model_b_folder:
            st.info("🔀 **雙 model 覆蓋比對**:請在左側『第二個 model 結果資料夾 B』填入 model B 的標註夾"
                    "(與 model A 對同一包影像),即可比對兩 model 在整包資料上的覆蓋差異(誰逮到、誰漏掉)。")
            return

        # --- 控制列:看哪種差異 + 信心區間(雙界)+ 類別(對整個影像集 triage)---
        # User:移除『IoU 配對門檻』滑桿(無實質意義)→ 固定 0.5(框級配對仍用 IoU,只是不暴露為控制)。
        iou_thr = 0.5
        c = st.columns([1.6, 1.6, 1.3, 1.3])
        mode = c[0].selectbox("看哪種差異", [m[0] for m in _CMP_MODES],
                              format_func=lambda k: dict(_CMP_MODES)[k], key="cmp_mode")
        lo, hi = c[1].slider("信心範圍(下界–上界)", 0.0, 1.0, (0.0, 1.0), 0.01, key="cmp_conf")
        c[3].selectbox("排序", ["檔名", "信心(高→低)"], key="sort_mode",
                       help="分歧佇列順序:by 檔名 或 by 信心(高→低)")
        all_cls = sorted({d.get("cls", "") for it in shown_items
                          for d in (it["detections"] + it.get("detections_b", [])) if d.get("cls")})
        cck = "cmp_cls"
        if cck in ss:  # 去掉已不在選項中的舊選類別(widget 實例化前改 state 合法)
            ss[cck] = [x for x in ss[cck] if x in all_cls]
        sel_cls = c[2].multiselect("Object 類別(空=全部)", all_cls, key=cck) or None

        # --- 對【整個影像集】算每張覆蓋差異(IoU 框級配對)---
        recs = []
        for it in shown_items:
            dd = modeldiff.diff_image(it["detections"], it.get("detections_b", []),
                                      iou_thr=iou_thr, conf_range=(lo, hi), classes=sel_cls,
                                      a_present=it.get("a_present", True),
                                      b_present=it.get("b_present", False))
            dd["name"] = it["name"]
            dd["_it"] = it
            recs.append(dd)
        summary = modeldiff.summarize(recs)
        # 佇列順序:User 排序(by 檔名 / by 信心,與單張共用 sort_mode);取代原 disagreement-priority 排序。
        _filtered = modeldiff.filter_images(recs, mode)
        if _sort_mode == "信心(高→低)":
            triaged = sorted(_filtered, key=lambda r: -max(
                (float(d.get("conf", 0.0)) for d in r["_it"]["detections"]), default=0.0))
        else:
            triaged = sorted(_filtered, key=lambda r: r["name"])

        # --- 覆蓋率儀表板(回答『弱 model 逮的更少』)---
        st.markdown(
            "<div style='font-size:0.95rem;line-height:1.7;'>"
            f"📊 <b>覆蓋率彙總</b>　·　🔵 A:{summary['total_a']} 框 / {summary['imgs_a']} 張　·　"
            f"🟠 B:{summary['total_b']} 框 / {summary['imgs_b']} 張　·　"
            f"<b>B 比 A 少 {summary['delta_imgs']} 張 · 少 {summary['delta_boxes']} 框</b>　·　"
            f"只A逮到 {summary['total_only_a']} 框 · 只B逮到 {summary['total_only_b']} 框　·　"
            f"B 缺檔 {summary['n_missing_b']} 張</div>",
            unsafe_allow_html=True)
        st.caption(f"分歧 triage:此條件下 **{len(triaged)} / {len(recs)}** 張符合"
                   f"（{dict(_CMP_MODES)[mode]};信心 {lo:.2f}–{hi:.2f};IoU≥{iou_thr:.2f}）")

        # --- 分歧 triage 佇列(橫向縮圖條;分歧/缺檔優先排序;整圖可點下鑽)---
        if triaged:
            tw = []
            for r in triaged:
                it = r["_it"]
                png = _thumb(it["path"], dets=it["detections"], show=True, conf_thr=0.0,
                             classes=sel_cls, max_px=104)
                emoji = _CMP_STATUS.get(r["status"], ("", ""))[0]
                tw.append({"img": "data:image/png;base64," + _b64.b64encode(png).decode("ascii"),
                           "label": f"{emoji}{it['name']}", "mark": emoji,
                           "nd": r["only_a"] + r["only_b"]})
            names = [r["name"] for r in triaged]
            cur = ss.get("cmp_sel")
            sidx = names.index(cur) if cur in names else 0
            clicked = thumbwall(tw, selected=sidx, height=150, key="cmp_qwall", horizontal=True)
            if clicked is not None and 0 <= clicked < len(triaged):
                ss.cmp_sel = triaged[clicked]["name"]
                st.rerun()
        else:
            st.info("此條件下沒有符合的圖（試著把『看哪種差異』改成『全部』或放寬信心範圍）。")

        # --- 下鑽:選中圖 雙色疊框(model A=藍 / model B=橘),caption 列出配對/只A/只B ---
        drill = next((r for r in triaged if r["name"] == ss.get("cmp_sel")),
                     triaged[0] if triaged else None)
        if drill:
            it = drill["_it"]
            ka = _cmp_filter(it["detections"], lo, hi, sel_cls)
            kb = _cmp_filter(it.get("detections_b", []), lo, hi, sel_cls)
            img = overlay.draw(_display_rgb(it["path"]), ka, color=(0, 160, 255),
                               thickness=2, draw_label=True)                        # model A = 藍
            img = overlay.draw(img, kb, color=(255, 90, 0), thickness=2, draw_label=True)  # model B = 橘
            emoji, lbl = _CMP_STATUS.get(drill["status"], ("", drill["status"]))
            st.image(img, width=_STRETCH, caption=(
                f"{emoji} {it['name']} — 🔵A:{drill['n_a']} 框 · 🟠B:{drill['n_b']} 框 · "
                f"配對 {drill['matched']} · 只A {drill['only_a']} · 只B {drill['only_b']} · {lbl}"))

        # --- 回寫 compare 統計供 P1 探針(穩定數字,E2E 機器讀;設計 §5 P1 機器讀面)---
        ss["_cmp_probe"] = {
            "queue_n": len(triaged), "only_a": summary["total_only_a"],
            "only_b": summary["total_only_b"], "missing_b": summary["n_missing_b"],
            "delta_imgs": summary["delta_imgs"], "delta_boxes": summary["delta_boxes"]}


    # 🔀 比較模式入口 toggle(在 stage 之前渲染 → ss.compare_on 在中欄渲染前已知)。標籤含『比較模式』供 E2E 命中。
    st.toggle("🔀 比較模式（雙 model 覆蓋比對）", key="compare_on")


    # ============================== Stage 兩欄:縮圖牆 | 主 viewer / 比較區塊 ==============================
    # 縮圖牆收合旗標控制欄寬(收成 0 寬把寬度讓回 viewer);旗標跨 rerun/跨圖持久(M7a-AC4)。
    # 比較模式:隱藏主縮圖牆(由 A/B 區塊各自的橫向縮圖條取代主導覽),整個 stage 讓給雙區塊(設計 23 §7)。
    _compare = ss.get("compare_on", False)
    _left_w = 0.0001 if (ss.thumb_collapsed or _compare) else 0.85
    left, center = st.columns([_left_w, 6.6])

    # -------- 左欄:縮圖牆(可收 0 寬;比較模式不渲染)--------
    with left:
        if not _compare:
            # 收合 toggle(名稱含『縮圖』+收合語義,供 M7a-AC4 定位)
            _lbl = "▸ 展開縮圖" if ss.thumb_collapsed else "◂ 收合縮圖"
            if st.button(_lbl, key="toggle_thumb", width=_STRETCH):
                ss.thumb_collapsed = not ss.thumb_collapsed
                st.rerun()
            if not ss.thumb_collapsed:
                # 排序(User:by 檔名 / by 信心高→低;單張與比較共用 sort_mode key,兩模式互斥不衝突)。
                # (『縮圖牆』標題文字依 User 回饋移除。)
                st.selectbox("排序", ["檔名", "信心(高→低)"], key="sort_mode",
                             help="縮圖牆與導覽順序:by 檔名 或 by 信心(高→低)")
                import base64 as _b64
                # ★ M7a §效能(PerfC 基礎 + 連改門檻不卡):縮圖牆的『燒框』只跟『顯示偵測框開關』走,
                #   不跟『信心門檻 slider』的每一格走 —— 否則每按一格 slider 都觸發縮圖牆 iframe 重渲染
                #   round-trip,serialize WebSocket,使連按 slider 的 widget 更新被丟棄(M7a-AC5 連改即時失效)。
                #   縮圖以固定門檻 0.0 燒『全部偵測框』(開關 on 時),主 viewer 的 k 計數才是 conf 即時反映處。
                #   on/off 切換仍即時改縮圖 src(AC15/AC18);conf 改動只重算主 viewer,不重建整牆。
                _TW_CONF = 0.0
                tw_items = []
                for i, it in enumerate(shown_items):
                    s = it["sidecar"]
                    mark = ("⭐" if s.get("bookmarked") else "") + ("✓" if tagging.is_reviewed(s) else "")
                    nd = len(it["detections"])
                    png = _thumb(it["path"], dets=it["detections"], show=show_overlay,
                                 conf_thr=_TW_CONF, classes=overlay_classes)
                    img_url = "data:image/png;base64," + _b64.b64encode(png).decode("ascii")
                    tw_items.append({"img": img_url, "label": str(i + 1), "mark": mark, "nd": nd})
                clicked = thumbwall(tw_items, selected=ss.idx, height=620, key="thumbwall")
                if clicked is not None and clicked != ss.idx and 0 <= clicked < total:
                    ss.idx = clicked
                    st.rerun()

    # -------- 中欄:比較模式 → 雙區塊;否則 → 主 viewer + viewer-footer --------
    with center:
        if ss.compare_on:
            _render_compare()
        else:
            w, h, bit, ch = cur["w"], cur["h"], *_meta(cur["path"])[2:]
            # 主 viewer:影像『不燒框』(url=純顯示圖);偵測框由元件畫成『類別色向量框 + cls·conf 文字標籤』——
            #   與縮圖同 _cls_color 色盤(顏色一致)、清晰不隨縮放糊化、且顯示類型/信心(User 回饋 B)。
            url = _display_url(cur["path"])
            sc = sidecar.load(cur["path"])
            rois_draw = [{"bbox": rr["bbox"], "label": rr.get("label", "")} for rr in sc.get("rois", [])]
            # dets 帶 color(類別色,與縮圖一致)+ cls/conf(元件畫框+文字標籤);kept 已於 Command Bar 算好。
            dets_draw = [{"bbox": d["bbox"], "cls": d.get("cls", ""),
                          "conf": float(d.get("conf", 0.0)), "color": list(_cls_color(d.get("cls")))}
                         for d in kept]
            # ★ M7a 固定 key(不含 idx)修 remount + auto_height 最大化;切圖只靠 args.image 改變。
            # ★ M7b:nav_keys=True 啟用鍵盤工作流(←/→/1/2/3/r/b/u)。
            ev = osd_viewer(url, rois=rois_draw, height=720, key="cv_viewer",
                            meta={"name": cur["name"], "idx1": ss.idx + 1, "total": total,
                                  "w": w, "h": h, "bit": bit, "channels": ch},
                            dets=dets_draw, auto_height=True, nav_keys=True)
            if isinstance(ev, dict) and ev.get("n", 0) > ss.get("last_n", 0):
                ss.last_n = ev["n"]
                if ev.get("type") == "click":
                    try:
                        val = imgio.value_at(_array(cur["path"]), ev["x"], ev["y"])
                    except Exception:
                        val = "(界外)"
                    ss.last_click = {"x": ev["x"], "y": ev["y"], "val": val, "zoom": ev.get("zoom")}
                elif ev.get("type") == "roi":
                    sidecar.save(cur["path"], sidecar.add_roi(sc, ev["bbox"]))
                    st.rerun()
                elif ev.get("type") == "nav":
                    _handle_nav(ev, cur, total)  # 內含對應 st.rerun()

            # 16-bit 影像保留一行語義界線(hover=顯示值、點擊=真值)—— 誠實物理界線(設計 §6);8-bit 不顯示。
            if bit > 8:
                st.caption("ℹ️ 16-bit 影像:hover RGB = 顯示值(8-bit 顯示);點擊取真值")
            # 點擊後的像素值資訊(僅在點擊時出現,平時不佔位)。
            lc = ss.get("last_click")
            if lc:
                st.info(f"📍 ({lc['x']}, {lc['y']}) → 像素值 **{lc['val']}**　(zoom {lc['zoom']}×)")

    # ============================== 比較模式說明 ==============================
    # User 裁決:工具台其餘 tab(標記/相似/聚類/DZI/漏檢/匯出)移除,只留比較;比較改為頂部『🔀 比較模式』
    # toggle 進入的『中欄雙區塊』(見上方 _render_compare / 設計 23_compare.md)。底層模組保留、未來可重接。


    # ============================== P1 隱藏效能探針(主文件 DOM;Playwright 讀 data-*)==============================
    # 設計 §5 效能量測機制 (P1):render 計時 + rerun 計數 + thumb 重算 + 工具台重函式呼叫。
    # M7a-AC8 只需『探針存在且 data-* 皆可解析為數字』;PerfA/B/C(M7b/M7c)再據此斷言門檻。
    _render_ms = round((time.perf_counter() - _T0) * 1000.0, 2)
    # ★ M7b:探針『額外』回寫目前顯示圖的判定狀態(供 test_m7b_e2e 機器讀;設計 §5 P1 已預期 data-verdict)。
    # ★ 本輪(資訊徽章移除):進度序號/總數從可見徽章移到探針 data-idx/data-total ——
    #   頂列「N / total」文字拿掉後,E2E 的 ready/進度錨點改讀此探針(主文件 DOM,設計 §5 P1 機器讀面)。
    _sc_cur = sidecar.load(cur["path"])
    _cur_verdict = _sc_cur.get("verdict", "unset")
    _cur_status = _sc_cur.get("review_status", "none")
    _cur_book = 1 if _sc_cur.get("bookmarked") else 0
    _cur_conf = ss.get("footer_conf_thr", 0.25)
    # ★ compare 第三輪(雙 model 覆蓋比對):探針回寫 compare 統計供 test_compare_e2e 用穩定數字斷言
    #   (queue 張數 / 只A / 只B / B 缺檔 / 覆蓋差),而非脆弱像素。只在比較模式有意義,否則空字串。
    _cmpp = ss.get("_cmp_probe", {}) if ss.get("compare_on") else {}
    st.markdown(
        f"<div id='perf' style='display:none' "
        f"data-render-ms='{_render_ms}' "
        f"data-reruns='{ss.reruns}' "
        f"data-thumb-recalc='{_counters['thumb_recalc']}' "
        f"data-tool-calls='{_counters['tool_calls']}' "
        f"data-verdict='{_cur_verdict}' "
        f"data-status='{_cur_status}' "
        f"data-bookmarked='{_cur_book}' "
        f"data-conf='{_cur_conf}' "
        f"data-idx='{ss.idx + 1}' "
        f"data-total='{total}' "
        f"data-shown-k='{len(kept)}' "
        f"data-shown-n='{len(cur['detections'])}' "
        f"data-undo-n='{len(ss.undo_stack)}' "
        f"data-cmp-queue-n='{_cmpp.get('queue_n', '')}' "
        f"data-cmp-only-a='{_cmpp.get('only_a', '')}' "
        f"data-cmp-only-b='{_cmpp.get('only_b', '')}' "
        f"data-cmp-missing-b='{_cmpp.get('missing_b', '')}' "
        f"data-cmp-delta-imgs='{_cmpp.get('delta_imgs', '')}'></div>",
        unsafe_allow_html=True,
    )

"""modeldiff — 兩 model 在同一影像集的覆蓋差異(IoU 框級配對)。

設計:3_Architect_Design/24_modeldiff.md。**Tier A 純邏輯**:零 I/O、不 import 其他實作、不 mutate 輸入。
只吃 Detection = {"bbox":[x,y,w,h] 絕對像素左上原點, "cls":str, "conf":float}(沿 M3 共用契約)。
把 CLAUDE.md 點名「配對/收斂是唯一無客觀紅綠保護單點」用純函式 + 單元 gate 鎖住。
"""


def iou(box_a, box_b):
    """[x,y,w,h] 兩框的 IoU;union<=0(退化框)→ 0.0(不除零);對稱。"""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    area_a, area_b = aw * ah, bw * bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def match(dets_a, dets_b, iou_thr=0.5, same_class=True):
    """貪婪框級配對。候選對 = iou>0 且 iou>=iou_thr 且 (not same_class 或 同類);
    依 (iou 降冪, i 升冪, j 升冪) 確定性排序逐對指派(每框至多配一次)。
    回 {"matched":[(i,j,iou)], "only_a":[i...], "only_b":[j...]}(only_* 升冪)。"""
    cands = []
    for i, a in enumerate(dets_a):
        for j, b in enumerate(dets_b):
            if same_class and a.get("cls") != b.get("cls"):
                continue
            v = iou(a["bbox"], b["bbox"])
            if v > 0 and v >= iou_thr:  # 無重疊(iou=0)永不配,即使 iou_thr=0
                cands.append((v, i, j))
    cands.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_a, used_b, matched = set(), set(), []
    for v, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, v))
    only_a = [i for i in range(len(dets_a)) if i not in used_a]
    only_b = [j for j in range(len(dets_b)) if j not in used_b]
    return {"matched": matched, "only_a": only_a, "only_b": only_b}


def diff_image(dets_a, dets_b, iou_thr=0.5, conf_range=(0.0, 1.0), classes=None,
               same_class=True, a_present=True, b_present=True):
    """單張圖兩 model 覆蓋差異:先用 conf 雙界 + 類別過濾,再 IoU 配對。
    回 {n_a,n_b,matched,only_a,only_b,status,a_present,b_present}(matched/only_* 為數量 int)。
    缺檔(a_present/b_present=False)昇為 missing_* status,與『有檔但 0 框(both_empty)』區分。"""
    lo, hi = conf_range
    cls_set = set(classes) if classes else None

    def _keep(det):
        c = float(det.get("conf", 0.0))
        if not (lo <= c <= hi):
            return False
        if cls_set is not None and det.get("cls") not in cls_set:
            return False
        return True

    fa = [x for x in dets_a if _keep(x)]
    fb = [x for x in dets_b if _keep(x)]
    m = match(fa, fb, iou_thr=iou_thr, same_class=same_class)
    n_a, n_b = len(fa), len(fb)
    n_match, n_oa, n_ob = len(m["matched"]), len(m["only_a"]), len(m["only_b"])

    if not a_present and not b_present:
        status = "missing_both"
    elif not b_present:
        status = "missing_b"
    elif not a_present:
        status = "missing_a"
    elif n_a == 0 and n_b == 0:
        status = "both_empty"
    elif n_oa == 0 and n_ob == 0:
        status = "agree"
    elif n_oa > 0 and n_ob == 0:
        status = "a_only"
    elif n_ob > 0 and n_oa == 0:
        status = "b_only"
    else:
        status = "disagree"

    return {"n_a": n_a, "n_b": n_b, "matched": n_match, "only_a": n_oa, "only_b": n_ob,
            "status": status, "a_present": a_present, "b_present": b_present}


def summarize(records):
    """資料集層級彙總(records = list[diff_image 輸出 + app 加的 name])。
    弱 model 覆蓋少 → total_b/imgs_b 較小、delta_boxes/delta_imgs > 0、total_only_a 多。"""
    total_a = sum(r.get("n_a", 0) for r in records)
    total_b = sum(r.get("n_b", 0) for r in records)
    imgs_a = sum(1 for r in records if r.get("n_a", 0) > 0)
    imgs_b = sum(1 for r in records if r.get("n_b", 0) > 0)
    return {
        "total_a": total_a, "total_b": total_b, "imgs_a": imgs_a, "imgs_b": imgs_b,
        "total_matched": sum(r.get("matched", 0) for r in records),
        "total_only_a": sum(r.get("only_a", 0) for r in records),
        "total_only_b": sum(r.get("only_b", 0) for r in records),
        "delta_boxes": total_a - total_b,
        "delta_imgs": imgs_a - imgs_b,
        "n_missing_a": sum(1 for r in records if r.get("status") in ("missing_a", "missing_both")),
        "n_missing_b": sum(1 for r in records if r.get("status") in ("missing_b", "missing_both")),
    }


_MODE_PRED = {
    "all": lambda s: True,
    "disagree": lambda s: s in ("a_only", "b_only", "disagree"),
    "a_only": lambda s: s == "a_only",
    "b_only": lambda s: s == "b_only",
    "agree": lambda s: s == "agree",
    "missing": lambda s: s in ("missing_a", "missing_b", "missing_both"),
}


def filter_images(records, mode):
    """對整個影像集 records 做 status triage,回符合 mode 的子集(順序保持)。"""
    pred = _MODE_PRED.get(mode, _MODE_PRED["all"])
    return [r for r in records if pred(r.get("status"))]


_RANK = {"missing_a": 0, "missing_b": 0, "missing_both": 0,
         "disagree": 1, "a_only": 2, "b_only": 3, "agree": 4, "both_empty": 5}


def queue(records):
    """分歧/缺檔優先排序:key=(rank(status), -(only_a+only_b), name)。"""
    def key(r):
        return (_RANK.get(r.get("status"), 9),
                -(r.get("only_a", 0) + r.get("only_b", 0)),
                r.get("name", ""))
    return sorted(records, key=key)

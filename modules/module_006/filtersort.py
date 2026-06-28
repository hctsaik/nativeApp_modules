"""filtersort 模組實作(PG / U-Net 上採樣)。

純邏輯、零 I/O、零 GUI、僅依賴 Python 標準庫。
依契約:3_Architect_Design/10_filtersort.md(AC1..AC26 + §4 邊界 + §5.2 metamorphic)。

對齊但不引用上游實作(ROADMAP M3 共用契約:消費端不得引用 yolo / sidecar / tagging 模組)。
reviewed 語義以「自帶等價邏輯」對齊 tagging 的 is_reviewed(見 _is_reviewed)。
"""

# 合法排序鍵全集(順序釘死;len == 6)。
SORT_KEYS = ["name", "time", "conf", "n_det", "reviewed", "tagged"]


def _is_reviewed(sidecar):
    """複用 tagging 語義(不 import 實作):
    verdict 非 "unset" 或 review_status == "done" 即視為已 reviewed。
    """
    verdict = sidecar.get("verdict", "unset")
    review_status = sidecar.get("review_status", "none")
    return verdict != "unset" or review_status == "done"


def _conf(item):
    """該圖最大偵測 conf;無偵測 → 0.0。"""
    dets = item.get("detections", [])
    if not dets:
        return 0.0
    return max(d["conf"] for d in dets)


def _n_det(item):
    """偵測數量;缺/空 → 0。"""
    return len(item.get("detections", []))


def _primary(item, key):
    """對單一 item 取指定 key 的派生「主鍵」量(純讀,不修改 item)。

    name 為契約必備鍵(缺 name 屬呼叫端違約,允許 KeyError 浮現,Tier A 不防禦)。
    其餘鍵缺資料時套「預設視同值」。
    """
    if key == "name":
        return item["name"]
    if key == "time":
        return item.get("time", 0.0)
    if key == "conf":
        return _conf(item)
    if key == "n_det":
        return _n_det(item)
    if key == "reviewed":
        return _is_reviewed(item.get("sidecar", {}))
    if key == "tagged":
        return len(item.get("sidecar", {}).get("tags", [])) > 0
    # 不會走到這(呼叫前已驗 key 合法),保底拋錯避免靜默。
    raise ValueError("invalid sort key: %r" % (key,))


def sort_items(items, key, reverse=False):
    """穩定排序,回新 list;不 mutate 輸入。

    - key ∈ SORT_KEYS,否則拋 ValueError。
    - tie-break 固定為 name 字典序(升序),且不受 reverse 影響。
    - reverse=True 僅反轉「主鍵」為降序;tie-break name 仍升序。
      (嚴禁 sorted(..., reverse=True) 整個 tuple 反轉。)
    """
    if key not in SORT_KEYS:
        raise ValueError(
            "invalid sort key: %r (must be one of %r)" % (key, SORT_KEYS)
        )

    if not reverse:
        return sorted(items, key=lambda it: (_primary(it, key), it["name"]))

    # reverse=True:僅反主鍵,tie-break name 仍升序。
    if key == "name":
        # 字串主鍵 = tie-break,無分歧,直接 name 降序。
        return sorted(items, key=lambda it: it["name"], reverse=True)
    if key in ("time", "conf", "n_det"):
        # 數值主鍵取負得降序,name 仍升序。
        return sorted(items, key=lambda it: (-_primary(it, key), it["name"]))
    # 布林主鍵(reviewed / tagged):以 -int(primary) 為主鍵(True 在前),name 升序。
    return sorted(items, key=lambda it: (-int(_primary(it, key)), it["name"]))


def review_queue(items):
    """釘死的預設 Review Queue 優先規則,回新 list;不 mutate 輸入。

    三層鍵(升序):
      1. 待審優先:has_det_and_unreviewed(n_det>0 且 未reviewed)→ 0 排前、其餘 → 1。
      2. 最大 conf 降序:-conf。
      3. name 字典序升序。
    """
    def _key(it):
        n_det = _n_det(it)
        reviewed = _is_reviewed(it.get("sidecar", {}))
        pending = n_det > 0 and not reviewed
        return (0 if pending else 1, -_conf(it), it["name"])

    return sorted(items, key=_key)

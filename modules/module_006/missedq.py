"""missedq 模組實作(/pg 上採樣)。

依契約:3_Architect_Design/13_missedq.md(I/O §2、資料流 §3、邊界 §4、AC1..AC27)。

對一組影像 item,用一組釘死規則自動挑出「需要人再看一眼」的圖
(漏檢 / 誤報 / 低信心 / 未審有框),每張附理由碼並依嚴重度排序成
Missed-Detection Review Queue。純邏輯、零 I/O、零副作用、不 mutate 輸入,
僅依賴 Python 標準庫。

本模組自帶 reviewed 等價語義(見 §3.5),不建立任何上游程式碼相依。
"""

# §2.1 模組常數(釘死字串;順序固定;供 UI/測試逐字斷言)
REASON_MISSED = "missed_detection"          # 漏檢:人判缺陷但模型 0 框
REASON_FALSE = "false_alarm"                # 誤報:有高信心框但人判 false_alarm
REASON_LOWCONF = "low_confidence"           # 低信心:有框但最大 conf 低於門檻
REASON_UNREVIEWED = "unreviewed_with_det"   # 未審有框:有框但尚未 reviewed

# 順序固定 = flags_for 的輸出檢查順序,亦是同一 item 多碼時的排列順序。
# 嚴重度(priority)由索引決定:索引越小越嚴重。
REASONS = [
    "missed_detection",
    "false_alarm",
    "low_confidence",
    "unreviewed_with_det",
]


def _is_reviewed(sidecar: dict) -> bool:
    """reviewed 語義(§3.5):verdict != "unset" 或 review_status == "done"。

    語義逐字等同 tagging.is_reviewed / filtersort._is_reviewed,但不建立相依
    (自帶等價邏輯)。
    """
    verdict = sidecar.get("verdict", "unset")
    review_status = sidecar.get("review_status", "none")
    return verdict != "unset" or review_status == "done"


def flags_for(item: dict, conf_low: float = 0.3, conf_high: float = 0.7) -> list[str]:
    """回該 item 適用的理由碼(順序 = REASONS 子序列,§3.3)。無問題 → []。

    純讀,不 mutate item。缺 sidecar/detections 一律套預設視同值(§4),不拋 KeyError。
    """
    # §3.1 / §2.3 派生量(對空 detections 不呼叫 max(),回 0.0)
    detections = item.get("detections", [])
    n_det = len(detections)
    max_conf = max((d["conf"] for d in detections), default=0.0)
    sidecar = item.get("sidecar", {})
    verdict = sidecar.get("verdict", "unset")
    reviewed = _is_reviewed(sidecar)

    # §3.2 四條規則(彼此獨立、可同時成立)
    c_missed = verdict == "true_defect" and n_det == 0
    c_false = verdict == "false_alarm" and max_conf >= conf_high
    c_lowconf = n_det > 0 and max_conf < conf_low
    c_unrev = n_det > 0 and reviewed is False

    # §3.3 命中者依 REASONS 固定順序加入 → 輸出恆為 REASONS 子序列
    conditions = [c_missed, c_false, c_lowconf, c_unrev]
    return [r for r, cond in zip(REASONS, conditions) if cond]


def missed_queue(items: list[dict], conf_low: float = 0.3, conf_high: float = 0.7) -> list[dict]:
    """回有 ≥1 flag 的 item,每筆新建 {"name","reasons","priority"};依 (priority, name) 升序(§3.4)。

    不 mutate 輸入或其子物件;輸出 dict 與 reasons list 皆為新建物件。
    """
    out = []
    for item in items:
        reasons = flags_for(item, conf_low, conf_high)
        if len(reasons) >= 1:
            # priority = 最嚴重 flag 的索引(最小索引);reasons 非空故 min 必有值
            priority = min(REASONS.index(r) for r in reasons)
            out.append({
                "name": item["name"],
                "reasons": reasons,
                "priority": priority,
            })
    # §3.4-4 全序排序:第一鍵 priority 升序,第二鍵 name 字典序升序
    out.sort(key=lambda e: (e["priority"], e["name"]))
    return out

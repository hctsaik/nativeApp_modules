"""tagging 模組(M2 / Tier A,純邏輯)。

三層標記模型(Bookmark / Verdict / Action-tags)+ 內建標籤清單,
提供「一筆 sidecar 是否符合查詢條件」的判定 predicate 與集合篩選。

純函式、無副作用、輸入不被 mutate;僅依賴 Python 標準庫。
契約來源:3_Architect_Design/05_tagging.md(AC1..AC35 + §4 邊界)。
"""

BUILTIN_TAGS = [
    "Need Review", "Need Discuss", "Potential Miss", "False Alarm",
    "True Defect", "Reflection", "Low Confidence", "New Pattern",
    "Golden Case", "Need Labeling", "Need Retrain",
]
VERDICTS = ["unset", "true_defect", "false_alarm", "reflection"]


def is_reviewed(sidecar) -> bool:
    """verdict != "unset" 或 review_status == "done"(布林 OR);缺鍵套預設值。"""
    verdict = sidecar.get("verdict", "unset")
    review_status = sidecar.get("review_status", "none")
    return verdict != "unset" or review_status == "done"


def matches(sidecar: dict, query: dict) -> bool:
    """逐條件以 AND 合併;query 未給的鍵 = 該條件不參與(視為通過)。"""
    # 1. tags(any / all / 空清單視為不參與)
    if "tags" in query:
        q_tags = set(query["tags"])
        if q_tags:  # 空集合 → 不參與(§4b / AC14)
            s_tags = set(sidecar.get("tags", []))
            if query.get("mode", "any") == "all":
                if not q_tags.issubset(s_tags):
                    return False
            else:  # any
                if not (s_tags & q_tags):
                    return False

    # 2. verdict(精確相等)
    if "verdict" in query:
        if sidecar.get("verdict", "unset") != query["verdict"]:
            return False

    # 3. review_status(精確相等)
    if "review_status" in query:
        if sidecar.get("review_status", "none") != query["review_status"]:
            return False

    # 4. reviewed(布林相等,可查未審)
    if "reviewed" in query:
        if is_reviewed(sidecar) != query["reviewed"]:
            return False

    # 5. bookmarked(布林相等;缺鍵視同 False)
    if "bookmarked" in query:
        if sidecar.get("bookmarked", False) != query["bookmarked"]:
            return False

    # 6. text(大小寫不敏感子字串;空字串 → 通過)
    if "text" in query:
        if query["text"].lower() not in sidecar.get("comment", "").lower():
            return False

    return True


def filter_records(items, query: dict) -> list:
    """items 為 (record, sidecar) 序列;保序回符合者的 record(不 mutate 輸入)。"""
    return [record for (record, sidecar) in items if matches(sidecar, query)]

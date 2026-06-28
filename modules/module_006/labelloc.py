"""labelloc 模組實作(/pg 產物;依 3_Architect_Design/22_labelloc.md 契約)。

給定一個 YOLO 資料夾,自動判定該批影像的偵測標註檔(`<stem>.json`)放在
`<folder>/labels/` 子資料夾、還是和影像同一層 `<folder>/`,回傳「該去哪讀標註」的
目錄絕對路徑。

對外承諾(§7):
- `resolve_label_dir` / `label_path` 永不拋例外;回「絕對路徑字串 / None」(resolve)
  或「路徑字串」(label_path)。
- 僅依賴 Python 標準庫(os / pathlib);不 import 任何本專案模組。
- 純讀檔(os.path.isdir / os.listdir);不建檔、不寫檔、不讀標註 JSON 內容、不改原圖。
- 大小寫 / 跨平台:子夾名 / stem / ext 比對皆以 os.path.normcase normalize(§2.1);
  命中後用磁碟真實名 + os.path.abspath 組回傳路徑。

依賴:os / pathlib(標準庫)。
"""
import os
from pathlib import Path


def _ext_matches(name, ext_nc):
    """檔名 name 的副檔名(normcase 後)是否等於 ext_nc(已 normcase 的 ext)。"""
    file_ext = os.path.splitext(name)[1]
    return os.path.normcase(file_ext) == ext_nc


def has_labels(label_dir, stems, *, ext=".json"):
    """label_dir 內是否含 ≥1 個對應的 <stem><ext>(helper,可對外)。

    - stems 非空 → 「label_dir 內符合 ext 的檔案 stem 集合」與「stems 集合」交集是否非空。
    - stems 為 None 或空 → 是否含任一 *<ext>(退化規則)。
    - 以單次 os.listdir + set 比對(O(n),不逐 stem isfile)。
    - 比對皆以 os.path.normcase normalize(跨平台一致;§2.1)。
    - label_dir 不存在 / 不可讀(OSError)→ False,不拋例外。
    """
    ext_nc = os.path.normcase(ext)
    try:
        entries = os.listdir(label_dir)
    except OSError:
        return False

    # 只收符合 ext 的項目(不檢 isfile:省 syscall;非檔者其 stem 不會混入 stems)。
    json_names = [e for e in entries if _ext_matches(e, ext_nc)]

    if stems is None or len(stems) == 0:
        # 退化規則:含任一 *<ext> 即 True。
        # (註:resolve_label_dir 對 stems=[] 不走此 helper 路徑,改直接退同層;見 §3/§4。)
        return len(json_names) > 0

    want = {os.path.normcase(str(s)) for s in stems}
    have = {os.path.normcase(os.path.splitext(e)[0]) for e in json_names}
    return len(want & have) > 0


def resolve_label_dir(folder, stems=None, *, ext=".json", subdir="labels"):
    """主入口(資料夾層級判定)。給定 YOLO 資料夾 folder,回「該去哪讀偵測標註檔」的目錄絕對路徑。

    precedence(§3):
    1. folder 非資料夾(os.path.isdir 為 False;含不存在、是檔案)→ None。
    2. <folder>/<subdir>/ 存在(normcase 枚舉命中真實子夾名)且 isdir 且
       has_labels(子夾, stems) 為真(子夾被「證實」含這批圖的 <stem><ext>)→ 採子夾,回其 abspath。
    3. 否則 → 回 os.path.abspath(folder)(同層)。

    - stems: 可迭代的影像 stem;stems=None → 退化(子夾含任一 *<ext> 即採);stems=[] → 不採子夾。
    - 純讀檔,不建檔、不拋例外(OSError 吞成 fallback 同層)。回傳一律 os.path.abspath 正規化。
    """
    if not os.path.isdir(folder):
        return None

    # stems=[](空清單)→ 視同「無對應檔」→ 不採子夾,直接退同層(§3/§4.g、AC14)。
    # 注意:只有 stems=None 走 has_labels 的退化規則(含任一 *<ext> 即採);
    # 空清單與 None 在 resolve 層語義不同,故在此先攔(不可一律委派給 has_labels)。
    if stems is not None and len(stems) == 0:
        return os.path.abspath(folder)

    # 探子夾:以 normcase 枚舉 folder 找真實名命中 subdir 的子目錄。
    subdir_nc = os.path.normcase(subdir)
    try:
        entries = os.listdir(folder)
    except OSError:
        # 枚舉不可讀 → 退同層(容錯第一)。
        return os.path.abspath(folder)

    for entry in entries:
        if os.path.normcase(entry) != subdir_nc:
            continue
        cand = os.path.join(folder, entry)
        if not os.path.isdir(cand):
            # 同名但是檔案 → 視為無子夾(§4.i);命中名唯一,直接停。
            break
        if has_labels(cand, stems, ext=ext):
            return os.path.abspath(cand)
        # 子夾存在但未被證實(空 / stem 不對應)→ 退同層;命中名唯一,直接停。
        break

    return os.path.abspath(folder)


def label_path(label_dir, image_path, *, ext=".json"):
    """per-image 路徑解析(純字串,不碰檔案系統、不檢存在、不拋例外)。

    回 os.path.join(label_dir, Path(image_path).stem + ext)。
    語義對齊現行 app._pred_path(label_dir, image_path),以便直接替換。
    存在性與壞檔交給 yolo.load 容錯。
    """
    stem = Path(image_path).stem
    return os.path.join(label_dir, stem + ext)

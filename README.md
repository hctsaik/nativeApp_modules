# nativeApp_modules — CIM 第一方 CV 模組包

CIM Hybrid Edge Platform（[hctsaik/nativeApp](https://github.com/hctsaik/nativeApp)）的**第一方 CV 模組**獨立 repo，比照 LV / AI4BI 以 **git submodule** 掛回平台：

```
nativeApp/sidecar/python-engine/plugins/cim-modules/   ← 本 repo（submodule）
```

平台啟動時掃 `plugins/*/modules/*/plugin.yaml` 自動註冊，**engine.py / plugin_loader.py 無需改動**。
設計與決議見平台 repo 的 [`docs/platform/modules-independence-and-store-plan.md`](https://github.com/hctsaik/nativeApp/blob/main/docs/platform/modules-independence-and-store-plan.md)。

## 結構

```
modules/
├─ module_001 ~ 005 / 007 / 021/   各模組（plugin.yaml + *_input/process/output.py）
└─ _shared/frame_fit_score.py       跨模組 CV 領域共用碼
```

## 開發（在平台內，不要單獨跑）

模組執行期依賴平台共用層（`scripts/shared/_config_base.py`、`_help.py` 等）與框架注入的環境變數，
**必須透過平台 `start-dev.bat` 啟動整個 app**，engine 才會正確注入 `CIM_*` 變數。

日常迴圈：
1. 在本 submodule 內改檔。
2. `POST http://127.0.0.1:<engine_port>/reload` 熱載即生效（**不需 commit**）。
3. 要讓別人看到改動：先在本 repo `git add/commit/push`，
   再回平台 `git add plugins/cim-modules` 釘 submodule 指標、平台 commit。

## 平台契約

本 repo 對平台 host 的依賴面被 `tests/test_modules_platform_contract.py`（在平台側）以 allowlist 鎖死：
僅允許 `core.*` 與 `scripts/shared/{_config_base, _help}`。新增平台依賴前先擴充 allowlist 並更新設計文件。

## 測試

- **本 repo 純邏輯單元測試**：`modules/module_*/*_process_test.py`（`py -3.11 -m pytest modules/`，需平台共用碼在 path 上時請在平台 super-repo 內跑）。
- **完整套件（含契約 / catalog 註冊）**：在平台 super-repo 跑 `npm run test:python`。

> 模型權重、資料不入本 repo。

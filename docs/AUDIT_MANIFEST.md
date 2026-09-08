# Audit Manifest — 提交溯源（SliceAudit）

> **权威仓库 = 远端 github.com/ausyeah/artical（main）。**
> 说明：2026-09-08 本机 git 历史曾因一次异常的 `git rebase` 被破坏后重建，导致本地
> 历史与远端**分叉**；此后推送一律经 GitHub Git Data API（`D:\论文\push_api.py`），
> **远端 main 才是审计轨迹的唯一权威**。核验请以远端 commit 为准。

## 远端 main 已知提交（新→旧）

| SHA（前 10 位） | 内容 | 日期 |
|---|---|---|
| b765b116dd | Preprint v1: Open-Source/Availability（HEAD） | 2026-09-08 |
| a7bdc76350 | Preprint v1 draft（SliceAudit） | 2026-09-08 |
| c2125b958a | Stage B eval script (02) | 2026-09-08 |
| 48adf516fe | Stage B pilot toolkit (00/01/RUN_GUIDE) | 2026-09-08 |
| e65908d440 | Stage B protocol v1.0 + REFERENCES | 2026-09-08 |
| 09d5aa7af9 | src/03, src/04 + step3 results | 2026-09-08 |
| 58e9b0ee48 | T2 mechanical null + prereg v1.1 | 2026-09-08 |
| 4be6a272c9 | Pre-registration v1 FREEZE | 2026-09-08 |
| 5714936cff | Stage A v3 cross-corpus audit（**预注册 §9 早期引用点**） | 2026-09-08 |

## 产物可复现状态（配合 §8）

| 产物 | 是否入库 | 再生成方式 |
|---|---|---|
| reports/step2_summary.json / step2_gate_regression.json | ✅ | `src/01_slice_audit.py`（需本地 UltraX 分片，见 SETUP.md） |
| reports/step3_*.json | ✅ | `src/03…/04…` |
| reports/step2_lines.csv.gz / step2_docs.csv | ✅ | 同上（doc 表已放开 gitignore） |
| figures/fig2_stageA_v3.png | ✅ | 01 脚本绘图段 |
| runs/pilot/*（训练/评估中间体） | ❌ 不入库 | `src/stageb/00…02…` 生成（大文件本地保留） |
| 数据分片（UltraX-Preview） | ❌ | 外源 Apache-2.0，HF `openbmb/UltraX-Preview` |

## 已知待办（写入 ledger，未阻塞本稿）
- `03_gate.py`（pilot 四门判据聚合）未写
- tail-restored 反事实臂未生成
- OOV/knowledge 分类器需与 03 统一 + 人工标注子集（T3）
- 中文语料切片（堵"只测英文"质疑）未做

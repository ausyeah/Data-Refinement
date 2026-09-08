# -*- coding: utf-8 -*-
"""
阶段 A · 零训练切片审计 —— 第 0 步：探查 UltraX-Preview 的真实 schema

目的
----
UltraX-Preview（HF: openbmb/UltraX-Preview，约 1.14 亿条 / 487GB）自带
「原文 + 精炼结果 + 编辑操作」三元组，是本研究的核心实验台。
但**我尚未见过它的真实字段结构**，本脚本先做探查，不臆测 schema。

为什么用 streaming
------------------
全量 487GB 远超笔记本磁盘。streaming=True 只拉取实际读取的样本，
**不落地全量数据** —— 这直接绕开了磁盘约束（用户磁盘 ≥30GB 已足够）。

用法
----
    uv pip install datasets pandas
    # 国内建议走 HF 镜像
    export HF_ENDPOINT=https://hf-mirror.com
    python src/00_probe_ultrax.py --n 200

产出
----
    reports/step0_schema.json   字段结构与样本示例
    reports/step0_samples.csv   原始样本（供人工核对）

脚本只读、不修改任何数据集。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

DATASET_ID = "openbmb/UltraX-Preview"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="探查样本条数")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("[错误] 未安装 datasets。请先： uv pip install datasets pandas")
        return 1

    print(f"[1/4] 流式加载 {DATASET_ID} (split={args.split}) —— 不会下载全量 487GB")
    try:
        ds = load_dataset(DATASET_ID, split=args.split, streaming=True)
    except Exception as e:  # noqa: BLE001
        print(f"[错误] 加载失败：{e}")
        print("排查：1) 网络/HF 镜像  2) 数据集名是否正确  3) 是否需登录（gated repo）")
        print("国内可试： export HF_ENDPOINT=https://hf-mirror.com")
        return 1

    print("[2/4] 读取前 %d 条 ..." % args.n)
    rows = []
    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        rows.append(ex)
    print(f"      实际读取 {len(rows)} 条")

    if not rows:
        print("[错误] 未读到任何样本，可能 split 名称不对。")
        return 1

    # ---- 3. 探查 schema：不臆测，直接看 ----
    print("[3/4] 探查字段结构")
    keys = list(rows[0].keys())
    schema = {}
    for k in keys:
        vals = [r.get(k) for r in rows[:20]]
        nonnull = [v for v in vals if v is not None]
        sample = nonnull[0] if nonnull else None
        schema[k] = {
            "type": type(sample).__name__,
            "nonnull_in_first20": len(nonnull),
            "sample_value": (str(sample)[:500] if sample is not None else None),
        }

    out = {
        "dataset_id": DATASET_ID,
        "n_probed": len(rows),
        "fields": keys,
        "schema": schema,
    }
    (REPORTS / "step0_schema.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(REPORTS / "step0_samples.csv", index=False, encoding="utf-8-sig")
    except ImportError:
        pass

    # ---- 4. 打印结论 ----
    print("[4/4] 字段清单：")
    for k in keys:
        s = schema[k]
        print(f"      - {k:<28} {s['type']:<10} 示例: {str(s['sample_value'])[:80]}")

    print()
    print("=" * 68)
    print("下一步（人工判断，脚本不替你决定）：")
    print("  1. 打开 reports/step0_schema.json，确认哪个字段是『原文』、『精炼结果』、『编辑操作』")
    print("  2. 确认『编辑操作』的数据结构（是字符串？列表？含 keep/del/replace/insert 标签？）")
    print("  3. 把字段名填进 src/01_slice_audit.py 的 CONFIG，再跑分层统计")
    print()
    print("⚠️ 若发现该数据集不含编辑操作留痕，则阶段 A 前提不成立，")
    print("   须立即回退到『自己注入伪影』方案（此时剂量须按严复核要求校准到实测发生率）。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())

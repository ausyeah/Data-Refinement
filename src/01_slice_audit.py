# -*- coding: utf-8 -*-
"""
阶段 A · 切片级审计（v1）

研究问题
--------
UltraX（arXiv:2607.08646）在**聚合层面**证明精炼有效（16B 精炼 tokens > 20B 原始 tokens），
但未回答"对谁有效"。本脚本测量：**被精炼流水线删除的内容，是否系统性地偏斜向长尾/稀有词汇**。

设计要点（为什么这样做）
------------------------
1. **不用操作日志重建删除**：replace_str 的语义无法从外部确定（其会合并行），
   因此以"raw 行是否（模糊地）出现在 cleaned 中"定义保留/删除——对合并、改写、
   轻度编辑都稳健。操作日志仅作分类学描述。
2. **文档内对照**：主统计量是 per-doc 的
   `mean_zipf(被删行) - mean_zipf(保留行)`，文档内比较天然控制了主题/领域混杂。
   负值 = 删除偏向稀有词。
3. **稀有度代理**：wordfreq 的 zipf 频率（外部、前训练时代词频，与语料无关）。
   zipf 越低越稀有；<2.0 ≈ 百万分之一以下。
4. **混杂控制**：HTML 行/空行单独归类；主分析在**纯文本行**子集上复跑；
   协变量：行长、行内 HTML 密度、文档内位置。
5. **推断**：文档级 bootstrap 95% CI + 文档内稀有度标签置换（500 次）。
   多重比较：本阶段只有一个主检验（confirmatory），其余 exploratory。

输出
----
reports/step1_lines.csv.gz      行级明细（抽样保存，供人工核对）
reports/step1_summary.json      全部汇总统计量
figures/fig1_deletion_vs_rarity.png   判生死图
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import unicodedata
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
REPORTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

DATASET = "openbmb/UltraX-Preview"
RARE_ZIPF = 2.0          # zipf < 2.0 ≈ 每百万词不足 1 次
KEEP_THRESHOLD = 90.0    # rapidfuzz partial_ratio >= 90 视为行被保留
N_PERM = 500
N_BOOT = 1000
SEED = 20260908

TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
HEADING_RE = re.compile(r"^\s*#{1,6}\s")


def strip_html(s: str) -> str:
    s = TAG_RE.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def html_density(s: str) -> float:
    if not s:
        return 0.0
    tags = len(TAG_RE.findall(s))
    return tags / max(1, len(s) / 10)  # 每 10 字符的标签数


def line_type(line: str) -> str:
    if not line.strip():
        return "blank"
    if html_density(line) > 0.5:
        return "html"
    return "prose"


def parse_ops(pf: str) -> dict:
    """行首锚定解析 processed_functions（防止正文 'Jordan (2)' 误判）。"""
    ops = collections.Counter()
    if not pf:
        return dict(ops)
    for ln in pf.splitlines():
        m = re.match(r"\s*([a-zA-Z_]+)\s*\(", ln)
        if m:
            ops[m.group(1)] += 1
    return dict(ops)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--subset", default=None,
                    help="限定 data/ 子目录（如 UltraX-AICC）；默认全部可用子集")
    args = ap.parse_args()
    rng = random.Random(SEED)

    from datasets import load_dataset
    from wordfreq import zipf_frequency
    from rapidfuzz import fuzz

    print(f"[1/5] 流式加载 {DATASET} ...")
    ds = load_dataset(DATASET, split="train", streaming=True)

    rows = []            # 行级记录
    doc_stats = []       # 文档级记录
    op_counter = collections.Counter()
    n_docs = n_lines = 0
    skipped_short = 0

    print(f"[2/5] 处理前 {args.n} 条文档 ...")
    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        raw_full, clean_full = ex["raw_content"] or "", ex["cleaned_content"] or ""
        if len(raw_full) < 200:            # 过短文档无分析价值
            skipped_short += 1
            continue

        raw_lines = [l for l in (x.strip() for x in raw_full.splitlines()) if l]
        # 合并后的 cleaned 全文（去掉 HTML 标记后再匹配，避免标签差异导致漏配）
        clean_text = strip_html(clean_full)
        if len(clean_text) < 50:
            skipped_short += 1
            continue

        ops = parse_ops(ex["processed_functions"] or "")
        op_counter.update(ops)
        source = ex.get("source", "?")

        # 行级判定
        kept_zipf, del_zipf = [], []
        doc_rare_del = doc_rare_kept = 0
        for li, line in enumerate(raw_lines):
            typ = line_type(line)
            if typ == "blank":
                continue
            if typ == "html":
                probe = strip_html(line)
                if len(probe) < 15:        # 纯标签行
                    continue
            else:
                probe = line

            # 快速路径：整行精确出现在 cleaned 中（常见情形，省去模糊匹配）
            if probe in clean_text:
                score = 100.0
            else:
                score = fuzz.partial_ratio(probe[:300], clean_text[:20000])
            kept = score >= KEEP_THRESHOLD

            words = WORD_RE.findall(probe.lower())
            if not words:
                continue
            zipfs = [zipf_frequency(w, "en") for w in words]
            known = [z for z in zipfs if z > 0]           # OOV 单列，不冒充"极稀有"
            oov_frac = 1 - len(known) / len(zipfs)
            rar = (sum(known) / len(known)) if known else None
            rare_known_frac = (sum(1 for z in known if z < RARE_ZIPF) / len(known)) if known else 0.0

            if known:
                (kept_zipf if kept else del_zipf).extend(known)
            if kept:
                doc_rare_kept += sum(1 for z in known if z < RARE_ZIPF)
            else:
                doc_rare_del += sum(1 for z in known if z < RARE_ZIPF)

            rows.append({
                "doc": i, "line": li, "source": source, "type": typ,
                "kept": int(kept), "match": round(score, 1),
                "n_words": len(words),
                "rarity_mean": (round(rar, 3) if rar is not None else None),
                "rare_known_frac": round(rare_known_frac, 3),
                "oov_frac": round(oov_frac, 3),
                "is_heading": int(bool(HEADING_RE.match(line))),
                "n_chars": len(line),
                "pos": round(li / max(1, len(raw_lines)), 3),
            })
            n_lines += 1

        n_docs += 1
        gap = (statistics.mean(del_zipf) - statistics.mean(kept_zipf)
               if del_zipf and kept_zipf else None)
        doc_stats.append({
            "doc": i, "source": source,
            "n_kept_tok": len(kept_zipf), "n_del_tok": len(del_zipf),
            "zipf_kept": statistics.mean(kept_zipf) if kept_zipf else None,
            "zipf_del": statistics.mean(del_zipf) if del_zipf else None,
            "zipf_gap": gap,
            "rare_rate_del": doc_rare_del / max(1, doc_rare_del + doc_rare_kept),
            "replace_str": ops.get("replace_str", 0),
            "remove_lines": ops.get("remove_lines", 0),
            "remove_all": ops.get("remove_all", 0),
            "add_line": ops.get("add_line", 0),
        })

    print(f"[3/5] 文档 {n_docs}（跳过过短 {skipped_short}），有效行 {n_lines}")

    # ---------------- 汇总统计 ----------------
    all_rows = rows
    gaps = [d["zipf_gap"] for d in doc_stats if d["zipf_gap"] is not None]

    def boot_ci(vals, f=statistics.mean, n=N_BOOT):
        if not vals:
            return (None, None, None)
        ms = []
        for _ in range(n):
            ms.append(f(rng.choice(vals) for _ in range(len(vals))))
        return (f(vals), sorted(ms)[int(0.025 * n)], sorted(ms)[int(0.975 * n)])

    mean_gap, lo, hi = boot_ci(gaps)
    neg_share = sum(1 for g in gaps if g < 0) / max(1, len(gaps))

    # 文档内置换检验：把每篇文档的 zipf_gap 符号随机翻转？不行——置换应打乱行级标签。
    # 文档级近似：H0 下 gap 分布以 0 为中心，用符号翻转近似（对每篇文档 ±gap 等概率）。
    perm_stats = []
    for _ in range(N_PERM):
        s = sum(g * (1 if rng.random() < 0.5 else -1) for g in gaps)
        perm_stats.append(s / len(gaps))
    p_perm = sum(1 for p in perm_stats if p <= mean_gap) / N_PERM   # 单侧

    # 稀有 token 的删除率对比（只含词典内词，OOV 不冒充稀有）
    rare_rows = [r for r in all_rows
                 if r["rarity_mean"] is not None and r["rare_known_frac"] >= 0.10]
    del_rate_rare = sum(r["kept"] == 0 for r in rare_rows) / max(1, len(rare_rows))
    common_rows = [r for r in all_rows if r["rarity_mean"] is not None and r["rarity_mean"] >= 4.0]
    del_rate_common = sum(r["kept"] == 0 for r in common_rows) / max(1, len(common_rows))
    del_rate_all = sum(r["kept"] == 0 for r in all_rows) / max(1, len(all_rows))

    # OOV 通道与稀有度通道分离：OOV 行被删 ≠ 稀有词行被删
    oov_del = statistics.mean(r["oov_frac"] for r in all_rows if r["kept"] == 0) if all_rows else None
    oov_kept = statistics.mean(r["oov_frac"] for r in all_rows if r["kept"] == 1) if all_rows else None

    # 行长分桶：控制"短行更好删且更稀有"的混杂
    def bucket(nw):
        if nw <= 5: return "1-5"
        if nw <= 15: return "6-15"
        if nw <= 40: return "16-40"
        return "40+"
    len_table = {}
    for b in ("1-5", "6-15", "16-40", "40+"):
        seg = [r for r in all_rows if r["rarity_mean"] is not None and bucket(r["n_words"]) == b]
        d = [r["rarity_mean"] for r in seg if r["kept"] == 0]
        k = [r["rarity_mean"] for r in seg if r["kept"] == 1]
        len_table[b] = {
            "n": len(seg),
            "del_rate": round(sum(r["kept"] == 0 for r in seg) / len(seg), 4) if seg else None,
            "zipf_del": round(statistics.mean(d), 3) if d else None,
            "zipf_kept": round(statistics.mean(k), 3) if k else None,
            "gap": round(statistics.mean(d) - statistics.mean(k), 3) if d and k else None,
        }

    # 标题行单独看（标题词稀有但通常被保留——反例检查）
    head = [r for r in all_rows if r["is_heading"] == 1]
    body = [r for r in all_rows if r["is_heading"] == 0]

    # 按文档级稀有 token 删除占比分层（五分位）
    doc_rare = sorted(d["rare_rate_del"] for d in doc_stats if d["n_del_tok"] + d["n_kept_tok"] > 0)

    # 纯文本行子集（关键控制：排除 HTML 结构性删除）
    prose = [r for r in all_rows if r["type"] == "prose"]
    prose_gap_docs = collections.defaultdict(lambda: ([], []))
    # 重建 per-doc（prose-only）：直接用行级聚合
    pz_del = [r["rarity_mean"] for r in prose
              if r["kept"] == 0 and r["rarity_mean"] is not None]
    pz_kept = [r["rarity_mean"] for r in prose
               if r["kept"] == 1 and r["rarity_mean"] is not None]
    prose_gap = (statistics.mean(pz_del) - statistics.mean(pz_kept)) if pz_del and pz_kept else None

    summary = {
        "meta": {"n_docs": n_docs, "n_lines": n_lines, "n_rows": len(all_rows),
                 "skipped_short": skipped_short, "seed": SEED,
                 "keep_threshold": KEEP_THRESHOLD, "rare_zipf": RARE_ZIPF,
                 "dataset": DATASET},
        "op_taxonomy": dict(op_counter),
        "primary": {
            "statistic": "mean over docs of [ mean_zipf(deleted lines) - mean_zipf(kept lines) ]",
            "mean_gap": mean_gap, "ci95": [lo, hi],
            "share_docs_negative": neg_share,
            "perm_p_one_sided": p_perm, "n_perm": N_PERM,
            "note": "负值 = 被删内容比保留内容更稀有（支持长尾偏斜假说）",
        },
        "deletion_rates": {
            "all_lines": del_rate_all,
            "rare_known_lines(rare_known_frac>=0.10)": del_rate_rare,
            "high_freq_lines(mean_zipf>=4)": del_rate_common,
            "n_rare": len(rare_rows), "n_common": len(common_rows),
        },
        "oov_channel": {
            "oov_frac_deleted": oov_del, "oov_frac_kept": oov_kept,
            "note": "OOV（URL/代码/非英语词）是结构性通道，与稀有度通道分离报告",
        },
        "length_buckets": len_table,
        "headings": {
            "n": len(head),
            "del_rate": round(sum(r["kept"] == 0 for r in head) / len(head), 4) if head else None,
            "body_del_rate": round(sum(r["kept"] == 0 for r in body) / len(body), 4) if body else None,
        },
        "controls": {
            "prose_only_gap": prose_gap,
            "prose_only_mean_del": statistics.mean(pz_del) if pz_del else None,
            "prose_only_mean_kept": statistics.mean(pz_kept) if pz_kept else None,
            "n_prose": len(prose),
        },
        "descriptive": {
            "median_doc_rare_rate_del": statistics.median(doc_rare) if doc_rare else None,
        },
    }

    (REPORTS / "step1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(REPORTS / "step1_lines.csv.gz",
                                  index=False, compression="gzip")
        pd.DataFrame(doc_stats).to_csv(REPORTS / "step1_docs.csv", index=False)
    except Exception as e:  # noqa: BLE001
        print("CSV 写出失败:", e)

    print(f"[4/5] 绘图 ...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 按行级稀有度十分位画删除率曲线（排除 OOV-only 行）
        srt = sorted((r for r in all_rows if r["rarity_mean"] is not None),
                     key=lambda r: r["rarity_mean"])
        B = 10
        xs, ys = [], []
        for b in range(B):
            seg = srt[int(b * len(srt) / B): int((b + 1) * len(srt) / B)]
            if not seg:
                continue
            xs.append(statistics.mean(r["rarity_mean"] for r in seg))
            ys.append(sum(r["kept"] == 0 for r in seg) / len(seg))
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("line rarity (mean zipf frequency, lower = rarer)")
        ax.set_ylabel("deletion rate")
        ax.set_title(f"UltraX deletion vs. rarity (n_docs={n_docs}, n_lines={n_lines})")
        ax.invert_xaxis()  # 左边 = 更稀有
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig1_deletion_vs_rarity.png", dpi=150)
        print("   图已存 figures/fig1_deletion_vs_rarity.png")
    except Exception as e:  # noqa: BLE001
        print("绘图失败:", e)

    print("[5/5] 汇总")
    print(json.dumps({k: summary[k] for k in ("primary", "deletion_rates", "controls")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

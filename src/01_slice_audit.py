# -*- coding: utf-8 -*-
"""
阶段 A · 切片级审计（v3）

相对 v2 的修正（对应严复核攻击）：
1. 修复截断 bug：不再把 cleaned 截到 2 万字符（v2 会把长文档后段的行系统性误判为删除）
2. 移除不合法的符号翻转置换 → Wilcoxon 符号秩 + 符号检验 + 聚类/文档 bootstrap
3. 主统计量改为**分桶加权**（1-5/6-15/16-40/40+ 词，权重=桶内 token 份额）——
   v2 的单一 gap 把"短行删高频垃圾(+0.29)"和"内容行删稀有内容(−1.07)"两个相反机制平均掉了
4. **跨语料采样**：AICC / FineWeb / RedPajama-V2 / Ultra-FineWeb 四个子集各抽 N 条
   （v2 的 5000 条全部来自 AICC，来源聚类退化）
5. 阈值敏感性：匹配分数一次计算，85/90/95 三档分别分类
6. 操作日志交叉验证：remove_lines 区间长度之和 vs 判删行数的相关性（匹配器合理性检查）
"""
from __future__ import annotations

import argparse
import json
import os
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
SUBSETS = ["UltraX-AICC", "UltraX-FineWeb", "UltraX-RedPajama-V2", "UltraX-Ultra-FineWeb"]
RARE_ZIPF = 2.0
THRESHOLDS = (85, 90, 95)          # 主阈值 90，其余做敏感性
PRIMARY_THR = 90
N_BOOT = 1000
SEED = 20260908

TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
HEADING_RE = re.compile(r"^\s*#{1,6}\s")
RANGE_RE = re.compile(r"remove_lines\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def strip_html(s: str) -> str:
    s = TAG_RE.sub(" ", s)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def html_density(s: str) -> float:
    return len(TAG_RE.findall(s)) / max(1.0, len(s) / 10.0) if s else 0.0


def line_type(line: str) -> str:
    if not line.strip():
        return "blank"
    return "html" if html_density(line) > 0.5 else "prose"


def parse_ops(pf: str):
    """行首锚定解析；同时累计 remove_lines 区间长度（供交叉验证）。"""
    ops = collections.Counter()
    removed = 0
    if not pf:
        return dict(ops), 0
    for ln in pf.splitlines():
        m = re.match(r"\s*([a-zA-Z_]+)\s*\(", ln)
        if m:
            ops[m.group(1)] += 1
        for m in RANGE_RE.finditer(ln):
            a, b = int(m.group(1)), int(m.group(2))
            removed += max(0, b - a + 1)
    return dict(ops), removed


def bucket_of(nw: int) -> str:
    if nw <= 5: return "1-5"
    if nw <= 15: return "6-15"
    if nw <= 40: return "16-40"
    return "40+"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def boot_ci(vals, f=statistics.mean, n=N_BOOT, clusters=None, rng=None):
    """clusters 至少 5 个才做聚类 bootstrap，否则退回文档级 iid bootstrap。"""
    if not vals:
        return None, None, None
    ms = []
    if clusters and len(clusters) >= 5:
        keys = list(clusters.keys())
        for _ in range(n):
            pick = [rng.choice(keys) for _ in range(len(keys))]
            pool = [v for k in pick for v in clusters[k]]
            ms.append(f(pool))
    else:
        for _ in range(n):
            ms.append(f(rng.choice(vals) for _ in range(len(vals))))
    ms.sort()
    return f(vals), ms[int(0.025 * n)], ms[int(0.975 * n)]


def wilcoxon_signed_rank(gaps):
    """无 scipy 依赖的 Wilcoxon 符号秩（正态近似 + 精确符号检验）。"""
    d = [g for g in gaps if abs(g) > 1e-12]
    n = len(d)
    if n < 10:
        return {"n": n, "note": "样本过小"}
    d.sort(key=abs)
    ranks = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[j + 1]) == abs(d[i]):
            j += 1
        r = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    wp = sum(ranks[k] for k in range(n) if d[k] > 0)
    mu, sigma = n * (n + 1) / 4, math_sigma(n)
    z = (wp - mu) / sigma
    from math import erf, sqrt
    p_two = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    n_pos = sum(1 for g in d if g > 0)
    return {"n": n, "W+": wp, "z": round(z, 3), "p_two_sided": p_two,
            "sign_test_p": min(1.0, 2 * sum(_binom_cdf(k, n, 0.5) for k in [n_pos]) ) if False else None}


def math_sigma(n: int) -> float:
    return (n * (n + 1) * (2 * n + 1) / 24) ** 0.5


def _binom_cdf(k, n, p):
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def sign_test(gaps, alt="negative"):
    """符号检验：P(一半以上文档 gap<0 | 中位数=0)。"""
    n = len(gaps)
    n_neg = sum(1 for g in gaps if g < 0)
    p = _binom_sf(n_neg, n, 0.5)
    return {"n": n, "n_negative": n_neg, "share_negative": n_neg / n, "p_one_sided": p}


def _binom_sf(k, n, p=0.5):
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, p))     # P(X >= k)


def process_doc(ex, i, zipf_frequency, fuzz):
    """单文档处理 → (行记录列表, 文档记录)。"""
    raw_full, clean_full = ex["raw_content"] or "", ex["cleaned_content"] or ""
    if len(raw_full) < 200:
        return None
    raw_lines = [l.strip() for l in raw_full.splitlines() if l.strip()]
    clean_text = strip_html(clean_full)          # 不截断（v2 截断 bug 的修复）
    if len(clean_text) < 50:
        return None
    # 行对行匹配池：合并后的 cleaned 行仍包含原行作为子串，partial_ratio 有效且代价小
    clean_pool = list({strip_html(l) for l in clean_full.splitlines()
                       if len(strip_html(l)) >= 15})
    ops, ops_removed = parse_ops(ex["processed_functions"] or "")
    source = ex.get("source", "?")

    lines, per_thr_deleted = [], collections.defaultdict(list)
    kept_zipf = {"prose": [], "all": []}
    del_zipf = {"prose": [], "all": []}
    for li, line in enumerate(raw_lines):
        typ = line_type(line)
        if typ == "blank":
            continue
        probe = strip_html(line) if typ == "html" else line
        if len(probe) < 15:
            continue
        score = 100.0 if probe in clean_text else (
            max((fuzz.partial_ratio(probe[:300], cl) for cl in clean_pool), default=0.0))
        words = WORD_RE.findall(probe.lower())
        if not words:
            continue
        zipfs = [zipf_frequency(w, "en") for w in words]
        known = [z for z in zipfs if z > 0]
        rar = statistics.mean(known) if known else None
        rec = {
            "doc": i, "subset": source, "type": typ,
            "n_words": len(words), "n_chars": len(line),
            "is_heading": int(bool(HEADING_RE.match(line))),
            "pos": round(li / len(raw_lines), 3),
            "rarity_mean": rar,
            "oov_frac": round(1 - len(known) / len(zipfs), 3),
            "rare_known_frac": (round(sum(1 for z in known if z < RARE_ZIPF) / len(known), 3)
                                if known else 0.0),
            "bucket": bucket_of(len(words)),
            "match": round(score, 1),
        }
        for thr in THRESHOLDS:
            rec[f"kept_{thr}"] = int(score >= thr)
        lines.append(rec)
        if rar is not None:
            target = "prose" if typ == "prose" else "all"
            if rec[f"kept_{PRIMARY_THR}"]:
                kept_zipf[target].append(rar)
            else:
                del_zipf[target].append(rar)
            per_thr_deleted[rec["bucket"]].append((rar, rec[f"kept_{PRIMARY_THR}"]))

    doc = {
        "doc": i, "subset": source,
        "n_lines": len(lines),
        "ops_removed_lines": ops_removed,
        "judged_deleted": sum(1 for r in lines if not r[f"kept_{PRIMARY_THR}"]),
        "replace_str": ops.get("replace_str", 0),
        "remove_lines": ops.get("remove_lines", 0),
        "remove_all": ops.get("remove_all", 0),
        "add_line": ops.get("add_line", 0),
        "zipf_gap_all": (mean(del_zipf["all"]) - mean(kept_zipf["all"]))
                         if del_zipf["all"] and kept_zipf["all"] else None,
        "zipf_gap_prose": (mean(del_zipf["prose"]) - mean(kept_zipf["prose"]))
                           if del_zipf["prose"] and kept_zipf["prose"] else None,
    }
    # 分桶 gap（文档级）
    bgap = {}
    for b in ("1-5", "6-15", "16-40", "40+"):
        pairs = per_thr_deleted.get(b, [])
        d = [z for z, k in pairs if not k]
        k = [z for z, k in pairs if k]
        bgap[b] = (mean(d) - mean(k)) if d and k else None
    doc["bucket_gaps"] = bgap
    return lines, doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subset", type=int, default=1500)
    ap.add_argument("--mode", choices=["local", "stream"], default="local",
                    help="local=读已下载的分片 parquet（推荐）；stream=hf 流式（慢，易被限速）")
    args = ap.parse_args()
    rng = random.Random(SEED)

    from datasets import load_dataset
    from wordfreq import zipf_frequency
    from rapidfuzz import fuzz

    import pyarrow.parquet as pq

    cache = Path(os.environ.get("HF_HOME", "E:/hf-cache")) / "hub"
    # 优先用内容识别后的规范分片（Windows 下 HF snapshot 符号链接常损坏）
    part_map = {}
    pm = ROOT / "reports" / "part_map.json"
    if pm.exists():
        part_map = json.loads(pm.read_text())
    SUB2SRC = {"UltraX-AICC": "AICC", "UltraX-FineWeb": "FineWeb",
               "UltraX-RedPajama-V2": "RedPajama", "UltraX-Ultra-FineWeb": "Ultra-FineWeb"}
    part_files = {}
    if args.mode == "local":
        for sub in SUBSETS:
            src_name = SUB2SRC.get(sub, sub)
            if src_name in part_map and Path(part_map[src_name]).exists():
                part_files[sub] = part_map[src_name]
        print(f"[local] 找到分片: { {k: Path(v).name for k, v in part_files.items()} }", flush=True)

    all_rows, doc_stats = [], []
    doc_counter = 0
    for sub in SUBSETS:
        print(f"[load] {sub} ...", flush=True)
        if args.mode == "local" and sub in part_files:
            pf = pq.ParquetFile(part_files[sub])
            done = 0
            exs = []
            for batch in pf.iter_batches(batch_size=256, columns=["uid", "raw_content", "cleaned_content", "processed_functions", "source"]):
                d = batch.to_pylist()
                for ex in d:
                    if done >= args.per_subset:
                        break
                    r = process_doc(ex, f"{sub}:{done}", zipf_frequency, fuzz)
                    if r is None:
                        continue
                    lines, doc = r
                    all_rows.extend(lines)
                    doc_stats.append(doc)
                    done += 1
                if done >= args.per_subset:
                    break
            print(f"  完成 {done} 篇", flush=True)
            continue

        # 流式回退
        url = f"hf://datasets/{DATASET}/data/{sub}/*.parquet"
        try:
            ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
        except Exception as e:
            print(f"  加载失败，跳过：{str(e)[:120]}")
            continue
        done = 0
        for ex in ds:
            if done >= args.per_subset:
                break
            r = process_doc(ex, f"{sub}:{done}", zipf_frequency, fuzz)
            if r is None:
                continue
            lines, doc = r
            all_rows.extend(lines)
            doc_stats.append(doc)
            done += 1
        print(f"  完成 {done} 篇", flush=True)

    n_docs = len(doc_stats)
    n_lines = len(all_rows)
    print(f"[done] 文档 {n_docs}，行 {n_lines}")

    subsets = sorted({d["subset"] for d in doc_stats})
    # ---------- 主统计量：分桶加权 gap（文档级） ----------
    weights = {}
    for b in ("1-5", "6-15", "16-40", "40+"):        # 先算全部权重，再归一（避免渐进分母 bug）
        weights[b] = sum(r["n_words"] for r in all_rows
                         if r["bucket"] == b and r["rarity_mean"] is not None)
    tot_w = sum(weights.values()) or 1
    bucket_head = {}
    for b in ("1-5", "6-15", "16-40", "40+"):
        g = [d["bucket_gaps"].get(b) for d in doc_stats]
        g = [x for x in g if x is not None]
        ci = boot_ci(g, clusters={s: [d["bucket_gaps"].get(b) for d in doc_stats
                                      if d["subset"] == s and d["bucket_gaps"].get(b) is not None]
                                   for s in subsets}, rng=rng)
        bucket_head[b] = {"mean_gap": ci[0], "ci95": [ci[1], ci[2]],
                          "n_docs": len(g),
                          "token_share": round(weights[b] / tot_w, 4)}
    primary = sum(bucket_head[b]["mean_gap"] * weights[b] for b in bucket_head
                  if bucket_head[b]["mean_gap"] is not None) / tot_w

    # ---------- 整体 gap（对照）+ 检验 ----------
    gaps_all = [d["zipf_gap_all"] for d in doc_stats if d["zipf_gap_all"] is not None]
    gaps_prose = [d["zipf_gap_prose"] for d in doc_stats if d["zipf_gap_prose"] is not None]
    clusters_all = {s: [d["zipf_gap_all"] for d in doc_stats
                        if d["subset"] == s and d["zipf_gap_all"] is not None] for s in subsets}
    ci_all = boot_ci(gaps_all, clusters=clusters_all, rng=rng)
    ci_prose = boot_ci(gaps_prose, clusters={s: [d["zipf_gap_prose"] for d in doc_stats
                                                 if d["subset"] == s and d["zipf_gap_prose"] is not None]
                                             for s in subsets}, rng=rng)
    wt = wilcoxon_signed_rank(gaps_prose)
    st_ = sign_test(gaps_prose)

    # ---------- 阈值敏感性（行级，pooled） ----------
    thr_sens = {}
    for thr in THRESHOLDS:
        k = f"kept_{thr}"
        d = [r["rarity_mean"] for r in all_rows if not r[k] and r["rarity_mean"] is not None]
        kp = [r["rarity_mean"] for r in all_rows if r[k] and r["rarity_mean"] is not None]
        thr_sens[thr] = {"del_rate": round(sum(1 for r in all_rows if not r[k]) / n_lines, 4),
                         "gap_pooled": round(mean(d) - mean(kp), 3) if d and kp else None}

    # ---------- 操作日志交叉验证 ----------
    import math as _m
    pairs = [(d["ops_removed_lines"], d["judged_deleted"]) for d in doc_stats]
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs) / max(1, len(pairs))
    sx = (sum((x - mx) ** 2 for x in xs) / len(xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / len(ys)) ** 0.5
    corr = cov / (sx * sy) if sx and sy else None
    xval = {"pearson_r": round(corr, 3) if corr is not None else None,
            "median_ops_removed": statistics.median(xs),
            "median_judged_deleted": statistics.median(ys),
            "note": "两者口径不同（操作区间数 vs 模糊匹配行数），只需正相关且量级可解释"}

    # ---------- 删除率（阈值 90） ----------
    k = f"kept_{PRIMARY_THR}"
    rare = [r for r in all_rows if r["rarity_mean"] is not None and r["rare_known_frac"] >= 0.10]
    common = [r for r in all_rows if r["rarity_mean"] is not None and r["rarity_mean"] >= 4.0]
    head = [r for r in all_rows if r["is_heading"] == 1]
    body = [r for r in all_rows if r["is_heading"] == 0]
    dr = lambda seg: round(sum(1 for r in seg if not r[k]) / len(seg), 4) if seg else None

    # ---------- 每子集 gap（跨语料稳健性） ----------
    per_subset = {}
    for s in subsets:
        g = [d["zipf_gap_prose"] for d in doc_stats if d["subset"] == s and d["zipf_gap_prose"] is not None]
        if len(g) >= 30:
            per_subset[s] = {"n": len(g), "gap_prose": round(statistics.mean(g), 3),
                             "share_neg": round(sum(1 for x in g if x < 0) / len(g), 3)}

    summary = {
        "meta": {"n_docs": n_docs, "n_lines": n_lines, "subsets": subsets,
                 "per_subset_docs": dict(collections.Counter(d["subset"] for d in doc_stats)),
                 "seed": SEED, "primary_threshold": PRIMARY_THR},
        "PRIMARY_bucket_weighted_gap": round(primary, 4),
        "bucket_gaps": bucket_head,
        "overall_gaps": {"all": {"mean": ci_all[0], "ci95": [ci_all[1], ci_all[2]]},
                         "prose": {"mean": ci_prose[0], "ci95": [ci_prose[1], ci_prose[2]]}},
        "tests": {"wilcoxon_prose": wt, "sign_test_prose": st_},
        "threshold_sensitivity": thr_sens,
        "ops_cross_validation": xval,
        "per_subset_gap_prose": per_subset,
        "deletion_rates": {"all": dr(all_rows), "rare_known": dr(rare),
                           "high_freq": dr(common), "headings": dr(head), "body": dr(body)},
        "oov": {"deleted": mean([r["oov_frac"] for r in all_rows if not r[k]]),
                "kept": mean([r["oov_frac"] for r in all_rows if r[k]])},
    }
    (REPORTS / "step2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import pandas as pd
        pd.DataFrame(all_rows).to_csv(REPORTS / "step2_lines.csv.gz",
                                      index=False, compression="gzip")
        pd.DataFrame(doc_stats).drop(columns=["bucket_gaps"]).to_csv(
            REPORTS / "step2_docs.csv", index=False)
    except Exception as e:
        print("CSV 写出失败:", e)

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bs = list(bucket_head)
    gv, er = [], []
    for b in bs:
        g = bucket_head[b]["mean_gap"]
        ci = bucket_head[b]["ci95"]
        if g is None:
            continue
        lo = (g - ci[0]) if ci and ci[0] is not None else 0.0
        hi = (ci[1] - g) if ci and ci[1] is not None else 0.0
        gv.append(g); er.append([max(lo, 0), max(hi, 0)])
    axes[0].errorbar(range(len(gv)), gv, yerr=list(zip(*er)) if gv else None,
                     fmt="o", capsize=4)
    axes[0].axhline(0, ls="--", c="grey", lw=0.8)
    axes[0].set_xticks(range(len(bs))); axes[0].set_xticklabels(bs)
    axes[0].set_xlabel("line length (words)"); axes[0].set_ylabel("zipf gap (deleted − kept)")
    axes[0].set_title("gap by length bucket (higher = deleted is rarer)")

    srt = sorted((r for r in all_rows if r["rarity_mean"] is not None),
                 key=lambda r: r["rarity_mean"])
    xs2, ys2 = [], []
    B = 10
    for b in range(B):
        seg = srt[int(b*len(srt)/B): int((b+1)*len(srt)/B)]
        if seg:
            xs2.append(statistics.mean(r["rarity_mean"] for r in seg))
            ys2.append(sum(1 for r in seg if not r[k]) / len(seg))
    axes[1].plot(xs2, ys2, marker="o"); axes[1].invert_xaxis()
    axes[1].set_xlabel("line rarity (mean zipf, lower = rarer)")
    axes[1].set_ylabel("deletion rate")
    axes[1].set_title(f"deletion rate vs rarity (n={n_docs} docs)")
    for ax in axes: ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_stageA_v3.png", dpi=150)

    print(json.dumps({k: summary[k] for k in
                      ("PRIMARY_bucket_weighted_gap", "bucket_gaps", "overall_gaps",
                       "tests", "per_subset_gap_prose", "threshold_sensitivity",
                       "ops_cross_validation")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

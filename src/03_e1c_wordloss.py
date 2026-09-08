# -*- coding: utf-8 -*-
"""
阶段 A · E1c 位置偏置 + 词级消失二分（v1，探索性）

E1c（与行匹配器无关，纯语料属性）
----------------------------------
文档内稀有负载行是否系统性偏居尾部？
统计量：per-doc [mean pos of rare-heavy lines − mean pos of all lines]，Wilcoxon。
注：本脚本跑在探索用样本上，仅作 contact check；确认性 E1c 需预注册 §5 的独立新切分。

词级消失二分（不依赖行匹配器，用多重集 diff）
--------------------------------------------
在 raw 出现但 cleaned 中减少的 token = "消失 token"。
对稀有层（zipf<2.5）的消失 token 做 v1 代理判定：
  knowledge-like := 词典内词(zipf>0) 或 专有名词(原词首字母大写) 或 长度≥7 的内容词
  noise-like     := 其余（纯 OOV 乱码/非字母数字混合/过短残片）
报告：稀有消失 token 中 knowledge 占比 vs 稀有未消失 token 中 knowledge 占比
（若消失集知识占比显著更高 → 尾切带走了知识；反之 cleaner 切的是噪声）。
"""
from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

RARE_ZIPF = 2.5
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")


def e1c_from_lines(fp: Path) -> dict:
    import pandas as pd
    from scipy import stats as st

    df = pd.read_csv(fp)
    df = df[df["rarity_mean"].notna()].copy()
    rare = df[df["rare_known_frac"] >= 0.30]     # 稀有负载行
    gaps = []
    for doc, g in df.groupby("doc"):
        r = rare[rare["doc"] == doc]
        if len(r) >= 3 and len(g) >= 10:
            gaps.append(r["pos"].mean() - g["pos"].mean())
    w = st.wilcoxon(gaps) if len(gaps) > 8 else (None, None)
    return {
        "n_docs": len(gaps),
        "mean_within_doc_pos_gap(rare-heavy − all)": round(statistics.mean(gaps), 4),
        "share_positive": round(sum(1 for x in gaps if x > 0) / len(gaps), 4),
        "wilcoxon": {"stat": getattr(w, "statistic", None), "p": getattr(w, "pvalue", None)},
        "reading": ">0 = 稀有负载行偏居文档尾部（支持 E1c）",
    }


def classify_tokens(counter, zipf_freq) -> dict:
    """把 token 出现次数的 multiset 分类。返回 {kind: count}。"""
    out = collections.Counter()
    for w, n in counter.items():
        if not w:
            continue
        lw = w.lower()
        z = zipf_freq(lw, "en")
        if z > 0:                       # 词典内词
            out["knowledge"] += n
        elif w[0].isupper():            # 首字母大写 OOV → 专有名词候选
            out["knowledge"] += n
        elif len(w) >= 7 and re.fullmatch(r"[a-z]+", w):   # 长小写 OOV → 罕见词候选
            out["knowledge"] += n
        else:
            out["noise"] += n
    return out


def word_loss(parts: dict, per_subset: int, zipf_freq) -> dict:
    import pyarrow.parquet as pq
    from rapidfuzz import fuzz  # noqa: F401  (保留对齐依赖)

    agg = {"n_docs": 0,
           "lost_rare_know": 0, "lost_rare_noise": 0,
           "kept_rare_know": 0, "kept_rare_noise": 0,
           "lost_common": 0, "kept_common": 0}
    doc_stats = []
    for sub, path in parts.items():
        pf = pq.ParquetFile(path)
        done = 0
        for batch in pf.iter_batches(batch_size=512,
                                     columns=["raw_content", "cleaned_content"]):
            for ex in batch.to_pylist():
                if done >= per_subset:
                    break
                raw, cl = ex.get("raw_content") or "", ex.get("cleaned_content") or ""
                if len(raw) < 500:
                    continue
                def _toks(s):
                    return [w for w in WORD_RE.findall(re.sub(r"\s+", " ",
                                                              TAG_RE.sub(" ", s)))]
                rw, cw = _toks(raw), _toks(cl)
                cr, cc = collections.Counter(rw), collections.Counter(cw)
                # 消失 = raw 中比 cleaned 多的那部分（多重集）
                lost = collections.Counter()
                for w, n in cr.items():
                    d = n - cc.get(w, 0)
                    if d > 0:
                        lost[w] = d
                # 稀有层判定
                def is_rare_tok(w):
                    z = zipf_freq(w.lower(), "en")
                    return z > 0 and z < RARE_ZIPF or (z == 0 and not w[0].isupper() and len(w) < 7)
                r_lost = {w: n for w, n in lost.items() if is_rare_tok(w)}
                r_kept = {w: n for w, n in cr.items() if w not in lost and is_rare_tok(w)}
                cl_lost = classify_tokens(r_lost, zipf_freq)
                cl_kept = classify_tokens(r_kept, zipf_freq)
                agg["lost_rare_know"] += cl_lost["knowledge"]
                agg["lost_rare_noise"] += cl_lost["noise"]
                agg["kept_rare_know"] += cl_kept["knowledge"]
                agg["kept_rare_noise"] += cl_kept["noise"]
                agg["n_docs"] += 1
                done += 1
        print(f"  {sub}: {done} docs", flush=True)
    agg["share_knowledge_lost"] = (agg["lost_rare_know"] /
                                   max(1, agg["lost_rare_know"] + agg["lost_rare_noise"]))
    agg["share_knowledge_kept"] = (agg["kept_rare_know"] /
                                   max(1, agg["kept_rare_know"] + agg["kept_rare_noise"]))
    return agg


def main():
    from wordfreq import zipf_frequency
    out = {}
    lines_fp = REPORTS / "step2_lines.csv.gz"
    if lines_fp.exists():
        out["E1c_contact"] = e1c_from_lines(lines_fp)
    import json as _json
    pm = _json.loads((ROOT / "reports" / "part_map.json").read_text())
    parts = {k: v for k, v in pm.items() if Path(v).exists()}
    out["word_loss"] = word_loss(parts, per_subset=500, zipf_freq=zipf_frequency)
    (REPORTS / "step3_e1c_wordloss.json").write_text(
        _json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(_json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

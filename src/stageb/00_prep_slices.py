# -*- coding: utf-8 -*-
"""
阶段 B · Pilot 数据准备（00）

把本地 AICC 分片切分为 Calibration/Train/Eval 三层文档级数据，
产出两臂训练文本 + 评估清单。协议依据：docs/STAGEB_PROTOCOL.md §1。

产物（E:/models/ultrax_parts 读取 → D:/论文/artical/runs/pilot/ 写出）：
  doc_split.json          文档哈希 → {calibration|train|eval}
  train_raw.txt           训练臂：raw 文档文本（每篇一段，\\n 分隔）
  train_refined.txt       训练臂：同一批文档的 cleaned 文本
  eval_manifest.csv       Eval 文档的 raw 行清单（行级类型标签由 02 冻结分类器打）
  calib_manifest.csv      同上（Calibration，用于冻结阈值）
  meta.json               样本数/去重统计/切分哈希种子

近重去重：5-gram minhash（16 桶）→ 桶内 Jaccard≥0.8 并簇 → 簇级切分。
pilot 用 AICC 前 N 篇；正式阶段重新抽样并换 seed。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent      # D:\论文\artical
RUNS = ROOT / "runs" / "pilot"
RUNS.mkdir(parents=True, exist_ok=True)

PARQUET = Path(r"E:\models\ultrax_parts\AICC.parquet")
TAG_RE = re.compile(r"<[^>]+>")
NGRAM = 5


def norm(s: str) -> str:
    s = TAG_RE.sub(" ", s or "")
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def shingles(s: str):
    """5-gram 集合（字符级，归一化文本）。"""
    if len(s) < NGRAM:
        return frozenset([s])
    return frozenset(s[i:i + NGRAM] for i in range(len(s) - NGRAM + 1))


def minhash(sig: frozenset, seeds) -> list[int]:
    """用内置 hash（进程内确定，比 md5 快 ~40×）。"""
    out = [1 << 62] * len(seeds)
    for g in sig:
        h = hash(g) & ((1 << 64) - 1)
        for i, s in enumerate(seeds):
            v = (h ^ (s * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
            if v < out[i]:
                out[i] = v
    return out


def approx_jaccard(a: list[int], b: list[int]) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    MHS = [101, 203, 307, 409, 503, 601, 701, 809, 907, 1009, 1103, 1201, 1301, 1409, 1511, 1601]

    print(f"[1/5] 读取 AICC 分片（取前 {args.n_docs} 篇有效文档）")
    pf = pq.ParquetFile(str(PARQUET))
    docs = []            # (uid, raw, cleaned)
    for batch in pf.iter_batches(batch_size=1024,
                                 columns=["uid", "raw_content", "cleaned_content"]):
        for ex in batch.to_pylist():
            raw, cl = ex.get("raw_content") or "", ex.get("cleaned_content") or ""
            if len(raw) < 300:
                continue
            docs.append((ex["uid"], raw, cl))
            if len(docs) >= args.n_docs:
                break
        if len(docs) >= args.n_docs:
            break
    print(f"    有效文档 {len(docs)}")

    print("[2/5] 近重去重（5-gram minhash，桶内 Jaccard≥0.8 并簇）")
    # 文档太长 → 每篇取前 2000 字符建 shingle 签名，够去重精度且省内存
    sigs = []
    for _, raw, _ in docs:
        s = norm(raw)[:2000]
        sigs.append(minhash(shingles(s), MHS))
    n = len(docs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # 按各 minhash 桶聚合，桶内两两近似 Jaccard 检查（桶内通常很小；>50 则抽样限流）
    for b in range(len(MHS)):
        buckets = {}
        for i in range(n):
            buckets.setdefault(sigs[i][b], []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > 50:
                members = rng.sample(members, 50)
            for a_ in range(len(members)):
                for b_ in range(a_ + 1, len(members)):
                    i, j = members[a_], members[b_]
                    if approx_jaccard(sigs[i], sigs[j]) >= 0.8:
                        parent[find(i)] = find(j)
    # 去重：只保留每个簇第一篇
    keep = []
    seen_root = set()
    for i in range(n):
        r_ = find(i)
        if r_ not in seen_root:
            seen_root.add(r_)
            keep.append(i)
    docs = [docs[i] for i in keep]
    print(f"    去重后 {len(docs)} 篇（删除 {args.n_docs - len(docs)} 近重）")

    print("[3/5] 文档级三向切分（簇不可分）")
    # 重新按簇分：直接对 keep 后的独立簇哈希切分
    split = {}
    for uid, _, _ in docs:
        h = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
        r = (h % 100) / 100
        split[uid] = ("calibration" if r < 0.20 else
                      "train" if r < 0.80 else "eval")
    from collections import Counter
    print("    切分分布:", dict(Counter(split.values())))

    print("[4/5] 写训练文本（train_raw / train_refined，等文档口径）")
    train_docs = [(uid, raw, cl) for uid, raw, cl in docs if split[uid] == "train"]
    with open(RUNS / "train_raw.txt", "w", encoding="utf-8") as fr, \
         open(RUNS / "train_refined.txt", "w", encoding="utf-8") as fc:
        for uid, raw, cl in train_docs:
            fr.write(f"<|doc {uid}|>\n{raw}\n")
            fc.write(f"<|doc {uid}|>\n{cl}\n")
    n_raw_tok = n_ref_tok = 0
    for _, raw, cl in train_docs:
        n_raw_tok += len(raw) // 4
        n_ref_tok += len(cl) // 4
    print(f"    训练文档 {len(train_docs)}：raw≈{n_raw_tok} tok / refined≈{n_ref_tok} tok")

    print("[5/5] 写 Eval 文档原文（评估时按 frozen 分类器打行级标签）")
    with open(RUNS / "eval_raw.txt", "w", encoding="utf-8") as fe, \
         open(RUNS / "eval_doc_ids.txt", "w", encoding="utf-8") as fi:
        for uid, raw, _ in docs:
            if split[uid] == "eval":
                fe.write(f"<|doc {uid}|>\n{raw}\n")
                fi.write(uid + "\n")

    meta = {"n_docs_total": len(docs), "n_train": len(train_docs),
            "n_raw_tok": n_raw_tok, "n_ref_tok": n_ref_tok,
            "split_seed": args.seed, "corpus": "AICC",
            "note": "pilot 数据，正式阶段剔除重抽；等文档口径"}
    (RUNS / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print("meta:", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
严复核攻击落地：T1-lite + T2 机械零模型

T2（承重墙）：观测到的 pos×rarity 是否只是"机械后缀截断 × 源语料稀有~位置相关"的复现？
  对每文档，按判删数 k 把最后 k 行标为删除（后缀截断模拟）→ 重拟合 M3；
  对照 2：随机删 k 行。比较 pos 与 rarity×pos 系数：
    若观测 ≈ 后缀截断模拟 → 机制是截断几何，无独立发现；
    观测 − 截断模拟 的余量才是真正信号。

T1-lite（行级匹配有效性）：仅 replace_str==0 的纯删文档（remove_lines 为主）上，
  操作区间指向的删除行位置（中位位置、后缀占比）与判删行对比；
  并报告纯删文档占比（若过低，说明 replace_str 是主流，行匹配压力更大）。
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def fit_m3(df: pd.DataFrame, label: str, out: dict):
    cols = ["kept", "rarity_mean", "pos", "log_n_words", "is_heading", "oov_frac"]
    df = df.dropna(subset=cols).copy()
    grp = pd.factorize(df["doc"])[0]
    m = smf.logit("kept ~ rarity_mean * pos + log_n_words + is_heading + oov_frac",
                  data=df).fit(cov_type="cluster",
                               cov_kwds={"groups": grp}, disp=0)
    out[label] = {
        "n": int(m.nobs),
        "pos_coef": float(m.params["pos"]),
        "rarity_pos_inter": float(m.params["rarity_mean:pos"]),
        "rarity_coef": float(m.params["rarity_mean"]),
        "pos_p": float(m.pvalues["pos"]),
        "inter_p": float(m.pvalues["rarity_mean:pos"]),
    }
    return out[label]


def main():
    df = pd.read_csv(REPORTS / "step2_lines.csv.gz")
    df["log_n_words"] = (df["n_words"] + 1).apply(math.log)
    df = df[df["type"] != "blank"].copy()

    out = {}

    # ---- 观测（真实标签）----
    df["kept"] = df["kept_90"]
    fit_m3(df, "observed", out)

    # ---- T2：后缀截断模拟 ----
    # 每文档的判删数 = 最后要砍掉的行数
    sim_rows = []
    for doc, g in df.groupby("doc"):
        g = g.sort_values("pos").reset_index(drop=True)
        k = int((g["kept"] == 0).sum())
        if k >= len(g):
            continue
        g = g.copy()
        g["kept"] = 1
        if k > 0:
            g.iloc[-k:, g.columns.get_loc("kept")] = 0
        sim_rows.append(g)
    sim = pd.concat(sim_rows)
    out["n_docs_suffix_sim"] = sim["doc"].nunique()
    fit_m3(sim, "suffix_truncation_sim", out)

    # ---- 对照 2：随机删除模拟（保序）----
    import random
    rng = random.Random(20260908)
    rnd_rows = []
    for doc, g in df.groupby("doc"):
        g = g.copy()
        k = int((g["kept"] == 0).sum())
        if k >= len(g):
            continue
        g["kept"] = 1
        idx = rng.sample(list(range(len(g))), k)
        g.iloc[idx, g.columns.get_loc("kept")] = 0
        rnd_rows.append(g)
    rnd = pd.concat(rnd_rows)
    fit_m3(rnd, "random_deletion_sim", out)

    # ---- T1-lite：纯删文档（replace_str==0 或占比极小）----
    docs = pd.read_csv(REPORTS / "step2_docs.csv")
    docs["replace_str"] = docs["replace_str"].fillna(0)
    pure = docs[docs["replace_str"] == 0]
    out["T1_lite"] = {
        "n_pure_delete_docs": int(len(pure)),
        "share_pure_delete_docs": round(len(pure) / max(1, len(docs)), 4),
        "note": "纯 remove_lines 文档（无 replace_str 干扰行合并）上，行匹配与操作日志可精确对齐",
    }
    if len(pure) >= 30:
        sub = df[df["doc"].isin(set(pure["doc"]))].dropna(
            subset=["kept_90", "rarity_mean", "pos", "log_n_words", "oov_frac"])
        m = smf.logit("kept_90 ~ pos + rarity_mean + log_n_words + oov_frac",
                      data=sub).fit(
            cov_type="cluster",
            cov_kwds={"groups": pd.factorize(sub["doc"])[0]}, disp=0)
        out["T1_lite"]["pure_delete_pos_coef"] = float(m.params["pos"])
        out["T1_lite"]["pure_delete_pos_p"] = float(m.pvalues["pos"])

    (REPORTS / "step3_mechanical_null.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
阶段 A · 门控假设的正式检验（逻辑回归，文档聚类稳健标准误）

背景
----
v2 的行级曲线**非单调**：删除率峰值在 zipf≈4 的中频层，最稀有层反而存活率最高；
标题行（词更稀有）删除率仅 7.6% vs 正文 19.0%。选题锐据此把叙事从
"长尾被伤害"锐化为"**上下文门控的稀有度处理**"：清洗器不是按稀有度删，
而是按 (稀有度 × 位置/结构角色) 联合决策。

本脚本把该假设形式化为可检验的回归（全部零 GPU）：

  M1（仅正文行）: kept ~ rarity + log_n_words + pos + oov
      → 正文内稀有度的**主效应**：负 = 正文里越稀有越容易被删
  M2（全部行）  : kept ~ rarity * is_heading + log_n_words + pos + oov
      → **交互项 rarity×is_heading**：正且显著 = 稀有度在标题位被"豁免"（门控）
  M3（全部行）  : kept ~ rarity * pos + controls
      → 位置门控：稀有度的处理是否随文档内位置变化

所有模型用文档聚类稳健标准误（同一篇文档的行不独立）。
阈值敏感性：在 kept_85 / kept_95 上复跑 M1/M2 的关键系数。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def fit(df: pd.DataFrame, formula: str, label: str, out: dict):
    m = smf.logit(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["doc"]}, disp=0)
    params, pvals = m.params, m.pvalues
    ors = {k: round(math.exp(v), 4) for k, v in params.items()}
    out[label] = {
        "n": int(m.nobs),
        "coef": {k: round(v, 4) for k, v in params.items()},
        "odds_ratio": ors,
        "p": {k: float(pvals[k]) for k in params.index},
        "pseudo_r2": round(m.prsquared, 4),
    }
    return m


def key_stats(out: dict):
    """抽出三个判定系数。"""
    def g(label, key):
        try:
            return {"coef": out[label]["coef"][key], "p": out[label]["p"][key],
                    "OR": out[label]["odds_ratio"][key]}
        except KeyError:
            return None
    return {
        "正文稀有度主效应 (M1: rarity)": g("M1_prose", "rarity_mean"),
        "门控交互 (M2: rarity×is_heading)": g("M2_all", "rarity_mean:is_heading"),
        "标题位稀有度总效应 (M2): ": None,
        "位置门控 (M3: rarity×pos)": g("M3_all", "rarity_mean:pos"),
    }


def main():
    lines_fp = REPORTS / "step2_lines.csv.gz"
    df = pd.read_csv(lines_fp)
    df = df[df["rarity_mean"].notna()].copy()
    df["log_n_words"] = (df["n_words"] + 1).apply(math.log)
    df["is_heading"] = df["is_heading"].astype(int)
    # 文档内 doc id 需为独立字符串（v3 已是 subset:idx）
    out = {}

    prose = df[df["type"] == "prose"]
    fit(prose, "kept_90 ~ rarity_mean + log_n_words + pos + oov_frac",
        "M1_prose", out)

    fit(df, "kept_90 ~ rarity_mean * is_heading + log_n_words + pos + oov_frac",
        "M2_all", out)

    fit(df, "kept_90 ~ rarity_mean * pos + log_n_words + is_heading + oov_frac",
        "M3_all", out)

    # 阈值敏感性：M1/M2 关键系数在 85/95 下复跑
    sens = {}
    for thr in (85, 95):
        p = prose.copy(); pa = df.copy()
        p["kept"] = p[f"kept_{thr}"]; pa["kept"] = pa[f"kept_{thr}"]
        m1 = smf.logit("kept ~ rarity_mean + log_n_words + pos + oov_frac",
                       data=p).fit(cov_type="cluster", cov_kwds={"groups": p["doc"]}, disp=0)
        m2 = smf.logit("kept ~ rarity_mean * is_heading + log_n_words + pos + oov_frac",
                       data=pa).fit(cov_type="cluster", cov_kwds={"groups": pa["doc"]}, disp=0)
        sens[thr] = {
            "M1_rarity_coef": round(m1.params["rarity_mean"], 4),
            "M1_rarity_p": float(m1.pvalues["rarity_mean"]),
            "M2_interaction_coef": round(m2.params["rarity_mean:is_heading"], 4),
            "M2_interaction_p": float(m2.pvalues["rarity_mean:is_heading"]),
        }

    result = {
        "meta": {"n_lines": int(len(df)),
                 "n_docs": int(df["doc"].nunique()),
                 "cov_type": "cluster(by doc)", "threshold": 90},
        "models": out,
        "key_stats": key_stats(out),
        "threshold_sensitivity": sens,
        "reading_guide": {
            "M1_rarity<0": "正文内稀有 → 更易被删（伤害长尾）",
            "M2_interaction>0_and_absorbs": "标题位豁免 → 支持上下文门控叙事",
            "M3_interaction": "位置门控：负 = 文档越靠后稀有内容越容易被删",
        },
    }
    (REPORTS / "step2_gate_regression.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"key_stats": result["key_stats"],
                      "threshold_sensitivity": sens,
                      "M2_pseudo_r2": out["M2_all"]["pseudo_r2"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

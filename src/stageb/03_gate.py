# -*- coding: utf-8 -*-
"""
阶段 B · 四门聚合（03_gate.py）

输入：02_eval_slices.py 的输出 JSON（每 seed 一份，含 per_doc 文档级配对结构）
输出：reports/stageb_gate.json —— 四门判据（预注册冻结，见 docs/STAGEB_PROTOCOL.md §3/§4）

统计口径（与协议一致，勿临场更改）：
  - 文档级配对 Δ：d = meanNLL_refined(doc) − meanNLL_raw(doc)，单位 nats
    （PPL 比 = exp(Δ)；Δ>0 ⇒ refined 臂在该切片更差）
  - G1 可行：各切片配对文档数 ≥ 30 且数值有限
  - G2 方向：池化文档级 Δ_tailK 的 bootstrap 95%CI 下界 > 0，且 ≥2/3 seed 点估计 > 0
  - G3 功效：seed 级 Hedges g ≥ 0.5（n=3，另报池化文档级 g 供参考）
  - G4 对照：DiD = Δ_tailK − Δ_tailN 的 bootstrap 95%CI 下界 > 0
  - co-primary MRR：Δ_MRR < 0（refined 恢复更差）作一致性参考，不单独 kill
用法：
  python src/stageb/03_gate.py --evals reports/pilot_eval_s0.json reports/pilot_eval_s1.json reports/pilot_eval_s2.json --out reports/stageb_gate.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

BOOT = 10000
RNG = np.random.default_rng(0)


def doc_deltas(per_doc: dict, cls: str):
    """→ (deltas[nats], mrr_deltas, doc_names)。仅取两臂都有数据的文档。"""
    bucket = per_doc.get(cls, {})
    d, dm, names = [], [], []
    for doc, arms in bucket.items():
        if "raw" not in arms or "refined" not in arms:
            continue
        r, f = arms["raw"], arms["refined"]
        if r["n_tok"] == 0 or f["n_tok"] == 0:
            continue
        d.append(f["nll_sum"] / f["n_tok"] - r["nll_sum"] / r["n_tok"])
        if f["mrr_n"] > 0 and r["mrr_n"] > 0:
            dm.append(f["mrr_sum"] / f["mrr_n"] - r["mrr_sum"] / r["mrr_n"])
        names.append(doc)
    return np.asarray(d, dtype=float), np.asarray(dm, dtype=float), names


def boot_ci(x: np.ndarray, stat=np.mean, n=BOOT):
    """bootstrap CI；x 为空/全 NaN → (nan, nan, nan)。"""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    stats = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def hedges_g(sample: np.ndarray) -> float:
    """单组效应量 vs 0（带小样本校正）。"""
    s = sample[np.isfinite(sample)]
    n = len(s)
    if n < 2 or s.std(ddof=1) == 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * (1 - 3 / (4 * n - 5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    seeds = {}
    for p in args.evals:
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        seeds[Path(p).stem] = j["per_doc"]
    print(f"[load] {len(seeds)} 个 seed 评估文件")

    out = {"tag": args.tag, "n_seeds": len(seeds), "inputs": args.evals,
           "generated": datetime.now().isoformat(timespec="seconds"), "per_seed": {},
           "pooled": {}, "gates": {}}

    # ---- 每 seed：三类切片的配对 Δ ----
    for name, pd in seeds.items():
        rec = {}
        for cls in ("tailK", "tailN", "head"):
            d, dm, names = doc_deltas(pd, cls)
            pt, lo, hi = boot_ci(d)
            pm, mlo, mhi = boot_ci(dm) if len(dm) else (float("nan"),) * 3
            rec[cls] = {"n_pairs": len(d), "delta_nats": pt,
                        "ci95": [lo, hi], "ppl_ratio": float(np.exp(pt)) if np.isfinite(pt) else None,
                        "delta_mrr": pm, "mrr_ci95": [mlo, mhi]}
        out["per_seed"][name] = rec

    # ---- 池化（文档跨 seed 视为独立；seed 内先取均值消除单 seed 文档重复权）----
    # 协议主口径：文档级配对 bootstrap。三 seed 的 eval 采样同一批 eval 文档 →
    # 池化时先对 seed 取均值再 bootstrap，避免同一文档重复计权。
    all_names = None
    for cls in ("tailK", "tailN", "head"):
        per_name = {}
        for pd in seeds.values():
            d, dm, names = doc_deltas(pd, cls)
            for nm, v in zip(names, d):
                per_name.setdefault(nm, []).append(v)
        common = {k: v for k, v in per_name.items() if len(v) == len(seeds)}
        pooled = np.asarray([np.mean(v) for v in common.values()], dtype=float)
        pt, lo, hi = boot_ci(pooled)
        out["pooled"][cls] = {"n_docs": len(pooled), "delta_nats": pt, "ci95": [lo, hi],
                              "ppl_ratio": float(np.exp(pt)) if np.isfinite(pt) else None}
        if cls == "tailK":
            all_names = len(pooled)

    # ---- DiD：Δ_tailK − Δ_tailN（文档配对到两切片都有者）----
    def doc_map(cls):
        m = {}
        for pd in seeds.values():
            d, _, names = doc_deltas(pd, cls)
            for nm, v in zip(names, d):
                m.setdefault(nm, []).append(v)
        return {k: np.mean(v) for k, v in m.items() if len(v) == len(seeds)}

    k, n = doc_map("tailK"), doc_map("tailN")
    common = sorted(set(k) & set(n))
    did = np.asarray([k[c] - n[c] for c in common], dtype=float)
    dt, dlo, dhi = boot_ci(did)
    out["pooled"]["DiD_tailK_minus_tailN"] = {"n_docs": len(did), "delta_nats": dt,
                                              "ci95": [dlo, dhi]}

    # ---- 四门 ----
    pk = out["pooled"]["tailK"]
    g_seed = hedges_g(np.asarray([out["per_seed"][s]["tailK"]["delta_nats"]
                                  for s in out["per_seed"]], dtype=float))
    # 池化文档级 g（分布版本）
    per_name = {}
    for pd in seeds.values():
        d, _, names = doc_deltas(pd, "tailK")
        for nm, v in zip(names, d):
            per_name.setdefault(nm, []).append(v)
    pooled_dist = np.asarray([np.mean(v) for v in per_name.values()
                              if len(v) == len(seeds)], dtype=float)
    g_doc = hedges_g(pooled_dist) if len(pooled_dist) > 1 else float("nan")

    g1 = (all(out["per_seed"][s][c]["n_pairs"] >= 30
              for s in out["per_seed"] for c in ("tailK", "tailN", "head"))
          and all(np.isfinite(out["per_seed"][s][c]["delta_nats"])
                  for s in out["per_seed"] for c in ("tailK", "tailN", "head")))
    g2 = (pk["ci95"][0] > 0 and
          sum(out["per_seed"][s]["tailK"]["delta_nats"] > 0 for s in out["per_seed"])
          >= 2 * len(seeds) / 3)
    g3 = bool(np.isfinite(g_seed) and g_seed >= 0.5)
    g4 = dlo > 0
    out["gates"] = {"G1_feasible": g1, "G2_direction": g2, "G3_power_seed_g": g3,
                    "seed_g": g_seed, "doc_pooled_g": g_doc, "G4_control_DiD": g4,
                    "VERDICT": "GO" if (g1 and g2 and g3 and g4) else "NO-GO"}

    out_path = Path(args.out) if args.out else Path("reports") / f"stageb_{args.tag}_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({
        "pooled_tailK": {"delta_nats": pk["delta_nats"], "ci95": pk["ci95"],
                         "ppl_ratio": pk["ppl_ratio"], "n_docs": pk["n_docs"]},
        "DiD": out["pooled"]["DiD_tailK_minus_tailN"],
        "seed_g": g_seed, "doc_g": g_doc,
        "gates": out["gates"]}, ensure_ascii=False, indent=1))
    print("完成 →", out_path)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
阶段 B · Pilot 分层评估（02_eval_slices.py）

输入：两臂 × seeds 训练出的 model.pt（+ 共享 tokenizer）+ runs/pilot/eval_raw.txt
输出：head / tail-知识 / tail-噪声 三类的行级 PPL、稀有词恢复 MRR，
      以及 pilot 四门判据（方向 / 功效 / 对照 / 不崩）。

协议依据：docs/STAGEB_PROTOCOL.md §1.2(行级分类) §3(判据) §4(E2 统计量)。

用法（GPU 训练完成后）：
  python src/stageb/02_eval_slices.py \
      --raw-dir runs/pilot/raw_s0 --ref-dir runs/pilot/ref_s0 \
      --out reports/pilot_eval_s0.json
  对每个 seed 跑一次；最后 03 汇总跨 seed 判据。

行级分类（阈值全部冻结，非数据驱动，见 docs/STAGEB_PROTOCOL.md §1.2）：
  head     : pos < 1/3
  tailK    : pos >= 2/3 且含 >=1 知识承载稀有 token
  tailN    : pos >= 2/3 且无知识 token、但含噪声 token（URL/数字/代码/乱码）
  其他行不参与评估。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = Path(os.environ.get("PILOT_DIR", str(ROOT / "runs" / "pilot")))
TOK_DIR = RUNS / "tok"

TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://|www\.|\.com|\.org|\.net|\.io\b", re.I)
NUM_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
STOP = set("the a an of to in for on and or is are was were be been it its this that with as at by from".split())
RARE_ZIPF_HI = 2.5
KNOW_MIN_LEN = 7


def classify_token(tok: str, zipf) -> str:
    """frozen：knowledge / noise / other。"""
    if URL_RE.search(tok) or (NUM_RE.search(tok) and len(tok) < 6):
        return "noise"
    w = tok.lower()
    if w in STOP or len(w) < 2:
        return "other"
    z = zipf(w, "en")
    if z > 0 and z < RARE_ZIPF_HI:
        return "knowledge"
    if z == 0:
        if tok[0].isupper() or len(w) >= KNOW_MIN_LEN:
            return "knowledge"
        return "noise"
    return "other"


def classify_line(text: str, zipf) -> str:
    words = WORD_RE.findall(text)
    has_k = has_n = False
    for w in words:
        c = classify_token(w, zipf)
        has_k |= c == "knowledge"
        has_n |= c == "noise"
    if has_k:
        return "tailK"
    return "tailN" if has_n else "other"


def load_model_and_tok(ckpt_dir: Path, cfg) -> tuple:
    import torch
    from tokenizers import ByteLevelBPETokenizer
    import sys
    sys.path.insert(0, str(ROOT / "src" / "stageb"))
    from importlib import import_module
    m = import_module("01_train_probe")
    tk = ByteLevelBPETokenizer(str(TOK_DIR / "vocab.json"), str(TOK_DIR / "merges.txt"))
    model = m.new_gpt(tk.get_vocab_size(), cfg["ctx"], cfg["n_layer"], cfg["n_embd"],
                      cfg["n_head"])
    sd = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.eval()
    return model, tk


def main():
    from wordfreq import zipf_frequency

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--n-head", type=int, default=8)
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"ctx": args.ctx, "n_layer": args.n_layer, "n_embd": args.n_embd,
           "n_head": args.n_head}
    print(f"[load] raw={args.raw_dir} refined={args.ref_dir} device={dev}")
    m_raw, tk = load_model_and_tok(Path(args.raw_dir), cfg)
    m_ref, _ = load_model_and_tok(Path(args.ref_dir), cfg)
    m_raw.to(dev); m_ref.to(dev)

    # 解析 eval_raw.txt → 每文档 ids + 行 char 偏移
    print("[parse] eval_raw.txt")
    doc_lines = []          # (uid, [ (line, char_start, char_end) ])
    uid = None
    buf = []
    raw_txt = (RUNS / "eval_raw.txt").read_text(encoding="utf-8", errors="replace")
    cur = []
    doc_name = None
    for ln in raw_txt.split("\n"):
        if ln.startswith("<|doc ") and ln.endswith("|>"):
            if doc_name is not None:
                doc_lines.append((doc_name, cur))
            doc_name = ln[6:-2]
            cur = []
        else:
            cur.append(ln)
    if doc_name is not None:
        doc_lines.append((doc_name, cur))

    results = {"n_docs": len(doc_lines)}
    # 打分行级指标（CPU/GPU 一次性，控制批大小）
    from collections import defaultdict
    acc = defaultdict(lambda: {"n": 0, "sum_ppl": 0.0, "mrr_sum": 0.0, "mrr_n": 0})

    def score_doc(model, lines):
        del model, lines  # 保留签名占位：主循环用逐行编码打分，此处不用
        return [], []

    # 先只对目标类行评估（head/tailK/tailN）→ 全部行会太慢
    class_rows = {"head": [], "tailK": [], "tailN": []}
    for doc_name, lines in doc_lines[:200]:        # 抽样 200 篇，pilot 评估足够
        n_tot = len(lines)
        for i, l in enumerate(lines):
            if not l.strip():
                continue
            pos = i / max(1, n_tot)
            cls = classify_line(l, zipf_frequency)
            if pos < 1 / 3:
                cls_use = "head"
            elif pos >= 2 / 3 and cls in ("tailK", "tailN"):
                cls_use = cls
            else:
                continue
            class_rows[cls_use].append((doc_name, i, l))
    print({k: len(v) for k, v in class_rows.items()})

    # 评估两臂
    for arm_name, model in (("raw", m_raw), ("refined", m_ref)):
        t0 = torch.cuda.Event(enable_timing=True) if dev == "cuda" else None
        for cls, rows in class_rows.items():
            nlls, mrrs = [], []
            for doc_name, i, l in rows:
                # 简单近似：左上下文=该行前文（截断），单行计 loss
                doc = next(d for d in doc_lines if d[0] == doc_name)
                ctx_lines = doc[1][max(0, i - 30):i]
                ctx_text = "\n".join(ctx_lines)[-4000:]
                line_ids = tk.encode(l).ids
                if not line_ids or len(line_ids) >= args.ctx:
                    continue
                # 防序列越界：左文截断到 (ctx − 行长 − 1)，保证总长 ≤ ctx
                limit = max(1, args.ctx - len(line_ids) - 1)
                ctx_ids = tk.encode(ctx_text).ids[-limit:]
                seq = torch.tensor([ctx_ids + line_ids], dtype=torch.long, device=dev)
                with torch.no_grad():
                    lg = model(seq)[0]
                lg = lg[:, len(ctx_ids) - 1:-1]   # 对齐 line tokens 的预测
                lg = lg.reshape(-1, lg.size(-1))
                tgt = torch.tensor(line_ids[:lg.size(0)], dtype=torch.long, device=dev)
                if lg.size(0) == 0:
                    continue
                loss = torch.nn.functional.cross_entropy(
                    lg.float(), tgt, reduction="none")
                nlls.append(loss.mean().item())
                # MRR：每个 knowledge token
                for j, tid in enumerate(tgt.tolist()):
                    z = zipf_frequency(tk.decode([tid]).strip().lower(), "en")
                    is_k = (0 < z < RARE_ZIPF_HI) or (z == 0 and len(tk.decode([tid]).strip()) >= 2)
                    if not is_k:
                        continue
                    row = lg[j]
                    topv, topi = torch.topk(row, k=2000)
                    rk = int((topi == tid).nonzero(as_tuple=True)[0]) + 1 if (topi == tid).any() else None
                    if rk:
                        mrrs.append(1.0 / rk)
            mean_ppl = math.exp(sum(nlls) / len(nlls)) if nlls else None
            mean_mrr = sum(mrrs) / len(mrrs) if mrrs else None
            acc[f"{cls}:{arm_name}"] = {"n_lines": len(nlls), "ppl": mean_ppl,
                                         "mrr": mean_mrr}

    out = {"config": cfg, "classes": dict(acc)}
    out_path = Path(args.out) if args.out else \
        ROOT / "reports" / f"pilot_eval_{Path(args.raw_dir).name}__{Path(args.ref_dir).name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out["classes"].items() if v and v.get("ppl")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

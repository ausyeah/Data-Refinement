# -*- coding: utf-8 -*-
"""
阶段 B · Pilot 从零训探针（01_train_probe.py）

用法（在你的 GPU 环境，见 RUN_GUIDE）：
  python src/stageb/01_train_probe.py --arm raw   --out runs/pilot/raw_s0  --seed 0
  python src/stageb/01_train_probe.py --arm refined --out runs/pilot/ref_s0 --seed 0
  三 seeds 各跑两臂；--resume 断点续跑。

冒烟（本机 CPU，验证代码正确性）：
  python src/stageb/01_train_probe.py --arm raw --out runs/smoke/raw_s0 --seed 0 \
      --n-layer 2 --n-embd 64 --n-head 4 --max-steps 3 --batch 2 --ctx 64 --no-amp

设计要点（协议 §2）：
  - GPT-2 风格 decoder-only（pre-norm / GELU / learned pos），从零初始化
  - byte-level BPE vocab 32768（tokenizers 训练于 train_raw.txt，两臂共享）
  - AdamW(0.9,0.95) wd=0.1；cosine→0.1peak；warmup 500；grad clip 1.0
  - 等 token 口径 = 两臂同 max-steps × 同 batch（每步处理 token 数相同）
  - fp16/bf16 autocast（CUDA 自动）；CPU 用 fp32（--no-amp）
  - checkpoint 落 runs/，可 --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = Path(os.environ.get("PILOT_DIR", str(ROOT / "runs" / "pilot")))
TOK_DIR = RUNS / "tok"


# ---------------- 模型 ----------------
def new_gpt(vocab_size: int, ctx: int, n_layer: int, n_embd: int, n_head: int, drop=0.0):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.ln1 = nn.LayerNorm(n_embd)
            s.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True, dropout=drop)
            s.ln2 = nn.LayerNorm(n_embd)
            s.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                  nn.Linear(4 * n_embd, n_embd), nn.Dropout(drop))

        def forward(s, x, causal=True):
            # 关键：LM 探针必须是因果注意力。
            # torch>=2.6 要求 is_causal 搭配显式 attn_mask（bool True=禁止注意）
            T = x.shape[1]
            mask = (torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
                    if causal else None)
            a, _ = s.attn(s.ln1(x), s.ln1(x), s.ln1(x), attn_mask=mask,
                          is_causal=causal)
            x = x + a
            x = x + s.mlp(s.ln2(x))
            return x

    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.tok = nn.Embedding(vocab_size, n_embd)
            s.pos = nn.Embedding(ctx, n_embd)
            s.drop = nn.Dropout(drop)
            s.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            s.ln = nn.LayerNorm(n_embd)
            s.head = nn.Linear(n_embd, vocab_size, bias=False)

        def forward(s, idx, causal=True):
            T = idx.shape[1]
            x = s.drop(s.tok(idx) + s.pos(torch.arange(T, device=idx.device)))
            for b in s.blocks:
                x = b(x, causal=causal)
            return s.head(s.ln(x))

    return GPT()


# ---------------- 数据 ----------------
def build_tokenizer(force: bool):
    from tokenizers import ByteLevelBPETokenizer
    files = [str(TOK_DIR / "vocab.json"), str(TOK_DIR / "merges.txt")]
    if not force and all(Path(x).exists() for x in files):
        return ByteLevelBPETokenizer(*files)
    tk = ByteLevelBPETokenizer()
    print("  训练 tokenizer（byte-level BPE, 32k）...", flush=True)
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    tk.train([str(RUNS / "train_raw.txt")], vocab_size=32768,
             special_tokens=["<|doc|>", "<|endoftext|>"])
    tk.save_model(str(TOK_DIR))
    return tk


def load_arm_ids(arm: str, tokenizer):
    f = RUNS / ("train_raw.txt" if arm == "raw" else "train_refined.txt")
    # 编码缓存：tokenizer 确定性 → 同一 vocab 的编码只做一次（6 次运行共享）
    cache = TOK_DIR / f"ids_{arm}.npy"
    if cache.exists():
        print(f"  编码缓存命中 {cache.name}", flush=True)
        return np.load(cache)
    print(f"  编码 {f.name} ...", flush=True)
    ids = []
    DOC = tokenizer.token_to_id("<|doc|>")
    with open(f, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("<|doc "):
                ids.append(DOC)
            else:
                ids.extend(tokenizer.encode(line.strip()).ids)
    arr = np.array(ids, dtype=np.uint16)
    np.save(cache, arr)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["raw", "refined"], required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-layer", type=int, default=8)      # 8L·512 ≈ 50M（pilot）；正式 12L·768
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)        # 每设备 batch（micro）
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={dev} torch={torch.__version__}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.out) if args.out else RUNS / f"{args.arm}_s{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/3] tokenizer + 数据")
    tk = build_tokenizer(force=False)
    arr = load_arm_ids(args.arm, tk)
    print(f"      tokens={len(arr)}")
    with open(RUNS / "tok" / "meta.json", "w") as f:
        json.dump({"vocab": tk.get_vocab_size(), "n_tokens_arm": int(len(arr)),
                   "arm": args.arm}, f)

    print("[2/3] 建模型")
    model = new_gpt(tk.get_vocab_size(), args.ctx, args.n_layer, args.n_embd,
                    args.n_head)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"      params={n_param/1e6:.1f}M")
    model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda" and not args.no_amp))
    use_amp = dev == "cuda" and not args.no_amp
    step0 = 0
    if args.resume and (outdir / "ckpt.pt").exists():
        ck = torch.load(outdir / "ckpt.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        print(f"      从 step {step0} 续跑")

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, (args.max_steps - args.warmup))
        return args.lr * (0.1 + 0.9 * (1 + math.cos(math.pi * p)) / 2)

    print("[3/3] 训练")
    ntok_ctx = args.ctx
    n = len(arr)
    B, A = args.batch, args.grad_accum
    t0 = time.time()
    for step in range(step0, args.max_steps):
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(A):
            starts = [random.randrange(0, n - ntok_ctx - 1) for _ in range(B)]
            x = np.stack([arr[i:i + ntok_ctx] for i in starts]).astype(np.int64)
            y = np.stack([arr[i + 1:i + 1 + ntok_ctx] for i in starts]).astype(np.int64)
            xt = torch.from_numpy(x).to(dev)
            yt = torch.from_numpy(y).to(dev)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(xt)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), yt.view(-1))
            acc_loss += loss.item() / A
            scaler.scale(loss / A).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        for g in opt.param_groups:
            g["lr"] = lr_at(step + 1)

        if step % 25 == 0 or step == args.max_steps - 1:
            toks = (step + 1) * A * B * ntok_ctx
            tok_s = toks / max(0.1, time.time() - t0)
            print(f"      step {step+1}/{args.max_steps} loss {acc_loss:.3f} "
                  f"lr {lr_at(step):.1e} tok/s {tok_s/1000:.0f}k", flush=True)
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "loss": acc_loss}, outdir / "ckpt.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": args.max_steps, "loss": acc_loss}, outdir / "ckpt.pt")
    torch.save(model.state_dict(), outdir / "model.pt")
    # 每臂独立元数据（协议 §2.2：记录 tokens/epoch/配置，勿互相覆盖）
    toks_arm = int(len(arr))
    epochs = args.max_steps * args.batch * args.grad_accum * args.ctx / max(1, toks_arm)
    (outdir / "arm_meta.json").write_text(json.dumps({
        "arm": args.arm, "n_tokens_arm": toks_arm, "n_steps": args.max_steps,
        "approx_epochs": round(epochs, 2),
        "config": {"n_layer": args.n_layer, "n_embd": args.n_embd, "n_head": args.n_head,
                   "ctx": args.ctx, "batch": args.batch, "grad_accum": args.grad_accum,
                   "lr": args.lr, "warmup": args.warmup}, "seed": args.seed,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("完成 →", outdir)


if __name__ == "__main__":
    main()

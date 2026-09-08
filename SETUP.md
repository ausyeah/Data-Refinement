# Environment Setup

> Target machine: Windows 11 laptop, single RTX 4060 8GB.
> **Design rule: nothing in this project needs more than this card.**

---

## 1. Put HuggingFace cache on a large drive (important)

The default HF cache lives on `C:` and will fill up. This project redirects it to `E:`.

```bash
# Git Bash / WSL
export HF_HOME=/e/hf-cache          # Windows path: E:\hf-cache
export HF_ENDPOINT=https://hf-mirror.com   # China mirror (optional but fast)
```

```powershell
# PowerShell
$env:HF_HOME = "E:\hf-cache"
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

To make it permanent, add the two lines to `~/.bashrc` (Git Bash) or
`setx HF_HOME E:\hf-cache` (PowerShell, permanent).

**Why**: `openbmb/UltraX-Preview` is ~487GB total. We never download it fully —
every script uses `streaming=True` so only the samples actually read are fetched.
Still, keep the cache off `C:`.

## 2. Install

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Stage-B training dependencies (torch / transformers / peft / bitsandbytes / accelerate)
are commented out in `requirements.txt` and should be installed only after
Stage A passes the go/no-go gate.

## 3. Run Stage A, step 0

```bash
python src/00_probe_ultrax.py --n 200
```

This **streams** the first N records of `openbmb/UltraX-Preview`, prints the real field
structure, and writes `reports/step0_schema.json` + `reports/step0_samples.csv`.
It intentionally makes no assumption about the schema — the field names for
*raw text / refined text / edit operations* are confirmed by inspection, not guessed.

## 4. What "done" looks like for step 0

- `reports/step0_schema.json` exists and lists every field with type + sample value;
- You can point at exactly three fields: raw text, refined text, and edit operations;
- The edit-operation field's structure is understood (string? list? typed labels?).

If the dataset turns out **not** to contain edit-operation traces, Stage A's premise
fails and the plan falls back to controlled self-injection (with dose calibration —
see `docs/RESEARCH_PLAN.md` §5.3).

## 5. Disk layout used in this project

| Path | Purpose |
|---|---|
| `E:\hf-cache` | HuggingFace cache (`HF_HOME`) |
| `E:\models` | local model weights |
| `D:\...\artical\` | this repository (code + small outputs only) |

Large intermediates (`reports/*.parquet`, `*.csv`) are gitignored by design.

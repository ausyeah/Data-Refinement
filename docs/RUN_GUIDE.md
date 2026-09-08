# Pilot 运行指南（在你自己的终端跑 GPU）

> **为什么不能在这里跑**：WorkBuddy 会话沙箱拿不到 GPU（nvidia-smi 报 NVML 初始化失败），
> 但代码已用 CPU 冒烟验证（tokenizer → 训练 → checkpoint 全链路通过）。
> **请在普通终端（PowerShell / CMD）里执行以下命令。**

## 0. 前置（一次性）

```powershell
# 1) 用与脚本一致的 Python（或你自己的）安装 CUDA 版 torch
#    4060 Laptop 驱动较新，可试 cu124；报错再降 cu121
C:\Users\26315\.workbuddy\binaries\python\envs\default\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu124
# 国内慢的话可加镜像环境变量或分文件下载；tokenizers 已装好

# 2) 验证
C:\Users\26315\.workbuddy\binaries\python\envs\default\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 预期：True NVIDIA GeForce RTX 4060 Laptop GPU
```

## 1. 跑 pilot（两臂 × 3 seeds = 6 次训练）

```powershell
cd D:\论文\artical
$env:HF_HOME = "E:\hf-cache"
$env:PILOT_DIR = "D:\论文\artical\runs\pilot"     # 数据已备好（00_prep_slices 产物）

# 每臂每 seed 一个进程；建议一次只跑一个（8GB 卡装不下两个）
C:\Users\26315\.workbuddy\binaries\python\envs\default\Scripts\python.exe src\stageb\01_train_probe.py --arm raw      --out runs\pilot\raw_s0 --seed 0
C:\Users\26315\.workbuddy\binaries\python\envs\default\Scripts\python.exe src\stageb\01_train_probe.py --arm refined  --out runs\pilot\ref_s0 --seed 0
# ... seed 1、2 同理（raw_s1/ref_s1/raw_s2/ref_s2）
```

- 默认配置 ≈50M（8L·d512·8H），ctx 512，micro-batch 8 × grad-accum 8 = 32k tok/步。
- **等 token 口径自动满足**：两臂同 max-steps × 同 batch（每步处理 token 相同）。
- 断点续跑：同一命令加 `--resume`。
- 每 500 步存一次 checkpoint（`ckpt.pt`），结束另存 `model.pt`。

### 预期墙钟
4060 上 50M 模型约 0.5–1.5 s/步（先看首 25 步打印的 tok/s 再推算），
2000 步 ≈ 30–50 分钟/次，6 次 ≈ 3–5 小时，**晚上挂着跑**。

## 2. 跑完把结果发回来
每臂输出目录的 `model.pt` + `ckpt.pt` 与打印的 loss 曲线；我跑 02 评估（分层 PPL + 稀有词恢复 + 四门判据）。

## 3. 正式阶段（pilot 通过后）
- 重新抽样换 seed 重跑 `00_prep_slices`（pilot 数据按协议不进正式分析）；
- 模型升 150–300M（--n-layer 12 --n-embd 768 --n-head 12，8GB 需确认显存 < 7GB）。

## 故障排查
| 现象 | 处理 |
|---|---|
| OOM | 减 `--batch`（8→4）或 `--grad-accum`（8→4），ctx 512 先别动 |
| CUDA out of memory（tokenizer 阶段不会） | 同上 |
| torch cu124 装不上 | 换 cu121 / cu118；或装 cpu 版先跑通逻辑再补 cuda |
| tokenizer 报 `<|doc|>` 找不到 | 确认 PILOT_DIR 指向 runs\pilot（00 产物所在） |

# Stage B Protocol — 从零训探针 + 分层诊断（pilot → 正式）

> 版本 1.0 · 2026-09-08 · 撰写：严复核（robustness-auditor）
> 依赖：`configs/preregistration.md`（v1.1，E2 = 裁决点）、阶段 A 产物（F1–F5、T1/T2、step3_e1c_wordloss）。
> 状态：**待 team-lead 拍板后并入预注册 §4**；任何阈值在本文件 vs 预注册冲突时，以预注册为准。

## 0. 设计目标与裁决链（把 E2 落到可执行）

阶段 B 用「从零训 150–300M 探针」把审计侧的**切片代价**翻译成**下游可测代价**。

### 0.1 为何"从零训"而非"已预训练模型 + CPT/LoRA"（方法理由，写作时按此措辞）

> 早期内部措辞"续训对语料差异钝感"被搜文献否决为**不可辩护的普适命题**
> （DAPT, Gururangan ACL'20, 2004.10964 在中规模域内数据即有显著增益；2026 年 Soro / RedSage
> 等 CPT 在 B 级 token 预算内即推动分布）。正式方法理由改为三条可辩护依据：

1. **基座锚定（anchoring）**：raw 与 refined 共享约 85% 的行（阶段 A 诊断），
   追加训练的差分信号被已预训练权重锚定淹没——CPT 下的 raw/refined 差异极小且不可归因；
2. **遗忘偏置（forgetting bias）**：Chang et al. (NeurIPS'24, arXiv 2406.11813) 证明
   事实获取是增量概率上升 + 遗忘幂律，**低暴露/被稀释的边际事实最先被遗忘**——
   CPT 恰好擦除我们想测的尾部信号，故 CPT 非但不能探测尾部知识、还主动删除它；
3. **协议镜像**：UltraX 自身的验证即"1B 从零训、16B vs 20B tokens"——审计对方声明须用对方协议，
   且从零训消除基座混杂，与本科防泄漏纪律一致。

（对照引用：DAPT 2004.10964 作"CPT 有效"的对照组；2406.11813 作"CPT 对长尾有遗忘偏置"的依据。）

- **裁决点**：精炼（refined）训练在 **tail-知识承载行**上显著劣于 raw 训练。
  主统计量 Δ_tailK > 0（refined 对同一批评估行的 PPL 更高 / 稀有词恢复更差），bootstrap 95% CI 下界 > 0。
- **Kill**：Δ_tailK ≤ 0（CI 上界 ≤ 0）→ 走 TOST 等价性检验，发 informative null。
- **语境检查（非裁决）**：head 分层 Δ 是否 < 0（聚合收益集中在 head）——按预注册"若复现"处理，
  head 不通过不 kill，但会改变论文叙事。

---

## 1. 切片构造：训练切片 vs 评估切片，如何不"测的就是训过的"

### 1.1 单元与三向切分
以 **raw 原始文档**为不可再分单元（refined 是其确定性变换，不能把同一文档的 raw 与 refined
分别放进两臂——那会造成文档级泄漏）。按 raw 文档哈希三向切分：

| 切分 | 占比 | 用途 |
|---|---|---|
| Calibration | 20% | 冻结分箱阈值、知识词判定、δ_min；**不计入最终 p 值** |
| Train | 60% | raw 臂喂 raw 全文；refined 臂喂同批文档的 cleaned 全文（文档集合相同，仅文本不同） |
| Eval | 20% | **两臂都从未见过的文档**，只用于终评一次 |

### 1.2 行级分类（在 Calibration 上冻结，不在 Eval 重定）
对 Eval 文档的行（raw 形式）打三类标签，全部冻结后一次性使用：

- **head 行**：pos < 1/3 且非标题/OOV 行；
- **tail-knowledge 行**：pos ≥ 2/3，且含 ≥1 个**知识承载稀有 token**；
- **tail-noise 行**：pos ≥ 2/3，无知识承载 token 但含噪声 token（URL/数字/乱码/代码）。
- 另存 **稀有度分箱**（zipf<2.0/2.0–2.5/2.5–3.0/≥3.0）供 E2 敏感性。

**知识承载 token**（沿用 step3 判定并收紧，冻结于 Calibration）：
词典内词 zipf∈(0,2.5) 的内容词、或首字母大写 OOV（专名候选）、或 ≥7 字母小写 OOV；
**排除** URL/纯数字/代码片段/单字母残片/停用词。判定器在 Calibration 上选阈值，写死。

### 1.3 防循环：训练与评估不相交的硬规则
1. Eval 文档**整体不出现在**两臂任何训练文本中（含 refined 形式）。
2. **簇级去重**：Calibration/Train/Eval 之间做 5-gram Jaccard ≥0.8 的近重簇去重——web 语料去重不彻底时，
   仅靠 doc-hash 会被近重文档击穿（refined 的删除模式会经近重文档迁移进 Eval）。
3. 分箱阈值与知识词表只由 Calibration 决定；Eval 只被读一次、只产聚合统计量。
4. 报告 Train↔Eval 的 minhash 重叠率，作为访问纪律证据。

### 1.4 评估切片的公平性（谁占便宜）
- **主评估 = deletion-blind（全行，raw 形式文本锚点）**：两臂对**同一份** Eval raw 行 token 流打分
  （各自带同文档左侧上下文）。文本、tokenizer、上下文完全一致 → 比较公平。
- 关键点：Eval 含被删内容时，**系统性不利于 refined**——refined 训练分布里没有这类文本。
  这正是**被测量对象**，不是 bug。要把"格式新颖性"与"知识损失"分开，靠两个对照：
  (a) **tail-noise 对照**：若 Δ_tailK ≈ Δ_tailN，说明 refined 只是没见过"尾部这种格式/写法"，
      知识叙事死亡，降级为 register/格式效应；
  (b) **head 对照**：若 Δ_head 也显著 > 0，说明 refined 探针整体更差（不是切片问题）。
- **辅评估 = kept-anchored（仅 refined 保留、语义保留的行）**：敏感性分析。若 Δ_tailK 在
  保留行上仍然 > 0 → refined 连自己保下来的尾部知识都没学好（更强）；若只在被删行上 > 0 → 机制成立。
  保留集在 Eval 里往往样本少（尾部删得多），只作辅助、不作主裁决。

---

## 2. 训练协议（150M from-scratch）

### 2.1 默认配置（pilot 实测后校准）
| 项 | 默认 | 备注 |
|---|---|---|
| 架构 | 12L · d=768 · 12H · ~150M | 300M 为正式规模上限，先过 pilot |
| context | 512 | 8GB 上最稳；1024 留作敏感性 |
| tokenizer | **自训 byte-level BPE，vocab 32k**，仅用 Calibration+Train 的 raw 样本（~1GB）训练 | 两臂共享、Eval 词表外留 buffer；兜底用 GPT-2 BPE（代价：OOV/专名分词差，但两臂同等受损） |
| 优化器 | AdamW (β=0.9,0.95), wd=0.1, fp16/bf16 | master weight fp32 |
| schedule | cosine → 0.1×peak，warmup 500 step | peak LR ~1e-3，pilot 微调 |
| 全局 batch | 512 seq × 512 ctx ≈ 262k tok/step | micro-batch 8 + grad accum，梯度裁剪 1.0 |
| 显存 | ~3–4GB（150M bf16 + Adam fp32 副本 + grad checkpoint） | 余量给激活，pilot 上报峰值 |

### 2.2 两臂公平性（最容易翻车处）
- **主口径 = 等 token**：两臂各训练 S 个 token（refined 需多取文档才能凑齐 S——会多看不同文档，
  单列为协变量记录）。
- **敏感口径 = 等文档**：同文档数、refined 训练 token 更少。
- **严格相同**：架构、优化器、schedule、总 step、峰值 LR 完全一致；仅输入文本不同。
- 报告两臂实际 distinct-doc 数、epoch 数——若 refined 因去重后更"干净"导致有效样本不足，需说明。

### 2.3 seeds 与方差分解
- 3–5 seeds：同一训练协议、仅换 data order + init seed。
- 方差分解：总方差 = **seed 间（训练噪声）** + **seed 内行抽样**。
  报告：每 seed 行级 bootstrap 的 mean Δ_tailK → seed 内 SE；跨 seed 的 SD → seed 间噪声。
  主 CI 保守做法：把 seed 当簇做**簇 bootstrap**；另报"seed 作随机效应、df=n_seeds−1 的 t 区间"作敏感性。
  **若 seed 间 SD 与 seed 内 SE 同量级或更大 → 结论不可靠，扩大 seeds 而非扩行。**

---

## 3. Pilot（50–100M）——方向门 + 功效门 + 对照门 + 可行性门

> 预算示例（数值为示例，pilot 首步先测 token/s 再倒算）：50–100M 模型、每臂 150–300M token、
> ctx 512、batch 512 → ~600–1150 step；8GB 笔记本 bf16 下 ~1–3s/step → 每臂/seed 约 20–60 分钟；
> 2 臂 × 3 seeds ≈ 2–6 小时/语料。**只用信号最强的一个语料**（阶段 A：AICC 或 RedPajama）。

### 3.1 Pilot 判据（全部通过才上正式）
| 门 | 指标 | 阈值 |
|---|---|---|
| 可行性 | token/s、峰值显存、无 OOM | 墙钟可接受；显存 < 7GB |
| 方向门 | Δ_tailK = refined−raw 的 PPL 差（Eval 行级，doc 簇 bootstrap） | 95% CI 下界 > 0 |
| 功效门 | seed 级标准化效应（用 3 seed 均值的 Hedges' g） | g ≥ 0.5 才够格上正式；g<0.2 放弃/换语料 |
| 对照门 | Δ_tailK − Δ_tailN | > 0（知识 vs 格式可分离） |
| 不崩门 | head Δ 不显著为正（若显著为正 → 探针整体失效，先修协议） | — |

- **δ_min 在 pilot 后预注册**：提议 δ_min = 0.05 × raw 臂在 tailK 的 mean PPL（相对恶化 5%），
  等价性检验用此界。最终值须在阶段 B 正式结果产生前写入预注册修订日志。
- Pilot 用的行/文档**全部剔除**出正式分析的任何切分（不能把 pilot Eval 并入正式 Eval）。

---

## 4. E2 确认性统计量与分箱（预注册衔接）

- **主统计量**：Δ_tailK = mean over Eval tailK 行 [PPL_refined − PPL_raw]，
  行内按 seed 配对（同一 seed 的两臂之差），行级 + seed 簇双重重采样，95% CI 下界 > 0 即支持。
- **共主（更知识特异）**：稀有词恢复——对 tailK 行内每个知识 token，给定左侧上下文算其在
  下一 token 分布中的 rank / MRR；Δ_MRR = MRR_refined − MRR_raw < 0。
  PPL 被功能词主导，恢复率更能抓到"稀有知识是否被学到"。两个指标方向一致才算通过。
- **分箱敏感性**：tailK 再按 zipf 分箱（2.0/2.5/3.0）与按语料分箱（4 语料），BH-FDR q=0.05。
- **E1c 衔接**：E2 的 tailK 切片里稀有知识 token 的分布偏置按预注册 E1c（token 级 + 文档内置换零分布）
  在正式数据上先验一次——E1c null 则"尾部有知识"不成立，E2 即便显著也只讲"格式损失"。

---

## 5. 防泄漏清单（执行纪律）

1. raw/refined 同源 → **文档级**三向切分 + **簇级**近重去重（1.3）。
2. 评估用 raw 形式文本、两臂共享、tokenizer 共享（2.1）。
3. 阈值/词表只在 Calibration 定；Eval 只终评一次。
4. PPL 与 NLL 是**因变量**，不是用来选行/分箱的**难度代理**（与预注册 §4 禁 NLL 的用意不冲突；
   若审稿人质疑，正文写明"NLL 仅作 outcome，未参与任何切片或阈值选择"）。
5. pilot 数据不进正式分析。
6. 不拟合缩放律；≤2B 结论仅作符号/方向外推。

---

## 6. 审稿人最可能打死的 3 点 + 缓解

1. **"refined 更差只是因为没见过尾部格式"（register/格式伪影）**
   缓解：tail-noise 为格式匹配对照；tailK vs tailN 必须可分离；稀有词恢复率作共主指标；
   kept-anchored 敏感性区分"被删"与"没学好"。
2. **从零训 150M 在 8GB/小预算下欠训 → 切片差异 = seed 噪声**
   缓解：seed 级方差分解与门控（§2.3/§3.1）；功效门 g≥0.5；若 seed 噪声主导则结论不成立，先扩 seeds。
3. **"知识承载词"分类是研究者拍脑袋，tailK 是事后挑出来的**
   缓解：分类阈值在 Calibration 冻结、禁止 Eval 上重定；tailK 定义 + δ_min 写进预注册修订日志并打时间戳；
   用 PopQA 长尾做**外部锚点**（方向一致即可，不入主统计量）。

# MathGPT A800 训练报告

> 训练日期: 2026-04-03 ~ 2026-04-04
> 硬件: NVIDIA A800-SXM4-80GB | PyTorch 2.7.1+cu128 | Python 3.11 | bf16

---

## 一、训练流程概览

| 阶段 | 状态 | 耗时 | 关键指标 |
|------|------|------|----------|
| 1/5 数据下载 (100 shards) | ✅ 完成 | ~30min | 101 shards, ~10GB (从 VPN 机器中转) |
| 2/5 Tokenizer 训练 | ✅ 完成 | ~3min | BPE, vocab=32768 |
| 3/5 Base 预训练 (5000步) | ✅ 完成 | **893 分钟 (~14.9h)** | BPB: 3.138 → **0.732** |
| 4/5 SFT 微调 (375步/3000步) | ✅ 完成(提前结束) | ~67 分钟 | BPB: 0.492 → **0.329** |
| 5/5 RL 强化学习 (699/699步) | ✅ 完成 | ~240 分钟 (~4h) | Pass@1: 2.5% → **14.0%** (峰值) → 12.3% (最终) |

**总训练时间**: ~20+ 小时 (不含环境搭建和数据传输)

---

## 二、对比 GTX 1080 第一轮训练

| 维度 | GTX 1080 (第一轮) | A800 (第二轮) | 提升 |
|------|-------------------|--------------|------|
| 模型参数量 | 73.5M (depth=6) | ~700M (depth=20) | **~10x** |
| 上下文长度 | 256 tokens | 2048 tokens | **8x** |
| 预训练数据 | ~24M tokens (8 shards) | ~2.6B tokens (100 shards) | **~108x** |
| 计算精度 | float32 | bfloat16 | 2x 效率 |
| Base BPB | 1.356 | **0.732** | **-46%** |
| SFT BPB | 0.961 | **0.329** | **-66%** |
| RL Pass@1 | 2.75% (峰值) | **14.0%** (峰值), 12.3% (最终) | **+5.1x** |
| RL Pass@16 | 未记录 | **38.0%** (峰值), 32.8% (最终) | — |
| 数学测试 | 8/8 全错，循环 | 复杂题有推理能力，简单题反而截断 | 部分质变 |

---

## 三、各阶段详细分析

### 3.1 Base 预训练

**配置**: depth=20, head-dim=128, n_embd=1280, seq=2048, batch=524K, 5000步

| 指标 | 值 |
|------|-----|
| 总训练 tokens | 2,621,440,000 (~2.6B) |
| Tokens:Params 比 | 6.02 |
| 训练时间 | 893.34 分钟 (~14.9小时) |
| 吞吐量 | ~48,800 tok/sec (稳定) |
| MFU (bf16) | **45.2%** |
| GPU 显存使用 | 76,314 MiB / 81,920 MiB (93%) |
| GPU 利用率 | 100% |

**Loss 曲线**:
| 阶段 | Loss | 说明 |
|------|------|------|
| step 0 | 10.398 | 随机初始化 |
| step 1763 (35%) | 2.828 | 稳定下降 |
| step 4999 (100%) | 2.501 | 收敛 |

**BPB**: 3.138 → **0.732** (下降 76.7%)

**硬件利用分析**: MFU 45.2% 对于单卡 A800 是优秀水平。A800 SM 8.0 无法使用 Flash Attention 3 (需 Hopper SM 9.0)，使用 SDPA 回退。每步耗时稳定在 ~10.7s。

**Base 模型生成样例** (step 5000):
- "The capital of France is Paris, the city of the 21st century" ✅ 基本常识正确
- "The chemical symbol of gold is Au. The atomic number is 79" ✅ 事实准确
- "If 5*x + 3 = 13, then x is 13" ❌ 数学推理仍然不对

### 3.2 SFT 微调

**配置**: 3000步计划，实际跑了 375 步 (脚本参数 `--num-iterations=3000` 但实际 SFT 总迭代为 375)

> 注: SFT 实际步数由数据量决定 — 1,301,335 rows × 1 epoch ÷ 524,288 batch ≈ 375 步。数据不够多轮迭代。

**训练混合数据**:
| 数据集 | 行数 | 用途 |
|--------|------|------|
| SmolTalk | 460K | 通用对话 |
| MMLU (x5 epochs) | ~500K | 学术知识 |
| GSM8K (x8 epochs) | ~60K | 数学推理 |
| SimpleSpelling | 200K | 拼写 |
| SpellingBee | 80K | 拼写 |
| Identity | ~2K | 身份 |
| **总计** | **~1.3M rows** | |

**训练指标**:
| 指标 | Step 0 | Step 200 | Step 375 (最终) |
|------|--------|----------|----------------|
| Val BPB | 0.492 | — | **0.329** |
| train/loss | 0.933 | 1.098 | 0.984 |
| tok/sec | 9,301 | 48,970 | 48,989 |
| MFU | 8.60% | 45.30% | 45.32% |

**Benchmark 评估 (TensorBoard)**:

| 评估指标 | Step 200 | Step 375 | 变化 |
|----------|----------|----------|------|
| chatcore | 0.257 | **0.296** | +15% |
| chatcore_cat | 0.154 | **0.189** | +23% |
| ARC-Easy | 0.435 | **0.481** | +11% |
| ARC-Challenge | 0.336 | **0.366** | +9% |
| MMLU | 0.325 | **0.329** | +1% |
| GSM8K | 0.083 | **0.125** | +50% |
| HumanEval | 0.000 | **0.083** | 从0到有 |
| SpellingBee | 1.000 | **1.000** | 满分 |

**关键发现**:
- GSM8K 从 8.3% 提升到 12.5%，说明 SFT 阶段数学能力有初步建立
- HumanEval 从 0 到 8.3%，模型学会了一定的代码生成
- ARC 系列提升显著，说明基础模型已具备一定的常识推理

**遇到的问题**: SFT 在 step 200 做 eval 时，`run_chat_eval` 需要在线下载 ARC/HumanEval 数据集，因 A800 服务器无外网导致崩溃 (RuntimeError: Cannot send a request)。后通过添加 `--offline` 参数和本地数据集修复。前后折腾了约 8 小时。

### 3.3 RL 强化学习 (已完成)

**配置**: 3 epochs, 16 samples/题, max_new_tokens=512, examples-per-step=32, 699 步

**Pass@k 评估 (GSM8K test, 400题, temperature=1.0)**:

| Step | Pass@1 | Pass@4 | Pass@8 | Pass@16 | 说明 |
|------|--------|--------|--------|---------|------|
| 0 (SFT基线) | 2.50% | 11.50% | 18.25% | 29.75% | 起点 |
| 60 | 7.00% | 19.50% | 27.25% | 35.50% | 快速上升 |
| **120** | **14.00%** | **24.00%** | **29.75%** | **38.00%** | **全局最优** |
| 180 | 11.00% | 25.25% | 30.25% | 35.75% | 开始波动 |
| 240 | 12.00% | 20.25% | 27.25% | 33.00% | |
| 300 | 12.00% | 19.50% | 24.75% | 30.75% | Pass@16 下降 |
| 360 | 12.25% | 23.00% | 29.25% | 36.00% | 回升 |
| 420 | 10.50% | 20.75% | 24.75% | 32.00% | |
| 480 | 14.00% | 21.25% | 27.00% | 32.25% | |
| 540 | 12.50% | 22.25% | 26.75% | 33.25% | |
| 600 | 10.00% | 21.25% | 27.25% | 31.75% | 后期退化 |
| **660** | **12.25%** | **23.00%** | **27.75%** | **32.75%** | **最终评估** |

**关键观察**: Pass@1 在 step 120 达到峰值 14.0% 后再未突破，后 540 步（77%训练量）全部在平台期波动。最终 step 660 的各项指标均低于 step 120，说明**最优 checkpoint 在 step 120 而非最终步**。

**Train Reward 趋势** (完整):

| 训练区间 | 平均 Reward | Min | Max |
|----------|------------|-----|-----|
| step 0-96 | 0.181 | 0.06 | 0.33 |
| step 96-192 | 0.192 | 0.11 | 0.37 |
| step 192-288 | 0.220 | 0.11 | 0.45 |
| step 288-384 | 0.271 | 0.13 | 0.49 |
| step 384-480 | 0.302 | 0.18 | 0.52 |
| step 480-576 | 0.318 | 0.16 | 0.53 |
| step 576-698 | **0.355** | 0.20 | 0.55 |

**Train reward 持续上升但 eval pass@k 停滞** — 这是典型的 RL 过拟合信号：模型在训练集题目上越来越好，但泛化能力没有提升。

**序列长度变化**: 222 → 129 tokens (模型学会了更短的回答，但也可能是截断/不完整回答)

**学习率衰减**: 1.0 → 0.001 (cosine decay 到接近零)

### 3.4 RL 训练后实际对话效果

训练完成后在 chat_cli 中测试：

| 输入 | 输出 | 评价 |
|------|------|------|
| "Solve: 2x + 5 = 17" | "...subtracting 5...2x=15...x=3" (正确应为6) | ❌ 推理步骤正确但计算错误 |
| "10 + 10" | "To solve the equation 10 + 10 = " (截断) | ❌ 回答不完整 |
| "10 + 10 = ?" | "10 + 10" (截断) | ❌ 回答不完整 |

**问题分析**:
1. **简单算术题反而回答不了** — RL 训练专注于 GSM8K 格式（需要 `#### answer` 标记），模型对非 GSM8K 格式的简单问题反而退化
2. **序列截断严重** — seq_len 从 222 降到 129，模型倾向于生成极短序列，很多回答在中途截断
3. **`<|assistant_end|>` 过早出现** — 模型学会了提前结束生成，可能是 RL 奖励压力导致（短回答容易包含 `#### number` 格式从而拿到 reward）

---

## 四、问题与挑战总结

### 4.1 环境搭建问题

| 问题 | 耗时 | 解决方案 |
|------|------|---------|
| A800 无法访问 HuggingFace | ~8h | 从 VPN 机器下载数据 → scp 传输到 A800 |
| `pyarrow`/`requests`/`datasets` 缺失 | ~30min | 逐个安装依赖 |
| SFT eval 在线下载崩溃 | ~4h | 添加 `--offline` 参数，所有 task 支持离线加载 |
| VPN 机器上传速度慢 (10KB/s) | — | 改为从 ubuntu 机器直接下载 |

### 4.2 训练过程问题

| 问题 | 影响 | 说明 |
|------|------|------|
| SFT 实际只跑 375 步 | 低于预期的 3000 步 | 数据集 1.3M rows / 524K batch = 375 步。需增加数据或减小 batch |
| RL Pass@1 在 step 120 后平台期 | 后续步数效益递减 | 10%-14% 间波动，没有稳定上升 |
| RL 大量 reward=0 样本 | 梯度信号稀疏 | 用户日志显示很多 example 的 loss=-0.000000, reward=0.000 |

---

## 五、核心发现与分析

### 5.1 成绩

1. **Base 模型质量大幅提升**: BPB 0.732 vs 1.356，说明 2.6B tokens + 700M 模型奠定了扎实的语言基础
2. **SFT 效果显著**: val BPB 降至 0.329，各 benchmark 全面提升
3. **RL 有一定效果**: Pass@1 从 2.5% 提升到 14.0% (5.6x)，train reward 持续上升
4. **硬件利用充分**: MFU 45.2%，GPU 利用率 100%，VRAM 93% 使用
5. **全流程跑通**: 从数据下载到 Base→SFT→RL→chat 推理的完整 pipeline 已验证

### 5.2 不足

1. **SFT 步数不足**: 计划 3000 步仅跑了 375 步，数学数据只有 GSM8K 8 epochs (~60K rows)，在 524K batch 下迭代次数太少
2. **RL 严重过拟合**: train reward 持续上升 (0.07→0.45) 但 eval pass@k 在 step 120 后停滞，最优 checkpoint 在训练 17% 处而非结尾
3. **RL 导致通用能力退化**: 简单算术题（10+10）无法回答，回答截断严重，模型过度适配 GSM8K 的 `#### answer` 格式
4. **Pass@16 从 38% 下降到 32.8%**: 多样性在减少（mode collapse 趋势），RL 后期模型探索能力降低
5. **预期 vs 实际**: TRAINING_LOG.md 预期 Pass@1 达 15-30%+，实际峰值 14% 勉强触达下限
6. **Tokens:Params 比偏低**: 2.6B tokens / 700M params = 3.7，远低于 Chinchilla 最优的 ~20

---

## 六、后续调优建议

### 优先级 1: 扩大 SFT 数据和步数 (影响最大)

**问题**: SFT 仅 375 步是当前最大瓶颈。GSM8K 8 epochs 仅 ~60K rows，在 524K batch 下只贡献了极少的有效梯度更新。

**建议**:
- **降低 SFT 的 total-batch-size** 到 65536 (device-batch-size=32 × 1 grad_accum)。这样 1.3M rows 可以产生 ~3000+ 步，充分学习
- **增加 GSM8K epochs 到 16-20**，数学数据占比要更大
- **添加更多数学数据集**: MATH、AIME、AQuA-RAT 等，扩充推理样本多样性
- **添加 MetaMathQA / Orca-Math**: 这些数据集专门为数学推理设计，可以显著提升 SFT 效果

```bash
# 修改建议
$RUN scripts.chat_sft \
    --total-batch-size=65536 \     # 降低 batch size, 增加步数
    --num-iterations=3000 \        # 确保跑满 3000 步
    --gsm8k-epochs=16 \            # 更多数学数据
    ...
```

### 优先级 2: 优化 RL 训练 (突破 14% 平台期)

**问题**: 大量 example reward=0, loss=0，梯度信号稀疏。Pass@1 在 10-14% 波动。

**建议**:
- **增加 num-samples 到 32 或 64**: 更多采样增加"至少答对一次"的概率，降低方差
- **使用 KL 惩罚**: 防止 policy 崩塌、保持探索多样性 (Pass@16 在下降说明探索在减少)
- **添加 reward shaping**: 不仅给最终答案 0/1，还给中间推理步骤部分奖励 (格式正确+0.1, 使用计算器+0.1, 步骤逻辑合理+0.2 等)
- **课程学习 (Curriculum)**: 先在简单数学题 (1-2步) 上 RL，逐步增加难度
- **降低学习率**: 当前 RL 的 lr_multiplier 从 1.0 衰减到 0.31 可能太快，可以尝试固定较低学习率

### 优先级 3: 增加预训练数据 (长期)

**问题**: Tokens:Params 比仅 3.7，远低于最优值。

**建议**:
- **增加到 200-400 shards** (约 15-30B tokens)，达到 Chinchilla 最优比例
- 或者**减小模型到 depth=12 (~300M)**，让 2.6B tokens 训练更充分
- 当前 Base BPB 0.732 仍有下降空间

### 优先级 4: 架构和工程优化

| 优化项 | 预期收益 | 难度 |
|--------|---------|------|
| 升级到 H100/H800 (Hopper) | Flash Attention 3, MFU 60%+ | 换硬件 |
| 多卡并行 (2-4x A800) | 训练速度线性提升 | 中等 |
| 使用 FSDP/DeepSpeed | 可训练更大模型 | 中等 |
| GQA (n_kv_head < n_head) | 推理速度提升，VRAM 降低 | 改配置 |
| 开启 gradient checkpointing | 可增大 batch size | 低 |

### 优先级 5: 评估体系完善

- 添加 MATH 数据集评估 (更难的数学)
- 添加 few-shot 评估 (当前是 zero-shot)
- 分难度统计 GSM8K 准确率 (1步题 vs 5步题)
- 记录 RL 训练中正确答案的典型 pattern

---

## 七、推荐的下一步训练计划

**最小改动方案** (建议先跑这个):

```bash
# 1. 从当前 SFT checkpoint 重新跑 SFT，降低 batch size
$RUN scripts.chat_sft \
    --total-batch-size=65536 \
    --num-iterations=3000 \
    --gsm8k-epochs=16 \
    --mmlu-epochs=5 \
    --offline=data/hf_datasets \
    --run=dummy

# 2. 然后重新跑 RL
python3 -m scripts.train_rl \
    --num-epochs=3 \
    --device-batch-size=16 \
    --examples-per-step=32 \
    --num-samples=32 \
    --max-new-tokens=512 \
    --eval-every=60 \
    --eval-examples=400 \
    --save-every=60 \
    --offline=data/hf_datasets \
    --run=dummy
```

**预期**: SFT 3000 步充分训练后，RL 起点应该比 Pass@1=2.5% 高很多 (预期 15-20%)，RL 训练有更大的提升空间，最终 Pass@1 目标 **25-35%**。

---

## 八、各阶段训练日志与指标详解

### 8.1 Base 预训练日志解读

**典型日志**:
```
step 04999/05000 (99.98%) | loss: 2.501108 | lrm: 0.05 | dt: 10740.16ms | tok/sec: 48,815 | bf16_mfu: 45.16 | epoch: 1 pq: 58 rg: 56 | total time: 893.34m | eta: 0.2m
```

**各字段含义**:

| 字段 | 含义 | 本例值 | 说明 |
|------|------|--------|------|
| `step 04999/05000` | 当前步 / 总步数 | 第 4999 步 | 即将完成 |
| `loss: 2.501108` | **交叉熵损失 (nats)** | 2.501 | 对 token 的预测难度，见下方详解 |
| `lrm: 0.05` | 学习率乘子 | 0.05 | cosine decay 尾部，已衰减到峰值的 5% |
| `dt: 10740.16ms` | 每步耗时 | 10.74 秒 | 包含 8 次 grad accumulation 的前后向 |
| `tok/sec: 48,815` | token 吞吐量 | ~49K/s | 每秒处理的 token 数 |
| `bf16_mfu: 45.16` | 模型 FLOPS 利用率 | 45.16% | 占 A800 bf16 峰值算力(312 TFLOPS)的比例 |
| `epoch: 1` | 数据 epoch | 1 | 5B tokens 数据只过了 ~52% (2.6B/5B) |
| `pq: 58 rg: 56` | parquet 文件 / 行组索引 | — | 数据读取进度 |
| `total time` | 已训练时间 | 893 分钟 | ~14.9 小时 |
| `eta` | 预计剩余时间 | 0.2 分钟 | 即将结束 |

#### loss 2.501 是好还是坏？

**结论: 2.501 是正常的，而且是好的。**

这里的 loss 是 **token 级别的交叉熵 (nats)**，不是 BPB。需要区分：

| 指标 | 初始值 | 最终值 | 含义 |
|------|--------|--------|------|
| **loss (nats)** | 10.398 | **2.501** | 每个 **token** 的预测损失 |
| **BPB (bits per byte)** | 3.138 | **0.732** | 每个 **字节** 的预测损失 |

为什么 loss=2.501 看起来"很高"但实际很好？

1. **随机猜测的 loss = ln(32768) = 10.40**。从 10.40 降到 2.501，模型已经学会了大部分语言规律
2. **loss 2.501 → 困惑度 (perplexity) = e^2.501 = 12.2**。意思是模型平均在 12 个候选 token 中犹豫不决——对于一个 32768 词表的 700M 模型来说，这是合理的
3. **BPB 0.732 才是真正衡量模型质量的指标**，因为每个 token 平均编码了多个字节。BPB < 1.0 意味着模型的压缩能力优于每字节 1 bit
4. **参考值**：GPT-2 (1.5B) 的 validation loss ~3.3，Chinchilla (70B) ~1.9。700M 模型达到 2.5 是在合理区间内

**loss 并不需要"接近 1"——那是不同的尺度。** loss=1.0 意味着 perplexity=2.7，对 700M 模型来说几乎不可能达到（需要数十B参数+数T tokens训练）。

### 8.2 SFT 日志解读

**典型日志**:
```
step 00200 (53.37%) | loss: 1.107496 | lrm: 0.93 | dt: 10696.00ms | tok/sec: 49,017 | mfu: 45.34 | epoch: 1 | total time: 33.90m
```

SFT 的 loss 含义与 Base 类似，但有关键区别：

- **SFT 只计算 assistant 回复部分的 loss**（prompt/user 部分用 `ignore_index=-1` 掩码掉了）
- 所以 SFT loss 1.10 **不能**直接和 Base loss 2.50 比较——SFT 只预测回答部分，天然更容易
- **BPB 从 0.492 降到 0.329**，说明模型在对话格式上的预测能力提升了 33%

**为什么 SFT loss 先升后降 (0.93 → 1.46 → 0.98)?**
- Step 1 的 loss=0.93 偏低，因为第一个 micro-batch 碰巧比较简单
- Step 7 的 loss=1.46 是正常水平——模型刚从 base 切换到对话格式，需要适应新分布
- 最终稳定在 ~0.98，说明模型学会了对话格式

### 8.3 RL 日志解读 (重点)

**典型日志**:
```
Step 485/699 | Ex 0  | Pass 0 | loss: 0.000139  | reward: 0.562
Step 485/699 | Ex 1  | Pass 0 | loss: -0.000687 | reward: 0.688
Step 485/699 | Ex 2  | Pass 0 | loss: -0.000000 | reward: 0.000   ← 问题在这里
Step 485/699 | Ex 15 | Pass 0 | loss: -0.000000 | reward: 0.000   ← 问题在这里
Step 485/699 | reward: 0.2773 | seq_len: 131.4                    ← 整步汇总
```

**各字段含义**:

| 字段 | 含义 |
|------|------|
| `Step 485/699` | 当前优化步 / 总步数 (3 epochs × ~233 步/epoch) |
| `Ex 0` | 第 0 个 example（每步处理 32 个 example） |
| `Pass 0` | 第 0 个 micro-batch pass（16 个 sample / device_batch_size=16 = 1 pass） |
| `loss` | **该 example 的 policy gradient loss** |
| `reward` | 该 example 16 个采样的 **平均 reward**（0.0~1.0，答对=1，答错=0） |
| `reward: 0.2773` | 该步所有 32 个 example 的**平均 reward** |
| `seq_len: 131.4` | 生成序列的平均长度 |

#### 为什么 loss = -0.000000 且 reward = 0.000？这正常吗？

**这是 RL 后期的常见现象，不是 bug，但说明训练遇到了瓶颈。**

RL 的 loss 计算过程（代码 `train_rl.py:253-258`）：

```python
# 1. 对同一道题生成 16 个回答
# 2. 每个回答得到 reward（答对=1.0，答错=0.0）
# 3. 计算 advantage = reward - mean(reward)
advantages = rewards - rewards.mean()    # REINFORCE baseline

# 4. 计算 policy gradient
logp   = -model(inputs, targets)         # 每个 token 的 log 概率
pg_obj = (logp * advantages).sum()       # 加权求和
loss   = -pg_obj / num_valid             # 取负 → 最小化 = 最大化好回答的概率
```

**当 reward=0.000 且 loss=-0.000000 时，发生了什么？**

`reward: 0.000` 意味着这道题的 **16 个采样全部答错**。此时：

```
rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
advantages = rewards - mean = [0, 0, 0, ..., 0] - 0 = [0, 0, ..., 0]
```

**所有 advantage 都是 0** → `logp × 0 = 0` → **loss = 0**

这意味着：**模型从这道题得不到任何梯度信号**。不知道应该往哪个方向调整。

#### 这是好还是坏？

| 情况 | 含义 | 是否正常 |
|------|------|---------|
| reward=0, loss=0 | 16个采样全错，无梯度 | ⚠️ **正常但不理想** |
| reward=1.000, loss=0 | 16个采样全对，无梯度 | ✅ 正常（题太简单，已学会） |
| reward=0.562, loss≠0 | 部分对部分错，有梯度 | ✅ **最理想**（学习最有效） |

**在 Step 485 的 32 个 example 中**：

从用户贴的日志可以看到，大量 example 的 reward=0.000 (16 个采样全错)。这说明：

1. **模型对大部分题仍然无能为力** — 即使采样 16 次也答不对
2. **有效学习只发生在少数 reward 在 0~1 之间的 example 上** — 比如 reward=0.562 (9/16 对)、reward=0.875 (14/16 对)
3. **这解释了 Pass@1 在 10-14% 平台期的原因** — 容易的题已经学会(reward→1)，难的题完全答不上(reward→0)，能提供有效梯度的"刚好够得着"的题越来越少

#### reward 字段的具体含义

`reward: 0.562` = 16 个采样中有 9 个答对 → 9/16 = 0.5625

这种情况 advantage 的分布是：
```
答对的 advantage = 1.0 - 0.5625 = +0.4375  → 增大这些回答的概率
答错的 advantage = 0.0 - 0.5625 = -0.5625  → 减小这些回答的概率
```
这才是有效的学习信号！

### 8.4 各阶段 Loss 对比总结

| 阶段 | Loss 类型 | 初始值 | 最终值 | 好坏判断标准 |
|------|----------|--------|--------|-------------|
| Base | 交叉熵 (nats/token) | 10.40 | **2.50** | 越低越好，理论下限 0。2.50 对 700M 模型正常 |
| SFT | 交叉熵 (nats/token, 仅 assistant) | 0.93 | **0.98** | 比 Base 低是因为只算回答部分。~1.0 正常 |
| RL | Policy gradient loss | 不固定 | **~0** | **不是越低越好！** RL loss≈0 说明没有学习信号 |

**关键理解**: 
- Base/SFT 的 loss 是**越低越好**（模型预测越准）
- RL 的 loss **不是越低越好**。RL 的 loss=0 说明没有学习发生。理想状态是 loss 有正有负地波动，同时 reward 在上升

### 8.5 BPB vs Loss 的关系

```
BPB (bits per byte) = loss (nats/token) × bytes_per_token的倒数 × (1/ln2)
```

简化理解：因为一个 token 平均编码了 ~3-4 个字节，所以 **BPB ≈ loss / 3.4 / ln(2) ≈ loss × 0.42**

| Loss (nats/token) | BPB (bits/byte) | 含义 |
|-------------------|-----------------|------|
| 10.40 (随机) | 3.14 | 比随机猜字节还差 |
| 2.50 (最终) | **0.73** | 每字节只需 0.73 bit 编码，压缩率很好 |
| 1.00 (极好) | ~0.29 | 需要数十B参数才能达到 |

---

## 九、资源消耗统计

| 资源 | 消耗 |
|------|------|
| GPU 时间 (A800) | ~20+ 小时 (全部完成) |
| 峰值 GPU 显存 | 73,600 MiB / 81,920 MiB |
| 磁盘 (数据) | ~10 GB (100 shards) + ~2 GB (HF datasets) |
| 磁盘 (checkpoints) | ~10 GB (base 5个 + sft + rl 8个) |
| 训练 FLOPs (Base) | 7.57 × 10^18 |
| 总训练 tokens | ~2.6B (base) + ~196M (SFT) + RL |

---

## 十、RL 训练完整复盘：为什么 Step 120 是最优？

### 10.1 核心结论

**RL 训练 699 步中，只有前 120 步（17%）是有效的，剩余 579 步（83%）不仅没有提升，反而导致了模型退化。**

完整 Pass@1 轨迹：
```
Step   0: 2.50%  ← SFT 基线
Step  60: 7.00%  ← 快速上升
Step 120: 14.00% ← ★ 全局最优
Step 180: 11.00% ← 开始下降
Step 240: 12.00%
Step 300: 12.00%
Step 360: 12.25%
Step 420: 10.50%
Step 480: 14.00% ← 偶尔回升但不稳定
Step 540: 12.50%
Step 600: 10.00% ← 明显退化
Step 660: 12.25% ← 最终评估
Step 698: (未评估，但 train reward=0.45)
```

### 10.2 为什么 Step 120 最好？——RL 过拟合的完整分析

**现象**: Train reward 持续上升 (0.07 → 0.45)，但 eval pass@k 在 step 120 后停滞甚至下降。

这是经典的 **RL 过拟合 (policy overfitting)**，原因链如下：

```
SFT 基础薄弱 (仅 375 步)
    ↓
模型起点能力有限 (Pass@1 = 2.5%)
    ↓
RL 前期: 低垂果实多，容易学到正确 pattern → Pass@1 快速上升到 14%
    ↓
RL 中期: 容易题已学会(reward→1)，难题完全不会(reward→0)
         → 有效梯度信号的题越来越少
    ↓
RL 后期: 模型开始"作弊"——学到的不是推理能力，而是：
         1. 更短的序列 (222→129 tokens) → 提前生成 <|assistant_end|>
         2. 训练集特定题目的记忆 → train reward 上升
         3. 在训练集见过的简单 pattern 上过拟合
    ↓
结果: train reward ↑ 但 eval 能力 ↓ (泛化失败)
```

**数据证据**：

| 指标 | Step 0 | Step 120 (最优) | Step 698 (最终) | 说明 |
|------|--------|----------------|-----------------|------|
| Train reward | 0.07 | ~0.21 | **0.45** | 持续上升（过拟合） |
| Eval Pass@1 | 2.5% | **14.0%** | ~12.3% | 先升后降 |
| Eval Pass@16 | 29.8% | **38.0%** | 32.8% | 多样性在丢失 |
| 序列长度 | 222 | ~170 | **129** | 回答越来越短 |
| LR multiplier | 1.0 | 0.83 | 0.001 | 学习率衰减 |

**Pass@16 下降是关键信号**：从 38% 降到 32.8%，说明模型不是"学得更精准"，而是"探索变少了"(mode collapse)。好的 RL 训练应该 Pass@1 上升的同时 Pass@16 不下降。

### 10.3 实际对话效果验证

用最优 checkpoint (step 120) 和最终 checkpoint (step 698) 分别测试：

| 输入 | Step 120 输出 | Step 698 输出 |
|------|--------------|---------------|
| "2x + 5 = 17" | "...2x = 12" (截断，未给最终答案) | "...x = 3" (答案错误，应为 6) |
| "10 + 10" | 截断 | "To solve the equation 10 + 10 = " (截断) |
| "10 + 10 = ?" | 截断 | "10 + 10" (截断) |

**两个 checkpoint 都表现不佳**：
- Step 120: 推理方向正确但提前截断
- Step 698: 偶尔能给出完整回答但计算错误

**根本原因不在 RL，在 SFT**：模型在 SFT 阶段就没有充分学会"完整地回答问题"这个基本能力。RL 只是在这个薄弱的基础上做微调，无法弥补 SFT 的不足。

### 10.4 学习率衰减与最优点的关系

RL 使用 cosine decay: `lr = initial_lr × (1 - step/699)`

| 阶段 | LR 乘子 | 状态 |
|------|---------|------|
| Step 0-120 | 1.0 → 0.83 | 学习率充足，有效学习 |
| Step 120-350 | 0.83 → 0.50 | 学习率仍可观，但有效样本不足 |
| Step 350-699 | 0.50 → 0.001 | 学习率太低+过拟合，无法逃出局部最优 |

学习率在 step 120 时仍有 83% 峰值，说明不是"学习率太高导致不稳定"，而是**有效训练信号耗尽**。

---

## 十一、下一步调优方案（按优先级）

### 11.1 最高优先级：重做 SFT（预期收益最大）

**当前瓶颈**: SFT 仅 375 步，是整个 pipeline 最薄弱的环节。

**问题根源**: `total-batch-size=524288` 太大。1.3M 数据行 ÷ 524K = 仅 ~2.5 步/epoch，总共只迭代了 375 步。模型还没学会完整的对话格式和推理链。

**具体方案**:

```bash
# 降低 batch size → 增加迭代次数
python3 -m scripts.chat_sft \
    --total-batch-size=65536 \        # 从 524K 降到 65K
    --num-iterations=3000 \           # 目标 3000 步
    --gsm8k-epochs=16 \               # 数学数据翻倍
    --mmlu-epochs=5 \
    --offline=data/hf_datasets \
    --run=sft_v2
```

**预期效果**:
- 1.3M rows ÷ 65K batch = ~20 步/epoch → 3000 步 = ~150 epochs 充分迭代
- SFT 完成后，直接用 `chat_cli --source sft` 测试，应该能给出完整、连贯的回答
- SFT 后的 GSM8K Pass@1 预期从 12.5% 提升到 **20-25%**（不需要 RL）

### 11.2 次高优先级：优化 RL 训练策略

在 SFT 充分训练后再进行 RL，并做以下改进：

**A. 早停 (Early Stopping)**

当前训练没有早停，导致 83% 的训练量浪费在过拟合上。

```python
# 在 train_rl.py 中加入早停逻辑
best_pass1 = 0
patience = 3  # 连续 3 次 eval 没有提升就停止
no_improve_count = 0

# 在 eval 之后:
if pass1 > best_pass1:
    best_pass1 = pass1
    no_improve_count = 0
    save_best_checkpoint()
else:
    no_improve_count += 1
    if no_improve_count >= patience:
        print("Early stopping triggered")
        break
```

**B. KL 散度惩罚（防止 mode collapse）**

当前 RL 没有 KL 约束，模型可以任意偏离 SFT 分布。Pass@16 从 38% 降到 32.8% 就是 mode collapse 的证据。

```python
# 在 loss 计算中加入 KL 惩罚
kl_coeff = 0.01
with torch.no_grad():
    ref_logp = -ref_model(inputs, targets, loss_reduction='none')
kl_penalty = (logp - ref_logp).mean()
loss = -pg_obj + kl_coeff * kl_penalty
```

需要保留一份 SFT 模型作为 reference model。

**C. 增加采样数 + Reward Shaping**

| 改进 | 当前值 | 建议值 | 原因 |
|------|--------|--------|------|
| num_samples | 16 | **32** | 更多采样 → 更多 reward≠0 的题 → 更多有效梯度 |
| max_new_tokens | 512 | **768** | 避免模型因 token 限制截断回答 |
| reward | 0/1 二元 | **部分奖励** | 给中间步骤格式正确、使用计算器等行为部分分数 |

```bash
python3 -m scripts.train_rl \
    --source sft \
    --num-samples=32 \
    --max-new-tokens=768 \
    --eval-every=30 \           # 更频繁 eval 以便早停
    --num-epochs=1 \            # 先跑 1 epoch 看效果
    --offline=data/hf_datasets \
    --run=rl_v2
```

**D. 学习率调整**

当前 `init-lr-frac=0.05` + cosine decay 意味着 RL 从 SFT 学习率的 5% 开始，然后一路降到 0。建议：

```bash
--init-lr-frac=0.02          # 更小的初始 LR，减少初期震荡
```

或者用恒定小学习率（需改代码），避免 cosine decay 尾部学习率过低。

### 11.3 中期：增加训练数据

| 数据集 | 大小 | 用途 | 获取方式 |
|--------|------|------|---------|
| MetaMathQA | 395K rows | 数学 SFT | HuggingFace: meta-math/MetaMathQA |
| Orca-Math | 200K rows | 数学 SFT | HuggingFace: microsoft/orca-math-word-problems-200k |
| MATH dataset | 12.5K rows | 更难的数学 RL | HuggingFace: hendrycks/competition_math |
| NuminaMath-CoT | 860K rows | 带思维链的数学 | HuggingFace: AI-MO/NuminaMath-CoT |

加入这些数据后重新 SFT，可以显著提升数学推理的起点能力。

### 11.4 长期：架构和工程改进

| 改进 | 预期效果 | 工作量 |
|------|---------|--------|
| 增加 Base 预训练到 200 shards (~15B tokens) | BPB 0.73 → ~0.55 | 需 ~30h GPU |
| 添加 gradient checkpointing | 可增大 RL batch size | 改几行代码 |
| 多卡并行 RL (2-4× A800) | RL 训练速度 2-4x | torchrun 配置 |
| 实现 PPO 替代 REINFORCE | 更稳定的 RL 训练 | 中等代码量 |

### 11.5 推荐的完整重训命令

```bash
# === Phase 1: 重做 SFT (最关键) ===
python3 -m scripts.chat_sft \
    --total-batch-size=65536 \
    --num-iterations=3000 \
    --gsm8k-epochs=16 \
    --mmlu-epochs=5 \
    --offline=data/hf_datasets \
    --run=sft_v2

# 先测试 SFT 效果
NANOCHAT_BASE_DIR=./runs python3 -m scripts.chat_cli --source sft

# SFT 评估
NANOCHAT_BASE_DIR=./runs python3 -m scripts.eval_report --source sft --quick --offline data/hf_datasets

# === Phase 2: RL (在 SFT 效果确认后) ===
python3 -m scripts.train_rl \
    --source sft \
    --num-epochs=1 \
    --num-samples=32 \
    --max-new-tokens=768 \
    --eval-every=30 \
    --save-every=30 \
    --examples-per-step=32 \
    --init-lr-frac=0.02 \
    --offline=data/hf_datasets \
    --run=rl_v2

# RL 评估 (对比所有 checkpoint)
NANOCHAT_BASE_DIR=./runs python3 -m scripts.eval_report --all-steps --offline data/hf_datasets
```

**预期最终效果**: SFT 充分训练后 Pass@1 ~20-25%，再经过有早停的 RL → Pass@1 **30-40%**，对话时能给出完整且正确的数学推理。

---

## 十二、经验总结

### 12.1 关键教训

1. **SFT 是 RL 的地基** — RL 无法弥补 SFT 的不足。375 步 SFT 是本轮训练效果不佳的根本原因
2. **RL 需要早停** — 不加监控的 RL 训练会过拟合。train reward 上升 ≠ 模型变好
3. **batch size 不是越大越好** — SFT 阶段 524K 的 batch size 导致数据只迭代了极少次数
4. **eval 频率要够高** — 每 60 步 eval 一次仍然太少，可能错过了 step 90-120 之间的真正最优点
5. **Pass@16 是过拟合的早期预警** — 当 Pass@16 开始下降而 train reward 仍在上升时，应该停止训练

### 12.2 本轮训练的价值

尽管最终模型效果不理想，本轮训练的核心价值在于：

1. ✅ **完整 pipeline 验证**: Base → SFT → RL → Chat 全流程跑通
2. ✅ **性能基线建立**: 知道了 700M 模型 + 2.6B tokens 的能力上限在哪里
3. ✅ **问题定位清晰**: 明确了 SFT 步数不足是首要瓶颈
4. ✅ **调优方向明确**: 下一轮只需改一个参数 (降 SFT batch size) 就能有质的提升
5. ✅ **工具链完善**: 评估脚本、TensorBoard 日志、报告生成全部就位

# MathGPT A800 训练报告 v2

> 训练日期: 2026-04-04 (从 SFT 开始，复用 v1 Base checkpoint)
> 硬件: NVIDIA A800-SXM4-80GB | PyTorch 2.7.1+cu128 | Python 3.11 | bf16
> 脚本: `scripts/full_train_a800_v2.sh`

---

## 一、v2 训练目标

v1 训练发现模型效果不理想（简单算术题截断、计算错误），经过复盘定位到**根本原因是 SFT 步数不足**。v2 的核心目标是验证这一诊断并解决问题。

### 1.1 v1 的问题

| 问题 | 证据 | 影响 |
|------|------|------|
| SFT 只跑了 375 步（计划 3000） | chat_sft.py bug：数据 1 epoch 用完自动停止，忽略 `--num-iterations` | 模型没学会完整对话格式 |
| RL 过拟合 | train reward 0.07→0.45 但 eval pass@k 在 step 120 后停滞 | 83% 训练量浪费 |
| 通用能力退化 | "10+10" 截断，"2x+5=17" 计算错误 | 最终模型不可用 |
| Pass@16 下降 (38%→32.8%) | mode collapse，探索多样性丢失 | RL 后期有害 |

### 1.2 v2 的核心假设

**假设**: SFT 从 375 步增加到 3000 步后，模型能学会完整的对话格式和推理链，RL 在此基础上训练效果会有质的飞跃。

**验证标准**:

| 指标 | v1 实际值 | v2 目标值 | 判断标准 |
|------|----------|----------|---------|
| SFT 实际步数 | 375 | **3000** | bug 是否修复 |
| SFT 后 GSM8K Pass@1 | 12.5% | **20-25%** | SFT 是否充分 |
| RL 峰值 Pass@1 | 14.0% | **30-40%** | RL 是否有效 |
| 对话完整性 | 截断/错误 | 完整正确回答 | 最终可用性 |
| "10+10=?" 能否正确回答 | ❌ | ✅ | 基本能力 |

---

## 二、v1 vs v2 参数对比

### 2.1 SFT 阶段

| 参数 | v1 | v2 | 改动原因 |
|------|----|----|---------|
| `total-batch-size` | 524,288 | **65,536** | 降低 8x，配合 bug 修复确保跑满步数 |
| `num-iterations` | 3000 (实际 375) | **3000 (预期 3000)** | 修复了 chat_sft.py 中数据耗尽自动停止的 bug |
| `gsm8k-epochs` | 8 (~60K rows) | **16 (~120K rows)** | 数学数据占比翻倍 |
| `mmlu-epochs` | 5 | 5 | 不变 |
| `device-batch-size` | 32 | 32 | 不变 |
| `max-seq-len` | 2048 | 2048 | 不变 |
| `eval-every` | 500 | **200** | 更频繁监控 |
| `chatcore-every` | 200 | **500** | benchmark 评估间隔拉大（耗时长） |

**SFT 数据量估算**:
- v1: 1.3M rows × 1 epoch / 524K batch ≈ 375 步 → 每行只被看了 ~1 次
- v2: 1.36M rows × 多 epoch / 65K batch ≈ 3000 步 → 每行被看了 ~96 次（充分迭代）

**bug 修复** (`scripts/chat_sft.py` 第 298-300 行):
```python
# v1 (bug): 数据用完就停，无视 num-iterations
if consumed >= dataset_size:
    last_step = True

# v2 (修复): 指定了 num-iterations 时，数据用完继续循环
if consumed >= dataset_size and args.num_iterations <= 0:
    last_step = True
```

### 2.2 RL 阶段

| 参数 | v1 | v2 | 改动原因 |
|------|----|----|---------|
| `num-epochs` | 3 | **1** | v1 最优在 step 120 (epoch 1 内)，后续全是过拟合 |
| `num-samples` | 16 | **32** | 更多采样 → 更多有效梯度信号，减少 reward=0 的比例 |
| `max-new-tokens` | 512 | **768** | v1 序列长度 222→129，回答被截断；加长生成上限 |
| `device-batch-size` | 16 | **32** | 配合 num-samples=32 |
| `eval-every` | 60 | **30** | v1 可能错过了 step 90-120 间的真正最优 |
| `save-every` | 60 | **30** | 配合更频繁 eval |
| `init-lr-frac` | 0.05 | **0.02** | 更小初始学习率，减少初期震荡 |
| `temperature` | 1.0 | 1.0 | 不变 |
| `examples-per-step` | 32 | 32 | 不变 |
| `eval-examples` | 400 | 400 | 不变 |

**RL 预计步数**: ~7,500 train examples / 32 per step ≈ 233 步 (v1 是 699 步)

### 2.3 不变的部分（复用 v1）

| 阶段 | 说明 |
|------|------|
| 数据下载 | 100 shards (~10GB), 已完成 |
| Tokenizer | BPE vocab=32768, 已完成 |
| Base 预训练 | d20, 5000 步, BPB 0.732, 已完成 |

---

## 三、预期时间

| 阶段 | 预计耗时 | 说明 |
|------|---------|------|
| 前置检查 | <1min | 确认 checkpoint/数据存在 |
| SFT v2 (3000 步) | **~8-10 小时** | 65K batch, 每步 ~10s (无梯度累积更快) |
| SFT 评估 | ~10 min | quick 模式 |
| RL v2 (~233 步) | **~2-3 小时** | 32 samples/题，每步更重但总步数少 |
| 最终评估 | ~30 min | 全 checkpoint 评估 |
| **总计** | **~11-14 小时** | |

---

## 四、训练日志（待填写）

### 4.1 SFT v2 训练

**开始时间**: （待记录）

**实际步数**: （待记录）— 预期 3000，如果仍然提前停止则需进一步排查

**训练指标**:

| 指标 | Step 0 | Step 1000 | Step 2000 | Step 3000 (最终) |
|------|--------|-----------|-----------|-----------------|
| Val BPB | — | — | — | — |
| train/loss | — | — | — | — |
| tok/sec | — | — | — | — |

**Benchmark 评估 (ChatCORE)**:

| 评估指标 | v1 Step 375 | v2 Step 500 | v2 Step 1000 | v2 Step 2000 | v2 Step 3000 |
|----------|-------------|-------------|--------------|--------------|--------------|
| chatcore | 0.296 | — | — | — | — |
| ARC-Easy | 0.481 | — | — | — | — |
| ARC-Challenge | 0.366 | — | — | — | — |
| MMLU | 0.329 | — | — | — | — |
| GSM8K | 0.125 | — | — | — | — |
| HumanEval | 0.083 | — | — | — | — |
| SpellingBee | 1.000 | — | — | — | — |

### 4.2 SFT v2 对话测试（RL 前）

| 输入 | v1 SFT 输出 | v2 SFT 输出 |
|------|-------------|-------------|
| "2x + 5 = 17" | (未单独测试 SFT) | （待测试） |
| "10 + 10 = ?" | (未单独测试 SFT) | （待测试） |
| GSM8K 风格题 | — | （待测试） |

### 4.3 RL v2 训练

**开始时间**: （待记录）

**Pass@k 评估 (GSM8K test, 400题)**:

| Step | Pass@1 | Pass@4 | Pass@8 | Pass@16 | Pass@32 |
|------|--------|--------|--------|---------|---------|
| 0 (SFT v2 基线) | — | — | — | — | — |
| 30 | — | — | — | — | — |
| 60 | — | — | — | — | — |
| 90 | — | — | — | — | — |
| 120 | — | — | — | — | — |
| 150 | — | — | — | — | — |
| 180 | — | — | — | — | — |
| 210 | — | — | — | — | — |
| 最终 | — | — | — | — | — |

**Train Reward 趋势**:

| 训练区间 | 平均 Reward | 说明 |
|----------|------------|------|
| step 0-60 | — | — |
| step 60-120 | — | — |
| step 120-180 | — | — |
| step 180-233 | — | — |

### 4.4 RL v2 对话测试（最优 checkpoint）

| 输入 | v1 RL 输出 (step 120) | v2 RL 输出 (step ?) |
|------|----------------------|---------------------|
| "2x + 5 = 17" | "...2x = 12" (截断) | （待测试） |
| "10 + 10 = ?" | "10 + 10" (截断) | （待测试） |
| "Solve: 2x + 5 = 17" | "...x = 3" (错误) | （待测试） |
| GSM8K 风格应用题 | — | （待测试） |

---

## 五、结果对比（待填写）

### 5.1 关键指标对比

| 指标 | v1 | v2 | 变化 | 目标达成？ |
|------|----|----|------|----------|
| SFT 实际步数 | 375 | — | — | 目标: 3000 |
| SFT 后 GSM8K | 12.5% | — | — | 目标: 20-25% |
| RL 峰值 Pass@1 | 14.0% (step 120) | — | — | 目标: 30-40% |
| RL 峰值 Pass@16 | 38.0% (step 120) | — | — | 目标: >40% |
| 最优 step 位置 | 17% (120/699) | — | — | 目标: >50% |
| "10+10=?" | ❌ | — | — | 目标: ✅ |
| 对话完整性 | 截断 | — | — | 目标: 完整 |

### 5.2 v2 假设验证

| 假设 | 验证结果 | 说明 |
|------|---------|------|
| SFT 步数不足是根因 | （待验证） | 如果 v2 SFT 3000 步后 GSM8K >20%，则假设成立 |
| RL 过拟合因 SFT 基础弱 | （待验证） | 如果 v2 RL 最优点不再在极早期，则假设成立 |
| 降低 RL epoch 可减少过拟合 | （待验证） | 对比 v1 和 v2 的最优 step 占总步数比例 |
| 增加采样数可提升梯度质量 | （待验证） | 对比 v1 和 v2 的 reward=0 比例 |

---

## 六、分析与结论（训练完成后填写）

### 6.1 SFT v2 分析

（训练完成后根据日志填写）

### 6.2 RL v2 分析

（训练完成后根据日志填写）

### 6.3 v1 → v2 整体总结

（训练完成后填写：哪些假设被验证、哪些意外发现、下一步建议）

---

## 七、执行命令参考

```bash
# 启动 v2 训练（一键）
cd /mnt/openclaw/MathGPT && bash scripts/full_train_a800_v2.sh

# 单独运行 SFT v2
NANOCHAT_BASE_DIR=./runs python3 -m scripts.run scripts.chat_sft \
    --total-batch-size=65536 --num-iterations=3000 --gsm8k-epochs=16 \
    --mmlu-epochs=5 --offline=data/hf_datasets --run=dummy

# 单独运行 RL v2
NANOCHAT_BASE_DIR=./runs python3 -m scripts.train_rl \
    --source=sft --num-epochs=1 --device-batch-size=32 --examples-per-step=32 \
    --num-samples=32 --max-new-tokens=768 --init-lr-frac=0.02 \
    --eval-every=30 --save-every=30 --offline=data/hf_datasets --run=dummy

# SFT 效果测试
NANOCHAT_BASE_DIR=./runs python3 -m scripts.chat_cli --source sft

# RL 效果测试（指定最优 step）
NANOCHAT_BASE_DIR=./runs python3 -m scripts.chat_cli --step <best_step>

# 生成评估报告
NANOCHAT_BASE_DIR=./runs python3 -m scripts.eval_report --all-steps --offline=data/hf_datasets
```

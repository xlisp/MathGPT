# MathGPT 训练日志

## 训练环境

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce GTX 1080 (8 GB VRAM, SM 6.1) |
| 计算精度 | float32 (GTX 1080 不支持 bfloat16) |
| Python | 3.12 |
| PyTorch | 2.3.1 |
| 训练日期 | 2026-03-31 |

## 模型架构

| 参数 | 值 |
|------|----|
| 层数 (n_layer) | 6 |
| 注意力头 (n_head) | 6 |
| KV 头 (n_kv_head) | 6 |
| 嵌入维度 (n_embd) | 384 |
| 头维度 (head_dim) | 64 |
| FFN 比例 | 4x |
| 词表大小 | 32,768 |
| 最大序列长度 | 256 |
| 注意力模式 | Full (L) |
| **总参数量** | **73,531,646 (~73.5M)** |

## 训练流程

### 阶段 1：数据集下载

下载了 8 个分片的 ClimbMix 预训练数据集 (~800 MB)，存储于 `runs/base_data_climbmix/`。

### 阶段 2：Tokenizer 训练

- 算法：BPE（Byte Pair Encoding）
- 词表大小：32,768
- 训练数据：最多 2B 字符（来自 ClimbMix + 身份对话数据）

### 阶段 3：基础模型预训练 (Base)

**超参数：**

| 参数 | 值 |
|------|----|
| 迭代步数 | 3,000 |
| 设备批次大小 | 16 |
| 总批次大小 | 8,192 tokens |
| 梯度累积步数 | 2 |
| embedding_lr | 0.3 |
| matrix_lr | 0.02 |
| weight_decay | 0.28 |
| 预热步数 | 40 |
| warmdown 比例 | 65% |
| 优化器 | MuonAdamW（Muon 用于矩阵参数，AdamW 用于其余） |

**训练时间：** 约 25 分钟（10:47 - 11:12）

**验证集 BPB（bits per byte）曲线：**

| 步数 | Val BPB |
|------|---------|
| 0 | 3.213 |
| 200 | 1.910 |
| 400 | 1.738 |
| 600 | 1.649 |
| 800 | 1.586 |
| 1000 | 1.538 |
| 1200 | 1.496 |
| 1400 | 1.465 |
| 1600 | 1.438 |
| 1800 | 1.418 |
| 2000 | 1.403 |
| 2200 | 1.389 |
| 2400 | 1.377 |
| 2600 | 1.368 |
| 2800 | 1.361 |
| **3000** | **1.356** |

**Checkpoint 保存：** `runs/base_checkpoints/d6/model_003000.pt`

### 阶段 4：SFT 监督微调 (Chat SFT)

**训练数据混合：**
- SmolTalk（HuggingFace, ~460K 通用对话）
- MMLU（3 个 epoch，学术知识 ~100K/epoch）
- GSM8K 数学题（4 个 epoch，~8K/epoch）
- SimpleSpelling（200K 拼写任务）
- SpellingBee（80K 拼写题）
- 身份对话（1K 行，2 个 epoch）

**超参数：**

| 参数 | 值 |
|------|----|
| 迭代步数 | 500（1000 个数据生成器迭代 / grad_accum=2） |
| 设备批次大小 | 16 |
| 总批次大小 | 8,192 tokens |
| 初始 LR 倍率 | 0.8 |
| warmdown 比例 | 50% |
| 加载基础模型优化器状态 | 是 |

**训练时间：** 约 4 分钟（11:46 - 11:50）

**验证集 BPB 曲线：**

| 步数 | Val BPB |
|------|---------|
| 0（初始化） | 1.182 |
| 200 | 1.172 |
| 400 | 1.021 |
| **500（最终）** | **0.961** |

**修复的 Bug：** 当批次中所有目标 token 均被掩码时（对话过长无法塞入 256 token 窗口，导致全填充行），`cross_entropy(ignore_index=-1)` 返回 NaN。通过在数据打包器中截断过长对话来修复。

**Checkpoint 保存：** `runs/chatsft_checkpoints/d6/model_000500.pt`

### 阶段 5：RL 强化学习（GSM8K 数学题）

**算法：** REINFORCE（类 GRPO）
- 奖励：答案正确 = 1.0，答案错误 = 0.0
- 优势：`advantage = reward - mean(reward)` per batch
- 损失：`loss = -logp × advantage`

**超参数：**

| 参数 | 值 |
|------|-----|
| 总步数 | 934（1 epoch GSM8K train） |
| 设备批次大小 | 4 |
| 每步样本数 | 8 个问题 × 2 次采样 = 16 条轨迹 |
| 最大新 token 数 | 256 |
| 温度 | 1.0 |

**Pass@k 评估（GSM8K test set 400 题）：**

| 步数 | Pass@1 | Pass@2 | Pass@3 | Pass@4 |
|------|--------|--------|--------|--------|
| 0 | 0.50% | 1.25% | 2.00% | 2.75% |
| 60 | 0.75% | 1.25% | 2.25% | 3.00% |
| 120 | 1.00% | 2.75% | 3.25% | 4.50% |
| **934（最终）** | **TBD** | **TBD** | **TBD** | **TBD** |

**Checkpoint 保存：** `runs/chatrl_checkpoints/math_d6/`

---

## 遇到的技术问题与修复

### 1. PyTorch 2.3 + Python 3.12 兼容性问题

| 问题 | 修复方案 |
|------|---------|
| `F.rms_norm` 不存在（PyTorch < 2.4） | 在 `nanochat/gpt.py` 中添加基于方差的回退实现 |
| `torch.compile` Dynamo 不支持 Python 3.12 | 在 `nanochat/optim.py` 中检测并替换为恒等装饰器 |
| SDPA `enable_gqa` 参数不存在（PyTorch < 2.5） | 在 `nanochat/flash_attention.py` 中运行时检测，用 `repeat_interleave` 展开 KV 头 |
| 0-D CPU 张量无法 lerp_ 到 CUDA 张量 | 在优化器步进函数中添加 `.to(dev)` |

### 2. GTX 1080 硬件限制

| 限制 | 处理方案 |
|------|---------|
| 不支持 bfloat16 | 使用 `COMPUTE_DTYPE = torch.float32` |
| 不支持 Flash Attention 3 | 使用 PyTorch SDPA（scaled_dot_product_attention） |
| VRAM 8 GB | 调小模型（depth=6, embd=384），降低 device_batch_size=16 |

### 3. SFT NaN 损失问题

**根因：** 当 SmolTalk 中某些对话超过 max_seq_len=256 token 时，BOS 最优填充器（bestfit packer）无法塞入任何对话，创建全填充（mask=0）的行。对全掩码批次调用 `cross_entropy(ignore_index=-1)` 返回 NaN，导致参数变 NaN。

**修复：** 当行为空且无对话可以完整放入时，截断最短对话至可用空间，确保每行至少有部分有效目标。

---

## 模型测试结果

*(RL 训练完成后补充)*

### 数学推理测试

待 RL 完成后测试以下问题...

---

## 结论与展望

本次训练为在 GTX 1080（8 GB VRAM，float32）上完整复现从零预训练 → SFT → RL 的全流程提供了验证。

**局限性：**
- 模型规模小（73.5M 参数）：受 VRAM 限制，无法训练更深的模型
- 上下文窗口短（256 token）：数学推理步骤容易被截断
- SFT 步数少（500 步）：SFT 阶段训练不足

**改进方向：**
- 升级到支持 bfloat16 / Flash Attention 的 GPU
- 增加上下文长度（512 或 1024）
- 增加 SFT 迭代次数（3000+ 步）
- 训练更大的模型（depth=12, embd=768 → ~300M 参数）

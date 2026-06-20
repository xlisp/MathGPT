# LLM 训练技术详解：从 SFT 到 GRPO

本文档介绍 MathGPT 项目中使用的核心训练技术，以及它们在强化学习历史脉络中的位置。

---

## 一、监督微调（SFT）是什么？

### 1.1 基本概念

**监督微调（Supervised Fine-Tuning，SFT）** 是在预训练语言模型基础上，用带标签的对话数据继续训练，让模型学会"好的回答方式"。

预训练阶段的模型（Base Model）只学会了"像人类写作"，可以预测下一个 token，但不知道什么是"有用的回答"。SFT 通过展示高质量的人类对话示例，教会模型：
- 按照提问者的意图作答
- 遵循特定格式（如 `<|user_start|>...<|assistant_start|>...`）
- 在适当场合使用工具（如计算器 `<|python_start|>...<|python_end|>`）
- 对数学问题展示推理步骤

### 1.2 SFT 的训练过程

```
输入序列:
[BOS] <|user_start|> 苹果的颜色是什么？ <|user_end|> <|assistant_start|>

目标序列:
苹果通常是红色、绿色或黄色的。 <|assistant_end|>
```

训练时：
- **掩码（loss mask）**：只计算助手回答部分的损失，用户问题不计入损失
- **损失函数**：交叉熵（Cross-Entropy），衡量模型预测分布与真实 token 的差距
- **优化器**：MuonAdamW（矩阵参数用 Muon，其余用 AdamW）

### 1.3 SFT 训练数据混合

本项目 SFT 使用多种数据混合训练（数值对应 A800 训练 `scripts/full_train_a800_v2.sh` 的实际配置）：

| 数据集 | 规模 | 用途 |
|--------|------|------|
| SmolTalk | 460K | 通用对话能力 |
| MMLU | ~500K（`--mmlu-epochs=5`） | 学科知识 |
| GSM8K | ~120K（`--gsm8k-epochs=16`，数学占比翻倍） | 数学推理 + 工具调用 |
| SimpleSpelling | 200K | 拼写任务 |
| SpellingBee | 80K | 字母计数任务 |
| Identity | ~2K | 模型身份 |

> 关键超参：`--max-seq-len=2048`、`--device-batch-size=32`、`--total-batch-size=65536`、`--num-iterations=3000`。其中 `total-batch-size` 从早期的 524K 降到 65K 是为了让有限的数据跑满 3000 步（详见第六节经验复盘）。

### 1.4 为什么 SFT 之后还需要 RL？

SFT 有几个内在局限：

1. **模仿而非创新**：SFT 模型只会模仿训练数据的模式，对于新问题缺乏泛化能力
2. **无法自我评价**：模型不知道自己的回答是否正确，无法主动改进
3. **奖励信号稀疏**：SFT 对每个 token 平等对待，无法区分"关键的推理步骤"和"填充词"
4. **数学特别难**：数学问题需要精确的计算，而 SFT 训练目标允许小概率的错误 token

这就是 RL 微调（RLFT）的动机所在。

---

## 二、MathGPT 的设计思路（A800 训练配置）

### 2.1 为什么选择这个架构？

本项目基于 [nanochat](https://github.com/karpathy/nanochat) 框架，并在 **NVIDIA A800-SXM4-80GB** 上完成了完整的 Base → SFT → RL 训练。设计原则：

**极简主义**：
- 不使用外部推理框架（无 LangChain、无 vLLM）
- 所有组件自包含，便于理解和修改
- 核心代码（`nanochat/`、`scripts/`）规模精简，便于通读

**完整流程**：
- 从头预训练（不依赖已有权重）
- SFT → RL 完整管线
- 推理引擎支持工具调用

**A800 训练配置**（来自 `scripts/full_train_a800_v2.sh` 与训练报告）：

```
硬件能力 → 设计决策
──────────────────────────────────────────────
A800-80GB, SM 8.0      → bfloat16 计算（2x 效率）
80 GB VRAM             → depth=20, n_embd=1280, head_dim=128（~700M 参数）
SM 8.0 < Hopper(SM 9.0) → 无 Flash Attention 3，训练用 flash-attn 2 / SDPA
上下文长度             → seq_len=2048（早期 GTX 实验仅 256）
预训练数据             → 100 shards ≈ 2.6B tokens（5000 步, BPB 0.732）
```

> 模型定义见 `nanochat/gpt.py`（`GPTConfig`），A800 实际用 depth=20 的配置（base checkpoint 路径 `runs/base_checkpoints/d20/model_005000.pt`）。相比早期在消费级显卡上的 73.5M 小模型实验，A800 把参数量提升到 ~10x、上下文 8x、预训练数据 ~100x，从而把数学推理能力从"全错循环"推进到"复杂题有推理、Pass@1 ~14.75%"。

### 2.2 工具调用设计（Calculator Tool）

模型通过特殊 token 调用 Python REPL 做精确计算（特殊 token 定义见 `nanochat/tokenizer.py` 的 `SPECIAL_TOKENS`）：

```
模型输出：...所以 Janet 每周储蓄 <|python_start|>16*7<|python_end|>
引擎计算：eval("16*7") = 112
强制注入：<|output_start|>112<|output_end|>
模型继续：112 美元...
```

完整的特殊 token 集合：`<|bos|>`、`<|user_start|>/<|user_end|>`、`<|assistant_start|>/<|assistant_end|>`、`<|python_start|>/<|python_end|>`（模型调用工具）、`<|output_start|>/<|output_end|>`（环境回注结果）。

这个设计让模型不依赖自身"记忆"做算术——直接对应训练报告中占比最高（+10~20pp ROI）的改进方向，也为第九节的"思考链中嵌入计算器"打下基础。

### 2.3 KV Cache 推理引擎

推理引擎（`nanochat/engine.py`）采用批量生成 + KV 缓存复用：

1. **一次 prefill**：对提示词做一次前向传播，缓存所有层的 KV
2. **复制 KV Cache**：将 batch=1 的缓存广播到 batch=N，用于并行采样（RL 一题采 N 条轨迹、推理时多数投票都依赖此）
3. **增量 decode**：每步只推理一个新 token，用 `flash_attn_with_kvcache` 原地更新 KV 缓存

---

## 三、强化学习发展历史：从 DQN 到 GRPO

### 3.1 深度 Q 网络（DQN，2013-2015）

**核心思想：** 用深度神经网络近似 Q 函数（状态-动作值函数）

```
Q(s, a) = 即时奖励 r + γ × max_{a'} Q(s', a')
```

DeepMind 的 DQN（2013）首次让智能体在不了解游戏规则的情况下，直接从像素学习打 Atari 游戏。

**关键技术：**
- **经验回放（Experience Replay）**：存储 (s, a, r, s') 元组，打破时序相关性
- **目标网络（Target Network）**：用滞后版本的网络计算目标值，稳定训练

**局限性：** DQN 只能处理离散动作空间（按哪个键），无法直接应用于生成文本（词表大小 32768 = 32768 种"动作"）。

### 3.2 策略梯度方法（Policy Gradient，2016）

**核心思想：** 直接优化策略 π(a|s)，不需要估计 Q 函数

```
∇J(θ) = E_{τ~π_θ}[∑_t ∇log π_θ(a_t|s_t) × G_t]
```

其中 G_t 是从 t 步开始的累积奖励。

**REINFORCE 算法** 是最简单的策略梯度实现：
1. 用当前策略采样轨迹
2. 计算每个动作的累积奖励
3. 梯度上升：增大好动作的概率，减小坏动作的概率

这直接可以应用于语言模型——每个生成的 token 就是一个"动作"！

### 3.3 PPO（近端策略优化，2017）

**问题：** 朴素策略梯度不稳定——步长太大会让策略崩溃

**PPO 解决方案：** 限制每次更新中新旧策略的差异

```
L_PPO = E[min(r_t(θ) × A_t, clip(r_t(θ), 1-ε, 1+ε) × A_t)]

其中 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)（重要性采样比）
```

PPO 是后来 RLHF 的基础算法。

### 3.4 从游戏 AI 到语言模型：RLHF（2022）

**InstructGPT / ChatGPT（OpenAI, 2022）** 将 RL 引入 LLM 训练，开创了 RLHF（Reinforcement Learning from Human Feedback）范式。

**RLHF 三步流程：**

```
Step 1: SFT（监督微调）
  人类示范 → 模型学会基本的回答格式

Step 2: 奖励模型训练（Reward Model）
  人类对比不同回答打分 → 训练一个"打分器"
  输入：(问题 + 回答) → 输出：标量分数

Step 3: PPO 强化学习
  用奖励模型提供奖励信号，用 PPO 优化策略
  同时加 KL 约束防止偏离 SFT 模型太远
```

**RLHF 的核心贡献：**
- 解决了"如何把人类主观偏好变成训练信号"的问题
- 奖励模型充当代理人类判断

**RLHF 的局限：**
- 需要大量人工标注对比数据（成本高）
- 奖励模型可能被"游玩"（reward hacking）
- PPO 实现复杂，需要 4 个模型同时在 GPU 上（policy, ref, reward, value）

### 3.5 GRPO（群体相对策略优化，2024）

**DeepSeek-Math（2024）** 提出 GRPO，专门为数学推理设计，大幅简化 RLHF。

**核心思想：** 用"同组内的相对表现"代替独立的价值函数估计

```
对同一个问题，采样 G 条轨迹：
{o_1, o_2, ..., o_G}，每条得到奖励 {r_1, r_2, ..., r_G}

优势函数：
A_i = (r_i - mean(r)) / std(r)

（把组内平均分当基线，高于平均的是正优势，低于平均的是负优势）
```

**GRPO 损失函数：**

```
L_GRPO = -E[(min(r_t × A, clip(r_t, 1-ε, 1+ε) × A)) - β × KL(π_θ || π_ref)]
```

**与 PPO/RLHF 的关键区别：**

| 对比维度 | PPO/RLHF | GRPO |
|---------|---------|------|
| 价值函数（Critic） | 需要（额外大模型） | 不需要 |
| 奖励模型 | 需要（人工标注训练） | 不需要（规则奖励即可） |
| 基线计算 | 价值函数预测 | 同组均值 |
| 实现复杂度 | 高（4 个模型） | 低（2 个模型） |
| 适用场景 | 通用对话对齐 | 有明确正确答案的任务 |

**为什么 GRPO 适合数学：** 数学题有明确的对错标准（最终答案是否正确），不需要人工打分，奖励函数是确定性的（正确=1，错误=0）。

### 3.6 MathGPT 使用的 RL 算法

本项目使用的是 **简化版 REINFORCE（接近 GRPO）**：

```python
# 对同一个问题采样 num_samples 条轨迹
rewards = [1.0 if correct else 0.0 for each sample]

# REINFORCE with baseline（组内均值作为基线）
advantage = reward - mean(rewards)  # 组内相对优势

# 策略梯度损失
loss = -logprob × advantage

# 只对采样 token（非强制注入的 token）计算梯度
```

**与完整 GRPO 的区别：**
- 没有 KL 散度正则项（简化实现）
- 没有 PPO-style clip（直接用 REINFORCE）
- 奖励：基于规则（答案数字是否匹配）

---

## 四、RLHF 和 GRPO 的关系图谱

```
强化学习基础
├── DQN（2013）：Q-learning + 深度网络，值函数近似
│   └── 解决离散动作空间问题（Atari 游戏）
│
├── A3C / A2C（2016）：Actor-Critic，同时学策略和值函数
│
├── PPO（2017）：近端策略优化，稳定的策略梯度
│   └── 成为 RLHF 的核心算法
│
└── REINFORCE（经典，Williams 1992）：
    朴素策略梯度，简单但方差大

应用于 LLM
├── RLHF（2022, InstructGPT）
│   ├── Step1: SFT 基础对话
│   ├── Step2: 奖励模型（人工标注对比）
│   └── Step3: PPO 强化学习（需要 Critic 网络）
│
├── DPO（2023, Direct Preference Optimization）
│   └── 绕过显式奖励模型，直接优化偏好
│
├── GRPO（2024, DeepSeek-Math）
│   ├── 无需奖励模型，规则奖励
│   ├── 无需 Critic 网络，用组内均值做基线
│   └── 专为有明确答案的任务设计
│
└── MathGPT（本项目）
    └── 简化版 REINFORCE ≈ GRPO（无 KL 项，无 clip）
```

---

## 五、为什么 GRPO 让小模型也能学数学？

一个有趣的现象是：GRPO 在推动数学能力方面比 SFT 更有效，即使模型参数量很小。

原因分析：

1. **探索-利用权衡**：SFT 只告诉模型"正确答案是什么"，但不告诉它为什么那条推理路径更好。GRPO 让模型自己探索不同的解题路径，只有得到正确答案的路径才被强化。

2. **正确的归因**：如果模型的推理过程有 5 个步骤，SFT 对每个步骤平等训练。GRPO 则只对"最终导致正确答案"的序列进行正向强化——即使模型偶然答对，那些 token 的概率也会被提高。

3. **批判性思维**：用不同采样得到的回答相互比较，模型能学到"在这类问题上，用哪种推理策略成功率更高"。

4. **计算器协同**：本项目的工具调用机制与 GRPO 天然契合——如果模型学会在关键计算步骤使用计算器（`<|python_start|>...<|python_end|>`），则最终答案的正确率会大幅提升，从而获得更多奖励信号。

---

## 六、从训练日志看实战：v1 → v2 复盘与大模型训练经验

> 本节基于两份真实训练报告（`reports/A800_TRAINING_REPORT_v1.md`、`reports/A800_TRAINING_REPORT_v2.md`）和 `runs/tb_logs/` 中的 TensorBoard 日志整理。**理论会告诉你算法长什么样，但只有日志会告诉你算法在你的数据和硬件上真正发生了什么。** 这一节是本文档最有价值的部分——它是用 GPU 时间换来的经验。

### 6.1 训练管线全景（A800-80GB, bf16, ~700M 参数）

```
数据下载(100 shards, ~2.6B tokens)
   → Tokenizer(BPE vocab=32768)
   → Base 预训练(5000 步, BPB 3.138→0.732, ~14.9h)
   → SFT(目标 3000 步, BPB→0.354)
   → RL(REINFORCE≈GRPO, Pass@1 峰值 ~14.75%)
   → Chat 推理(带计算器工具)
```

### 6.2 核心经验一：SFT 是 RL 的地基，不是配角

v1 训练最大的教训是一个**沉默的 bug**：`chat_sft.py` 在数据跑完 1 个 epoch 后自动停止，无视 `--num-iterations=3000`，导致 SFT 实际只跑了 **375 步**（计划的 12.5%）。

```python
# v1 (bug): 数据用完就停，无视 num-iterations
if consumed >= dataset_size:
    last_step = True

# v2 (修复): 指定了 num-iterations 时，数据用完继续循环
if consumed >= dataset_size and args.num_iterations <= 0:
    last_step = True
```

修复后 v2 跑满 2999 步，结果是**链式反应级的提升**：

| 指标 | v1 (SFT 375步) | v2 (SFT 2999步) | 倍数 |
|------|---------------|-----------------|------|
| SFT Val BPB | 0.329 | 0.354* | — |
| **RL 初始 reward** | **0.07** | **0.41** | **5.9x** |
| RL 峰值 Pass@1 | 14.0% | 14.75% | +0.75pp |
| RL 峰值 Pass@16 | 38.0% | 43.0% | +5.0pp |

> *v2 的 BPB 数值口径与 v1 略有不同（batch size 不同），不可直接横比，关键证据是 RL 初始 reward 的 5.9x 提升。

**为什么 RL 初始 reward 是最关键的信号？** RL（REINFORCE/GRPO）的学习信号来自"采样出的多条轨迹之间有好有坏"。如果模型太弱、所有采样全错（reward 全为 0），那么 `advantage = reward - mean(reward) = 0`，**梯度为零，RL 学不到任何东西**。SFT 把起点从 reward=0.07 抬到 0.41，意味着有大量"部分答对"的题目能提供有效梯度——**这就是"SFT 是地基"的量化含义**：RL 不能凭空创造能力，它只能放大 SFT 已经埋下的能力种子。

### 6.3 核心经验二：batch size 改变的是"步数组织方式"，不是总算力

v1 预估 SFT 要跑 8-10 小时，实际 70 分钟就跑完 3000 步。原因不是"变快了"，而是 batch size 从 524K 降到 65K，梯度累积从 8 次降到 1 次：

```
v1:  375 步 × 8 次梯度累积 = 3,000 次 forward/backward
v2: 2999 步 × 1 次梯度累积 = 2,999 次 forward/backward
   ⇒ 总计算量几乎相同！
```

**经验**：小 batch + 多步数 让每条数据被"看"更多遍（更高 epoch），模型更充分地学习对话格式和推理链。在数据量有限时，**降 batch size 是"免费"地增加有效迭代次数**——总算力不变，但优化轨迹更精细。

### 6.4 核心经验三：train reward 上升 ≠ 模型变好（RL 过拟合）

这是整个项目最重要的反直觉发现。两版 RL 都出现了同一模式：

```
train reward: 0.07 → 0.45 (持续上升，看起来很美好)
eval Pass@1:  step 120/90 达峰值后停滞甚至下降
eval Pass@16: 38%→32.8% (v1) / 43%→31.5% (v2)  ← 多样性崩溃
seq_len:      222 → 129 tokens (回答越来越短)
```

模型在训练后期学到的不是"推理能力"，而是**作弊捷径**：
1. 生成更短的序列、提前吐出 `<|assistant_end|>`（短回答更容易凑出 `#### number` 格式拿 reward）
2. 记忆训练集特定题目
3. 在见过的简单 pattern 上过拟合

**Pass@16 下降是过拟合的早期预警**：好的 RL 应该 Pass@1 升的同时 Pass@16 不降。一旦 Pass@16 开始掉，说明模型在丢失探索多样性（mode collapse），应当立即停止。

### 6.5 核心经验四：最优 checkpoint 几乎从不在最后一步

| 版本 | 最优步 | 占总训练 | 现象 |
|------|--------|---------|------|
| v1 | step 120 | 17% | 后 83% 步数纯属过拟合浪费 |
| v2 | step 90 (Pass@16) / step 480 (Pass@1) | — | U 型轨迹，epoch 边界崩溃后又恢复 |

v2 还观察到一个有趣的 **U 型轨迹**：epoch 1 末（step 120-232）性能崩溃到谷底（Pass@1 4.75%），epoch 3（step 240+）又反弹到 14.75%。最可能的原因是 **epoch 边界的数据重新洗牌造成分布突变**。这进一步印证：**多 epoch RL 有害，应该 `--num-epochs=1` 并在 Pass@16 峰值处早停。**

> ⚠️ v2 的一个真实事故：脚本误用了 v1 的 3-epoch 配置，导致 RL 跑了 698 步（15.6h）而非预期的 233 步（2-3h），白白浪费 5x 时间。**经验：配置文件比代码更危险，一个参数能烧掉一天 GPU。**

### 6.6 核心经验五：评估口径必须统一，否则数字会"互相打架"

v2 报告里出现过看似矛盾的数字：同一个 SFT checkpoint，GSM8K 一会儿是 16.67%，一会儿是 4.0%。真相是**温度不同**：

| 评估方式 | 脚本 | Temperature | 测的是什么 | GSM8K |
|----------|------|-------------|-----------|-------|
| ChatCORE | `chat_eval.py` | **0.0 (贪心)** | 最优路径单次准确率 | **16.67%** |
| Pass@1 | `eval_report.py` | **1.0 (随机)** | 随机采样一次的命中率 | **4.0%** |

**经验**：
- 贪心（t=0）测"模型能力上限"，随机采样（t=1.0）测"探索多样性"。
- RL 训练前看 t=1.0 的 Pass@k 基线（衡量探索空间），部署后看 t=0 的准确率。
- **报告里每个准确率数字都必须标注 temperature 和题目数量**，否则无法横比，甚至会误导决策。

### 6.7 核心经验六：边际收益递减与能力天花板

| 投入变化 | Pass@1 收益 |
|---------|------------|
| SFT 375 → 2999 步 (8x) | +0.75pp (+5.4%) |
| RL 120 → 480 步 (4x) | 几乎持平 |

v2 的 ROI 测算是 **0.15pp / $1 GPU时**，严重的边际收益递减。当前 ~700M 模型在 GSM8K 上的 Pass@1 天花板就是 **~14-15%**，**靠加训练步数撞不破**。EVAL 报告显示 **68.5% 的题目（274/400）即使采样 16 次也全错**，失败模式分布：

| 失败模式 | 占比 | 根因 | 可改进性 |
|---------|------|------|----------|
| 比例/分数理解错误 | 33% | 概念缺失 | ⚠️ 难（需更多数据） |
| 多步推理截断 | 28% | max_tokens 不足 | ✅ 易（调到 1024+） |
| 时间/单位错误 | 18% | 领域知识缺失 | ⚠️ 中 |
| 多变量跟踪失败 | 21% | 工作记忆/容量不足 | ⚠️ 难 |

**经验**：资源要投在瓶颈处。当前瓶颈是**模型容量和推理长度**，不是训练步数。这直接引出了第八、九节的改进方向。

---

## 七、读懂 runs 日志：每个字段都在告诉你什么

训练日志在 `runs/tb_logs/{chat_sft,train_rl}/default/` 下（TensorBoard 格式），可用 `tensorboard --logdir runs/tb_logs` 查看。但更重要的是看懂控制台日志。

### 7.1 Base / SFT 日志

```
step 04999/05000 (99.98%) | loss: 2.501108 | lrm: 0.05 | dt: 10740ms | tok/sec: 48,815 | bf16_mfu: 45.16 | total time: 893.34m
```

| 字段 | 含义 | 怎么判断好坏 |
|------|------|-------------|
| `loss` | 交叉熵 (nats/token) | **越低越好**。理论下限 0，随机=ln(32768)=10.4 |
| `lrm` | 学习率乘子 | cosine 衰减进度，尾部接近 0 |
| `dt` | 每步耗时 | 含全部梯度累积的前后向 |
| `tok/sec` | 吞吐量 | 越高越好，反映数据/计算流水线效率 |
| `bf16_mfu` | 算力利用率 | A800 单卡 45% 已属优秀（无 FA3，用 SDPA） |

**关键认知**：loss=2.50 看起来高，其实很好。它对应 perplexity=e^2.5≈12.2（模型平均在 12 个候选里犹豫），对 700M 模型完全正常。**BPB=0.732 才是真正的质量指标**（每字节只需 0.73 bit，压缩优于 1 bit/byte）。loss "接近 1" 是不同尺度的误解——那需要数十 B 参数 + 数 T tokens。

SFT 的 loss（~0.98）比 Base（2.50）低，**不是因为 SFT 更强**，而是 SFT 只对 assistant 回复部分算 loss（user/prompt 用 `ignore_index=-1` 掩码），预测回答天然更容易，两者不可直接比。

### 7.2 RL 日志（重点，最容易误读）

```
Step 485/699 | Ex 0  | Pass 0 | loss: 0.000139  | reward: 0.562   ← 理想：部分对部分错
Step 485/699 | Ex 2  | Pass 0 | loss: -0.000000 | reward: 0.000   ← 警报：16采样全错，无梯度
Step 485/699 | reward: 0.2773 | seq_len: 131.4                    ← 整步汇总
```

| 字段 | 含义 |
|------|------|
| `reward` (单 Ex) | 该题 N 个采样的平均（1=答对, 0=答错） |
| `loss` | 该 example 的策略梯度损失 |
| `reward` (汇总) | 该步全部 example 的平均 reward |
| `seq_len` | 生成序列平均长度 |

**最重要的一条：RL 的 loss 不是越低越好！**

```python
advantages = rewards - rewards.mean()       # REINFORCE baseline (train_rl.py:158)
logp   = -model(inputs, targets, loss_reduction='none')
pg_obj = (logp * advantages).sum()
loss   = -pg_obj / num_valid                # 取负号：最小化 loss = 最大化好回答概率
```

| 日志现象 | 含义 | 是否正常 |
|---------|------|---------|
| reward=0, loss=0 | 16 采样全错 → advantage 全 0 → **无梯度** | ⚠️ 正常但不理想（题太难） |
| reward=1, loss=0 | 16 采样全对 → advantage 全 0 → 无梯度 | ✅ 正常（题已学会） |
| reward=0.56, loss≠0 | 部分对部分错 → 有正有负的 advantage | ✅ **最理想，学习最有效** |

当日志里**大量 example 出现 reward=0/loss=0**，说明模型对多数题"够不着"——这正是 Pass@1 卡在平台期的微观原因：容易题已学会（reward→1 无梯度），难题完全不会（reward→0 无梯度），只有"刚好够得着"的题在贡献学习，而这类题越来越少。

---

## 八、当前模型的改进路线（按 ROI 排序）

综合两份报告的失败模式分析，改进优先级如下：

| 策略 | 预期 Pass@1 提升 | 成本 | 推荐度 |
|------|----------------|------|--------|
| **集成外部工具（计算器/Python REPL）** | +10~20pp | 低（工程） | ⭐⭐⭐⭐⭐ |
| **引入 CoT 数据 + 长推理 RL**（见第九节） | +5~15pp | 中 | ⭐⭐⭐⭐⭐ |
| 增大 `max_new_tokens` 768→1024 | 解决 28% 截断题 | 极低 | ⭐⭐⭐⭐ |
| 加 GSM8K/MetaMathQA/NuminaMath 数据重做 SFT | +2~5pp | 低 | ⭐⭐⭐⭐ |
| `--num-epochs=1` + Pass@16 早停 | 防过拟合、省 5x 时间 | 零 | ⭐⭐⭐⭐ |
| RL 加 KL 惩罚（防 mode collapse） | 保住 Pass@16 | 低（改代码） | ⭐⭐⭐ |
| 增大模型 700M→1.5B+ | +5~8pp | 5x GPU | ⭐⭐⭐ |

立即可做（改一行）：
```bash
# RL：单 epoch + 更长生成 + 更密集 eval 以便早停
NANOCHAT_BASE_DIR=./runs python3 -m scripts.train_rl \
    --source=sft --num-epochs=1 --num-samples=32 \
    --max-new-tokens=1024 --init-lr-frac=0.02 \
    --eval-every=30 --save-every=30 --offline=data/hf_datasets
```

**加 KL 惩罚的代码草图**（防止 Pass@16 崩溃，需保留一份 SFT 作为 reference）：
```python
kl_coeff = 0.01
with torch.no_grad():
    ref_logp = -ref_model(inputs, targets, loss_reduction='none')
kl_penalty = (logp - ref_logp).mean()
loss = -pg_obj / num_valid + kl_coeff * kl_penalty   # 把策略锚在 SFT 附近
```

---

## 九、引入思维链（CoT）：通往 GPT-o1 的设计与原理 🚧 规划中

> **⚠️ 本节为路线图（Roadmap），尚未实现。** 以下的理论基石（9.2）、专用 token、两阶段配方、奖励塑形、思考预算等，都是本项目**下一阶段的实现计划**，目前代码（`scripts/train_rl.py` 等）仍是"短回答 + 计算器工具"的形态。本节描述的是"准备怎么做、为什么这么做"，欢迎参与共建或[打赏支持](#十打赏支持-mathgpt-训练)这部分训练。

这是本项目下一阶段最有价值的方向。报告显示 **28% 的失败是"推理截断"、33% 是"概念/多步推理错误"**——这两类问题正是 CoT 和"长推理 + 强化学习"要解决的。

### 9.1 o1 范式的核心原理：把"思考"变成可被 RL 优化的对象

GPT-o1 / DeepSeek-R1 的突破，可以用一句话概括：

> **不再只优化"答案"，而是优化"产生答案的思考过程"；并且让模型在推理时（test-time）花更多 token 思考，就能换来更高的准确率。**

两个关键转变：

1. **训练时（train-time）scaling**：用 RL 在大量"有标准答案"的题目上训练。模型自由生成一长段思考（CoT），只要最终答案对就给正奖励。模型会**自发学到**反思、验证、回溯、换思路等行为——DeepSeek-R1 称之为 "aha moment"。这正是 GRPO 的强项：规则奖励 + 组内相对优势，不需要人工标注每一步。

2. **推理时（test-time）scaling**：思考链越长，准确率越高。o1 允许模型"想很久"再回答。这与传统 LLM "一次前向就出答案"根本不同。

**MathGPT 已经具备实现这个范式的全部基础设施**：GRPO 式 RL（`train_rl.py`）、规则奖励（GSM8K 答案匹配）、特殊 token 机制（`<|python_start|>` 工具调用）、loss 掩码。要做的是把这套机制从"短回答"升级到"长思考链"。

### 9.2 理论基石：CoT 让 Transformer 趋近图灵完备

为什么"想得越久越强"不只是工程经验，而是有数学保证的？李志远、Hong Liu、Denny Zhou、马腾宇的工作 *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems*（ICLR 2024, arXiv:2402.12875）给出了严格证明。

**核心结论（按计算复杂度类表述）**：

| 模型设置 | 表达能力上限 | 含义 |
|---------|------------|------|
| 固定深度、常数精度 Transformer，**无 CoT** | **AC⁰** | 只能解可高度并行的浅层问题 |
| 固定深度、对数精度 Transformer，无 CoT | **TC⁰** | 仍受限，无法做本质串行的计算 |
| 固定深度、常数精度 Transformer，**+ 长度 O(T) 的 CoT** | **可模拟任意 T 大小的布尔电路（→ P/poly）** | 表达能力质变 |

**直觉**：没有 CoT，Transformer 的"计算深度"被层数死死锁住，只能干并行的活（AC⁰）；一旦允许它把中间结果一个个写进思维链，**每个 CoT token 就相当于多算了一步串行计算**，于是浅层网络也能"展开"成任意深的电路。

**证明思路**（很优雅，也直接启发了工程设计）：把一个有 T 个门的布尔电路编码成输入序列（每个门用 4 个 token 描述：门类型、两个输入门索引、本门索引）。构造一个常数深度、嵌入维度仅 O(log n) 的 Transformer，让它逐步生成 4T 个 CoT token，**每生成 4 个 token 就模拟一个门**：用注意力去思维链里读取两个输入门已算好的值，用前馈网络按门类型算出本门输出，再写回思维链供后续门读取。把电路"展开"成长度 O(T) 的思维链后，最后一个门的输出就是答案。

实验在四个任务上验证了理论：模加（可并行）、置换群复合、迭代平方、电路值问题（P-完全）。**最震撼的对比**：置换群复合和迭代平方这类"本质串行"的任务，不用 CoT 时即使 16 层 Transformer 也学不会（准确率 ~20%），**用 CoT 后 1 层 Transformer 就能 100% 解决**。

**这对 MathGPT 意味着什么**：
1. **"用计算换容量"有了理论背书**——我们的 ~700M 小模型受层数限制（对应报告里 21% 的多变量跟踪失败、深层串行推理失败），但**给足 CoT token，浅模型也能模拟深层串行计算**。这正是突破 14-15% Pass@1 天花板的根本路径之一。
2. **CoT 长度本身就是一种"算力旋钮"**——第 9.6 节的"思考预算"不是临时 trick，而是在调节模型可达的复杂度类。
3. **现实约束依然存在**：定理假设权重被正确设置，而真实训练（SFT+RL）能否逼近这组权重是另一回事；且有限上下文窗口和算力成本是硬边界。所以理论说明"上限很高"，工程（第 9.3~9.7 节）决定"能逼近多少"。

> 一句话：**没有 CoT，Transformer 困在 AC⁰；有了 CoT，它走向 P/poly，逼近图灵机。** o1 / R1 的"思考越久越准"在这里得到了复杂度理论的解释——Compute（以 CoT token 的形式）确实能换正确率。

### 9.3 设计一：为"思考"引入专用 token

复用现有的特殊 token 设计思路，新增一对思考标记：

```
<|user_start|> 解方程 2x + 5 = 17 <|user_end|>
<|assistant_start|>
<|think_start|>
我要解 2x + 5 = 17。
先两边减 5：2x = 17 - 5 = 12。
等等，让我验证：如果 2x=12，那 x=6，代回 2*6+5=17 ✓。
两边除以 2：x = 6。
<|think_end|>
所以 x = 6。
<|assistant_end|>
```

**关键设计点**：
- `<|think_start|>...<|think_end|>` 内是"草稿纸"，可以又长又乱、允许自我纠错；`<|think_end|>` 之后是给用户的简洁答案。
- 推理时可以选择**对用户隐藏 think 段**（像 o1 那样只显示最终答案），但 think 段全程参与生成。
- 思考段内可以**嵌套工具调用**：`<|think_start|> ... <|python_start|>12/2<|python_end|><|output_start|>6<|output_end|> ... <|think_end|>`。这把"长推理"和已有的"计算器精确计算"结合——直接打掉报告里"计算错误"和"截断"两类失败。

### 9.4 设计二：两阶段训练配方（SFT 冷启动 + RL 自我进化）

这是 DeepSeek-R1 验证过、最适合本项目的路线：

```
阶段 A：CoT SFT 冷启动 (cold start)
  数据：NuminaMath-CoT(860K) / MetaMathQA / 蒸馏自更强模型的思考链
  目标：让模型先学会"在 <|think_start|> 里写出多步推理"的格式和基本节奏
  作用：避免直接 RL 时模型不会生成长链（对应 6.2 节"SFT 是地基"）

阶段 B：长 CoT 上的 GRPO/RLVR
  采样：每题生成 G 条带 think 段的完整轨迹
  奖励：只看最终答案对不对（规则奖励，沿用现有 reward 函数）
  优势：A_i = (r_i - mean(r)) / std(r)，组内相对
  涌现：正确的长思考被强化 → 模型自发学会反思/验证/回溯
```

**与本项目现有 RL 的差别仅在两处**，工程量不大：
1. `--max-new-tokens` 必须大幅调高（如 2048~4096），给思考链留空间。
2. loss 掩码要正确：**think 段和最终答案的 token 参与策略梯度**，但**工具强制注入的 `<|output_start|>...<|output_end|>` 不参与**（那是环境给的，不是模型的决策——这一点现有 `train_rl.py` 已对"采样 token vs 强制注入 token"做了区分，直接复用）。

### 9.5 设计三：奖励塑形（Outcome vs Process Reward）

| 奖励类型 | 怎么给 | 优缺点 | 本项目建议 |
|---------|--------|--------|-----------|
| **结果奖励 ORM** | 只看最终答案对错（现有做法） | 简单、无需标注、不易被 hack | ✅ 主力，先用这个 |
| **格式奖励** | think 段存在且闭合 +0.1，调用了计算器 +0.1 | 引导格式，低成本 | ✅ 配合 ORM |
| **过程奖励 PRM** | 给每个中间推理步骤打分 | 信号密集、但需训练步骤打分器，易被 reward hacking | ⚠️ 后期再考虑 |

DeepSeek-R1 的经验是：**纯结果奖励 + 简单格式奖励就足以涌现出强推理**，过程奖励（PRM）反而容易被模型"钻空子"。所以 MathGPT 应当先走 "ORM + 格式奖励" 这条最稳的路。

一个组合奖励的草图：
```python
def reward(conversation, generated_text):
    r = 0.0
    if answer_correct(generated_text):        r += 1.0    # 结果奖励（主信号）
    if has_balanced_think_tags(generated_text): r += 0.1  # 鼓励显式思考
    if used_calculator(generated_text):       r += 0.1    # 鼓励精确计算
    return r
```

### 9.6 设计四：推理时的"思考预算"（test-time compute）

o1 的另一半是推理时扩展。在 `engine.py` 推理引擎层面可加入：
- **思考预算**：允许 think 段最多生成 K 个 token，超过则强制写 `<|think_end|>` 收尾给答案——避免无限思考。
- **自洽采样 (self-consistency)**：对同一题采样多条思考链，对最终答案做多数投票。报告里 Pass@16=43% 但 Pass@1 只有 14.75%，说明**正确答案常常"采样得到但没被选中"**——多数投票能把这部分免费捞回来，是性价比极高的推理时技巧。

### 9.7 为什么这条路对小模型尤其有效

回到第五节"GRPO 让小模型学数学"的逻辑，CoT + 长推理 RL 把它推到极致：
1. **把隐式推理外化为 token**：小模型"脑内"工作记忆有限（对应报告里 21% 的多变量跟踪失败），写在草稿纸上就把记忆负担转移到了上下文里。
2. **用计算换容量**：模型不够大，就让它想更久、验证更多遍——test-time compute 部分补偿了参数量的不足。
3. **工具协同**：think 段里嵌入计算器，直接消灭"计算错误"和"截断"——这正是报告中 ROI 最高（+10~20pp）的改进。

**一句话总结设计哲学**：MathGPT 已经有了 o1 范式所需的全部零件（GRPO、规则奖励、特殊 token、工具调用、loss 掩码），引入 CoT 不是推倒重来，而是**把这些零件从"短回答"重新组装成"长思考 + 自我验证 + 工具调用"**。

---

## 参考文献

- Mnih et al. (2013). *Playing Atari with Deep Reinforcement Learning*. DQN.
- Williams (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*. REINFORCE.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. PPO.
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback*. InstructGPT/RLHF.
- Rafailov et al. (2023). *Direct Preference Optimization*. DPO.
- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. GRPO.
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. CoT.
- Wang et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning*. 自洽采样/多数投票.
- Lightman et al. (2023). *Let's Verify Step by Step*. 过程奖励模型 (PRM).
- Li, Liu, Zhou, Ma (2024). *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems*. ICLR 2024, [arXiv:2402.12875](https://arxiv.org/abs/2402.12875). CoT 的复杂度理论基石（AC⁰ → P/poly）.
- OpenAI (2024). *Learning to Reason with LLMs*. o1，train-time / test-time scaling.
- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. 纯 RL 涌现长推理（RLVR / "aha moment"）.
- Karpathy (2024). [nanochat](https://github.com/karpathy/nanochat). 本项目基础框架.
- 训练实录: `reports/A800_TRAINING_REPORT_v1.md`、`reports/A800_TRAINING_REPORT_v2.md`、`reports/EVAL_REPORT_RL_V2.md`、`runs/tb_logs/`.

---

## 十、打赏支持 MathGPT 训练

☕ MathGPT 是一个从零开始、完全开源的数学推理大模型训练项目——从 Base 预训练、SFT、GRPO 强化学习，到下一阶段的 **CoT / o1 范式（第九节路线图）**，每一步都需要真金白银的 GPU 算力（A800-80GB，一轮完整训练约 ~20 小时）。

如果这份文档或这个项目对你有帮助，**欢迎打赏支持我的训练**，你的每一份支持都会直接变成 GPU 时数，推动下面这些计划落地：

- 🚧 实现第九节的 **CoT 长推理 + 长 CoT GRPO**（通往 GPT-o1）
- 🔧 集成外部工具（计算器 / Python REPL，预期 +10~20pp）
- 📈 更大模型 / 更多数学数据，冲击更高 Pass@1

> 打赏方式：微信 / 支付宝（二维码见仓库 `README.md`，或通过邮箱联系）。感谢每一位支持者，你们的名字将出现在 MathGPT 的致谢名单里 🙏

**也欢迎以代码、数据、算力的形式参与共建**——提 Issue / PR，一起把这个小模型推到 o1 的方向。

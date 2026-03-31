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

本项目 SFT 使用多种数据混合训练：

| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| SmolTalk | 460K | 通用对话能力 |
| MMLU | 300K（3 epoch） | 学科知识 |
| GSM8K | 32K（4 epoch） | 数学推理 + 工具调用 |
| SimpleSpelling | 200K | 拼写任务 |
| SpellingBee | 80K | 字母计数任务 |

### 1.4 为什么 SFT 之后还需要 RL？

SFT 有几个内在局限：

1. **模仿而非创新**：SFT 模型只会模仿训练数据的模式，对于新问题缺乏泛化能力
2. **无法自我评价**：模型不知道自己的回答是否正确，无法主动改进
3. **奖励信号稀疏**：SFT 对每个 token 平等对待，无法区分"关键的推理步骤"和"填充词"
4. **数学特别难**：数学问题需要精确的计算，而 SFT 训练目标允许小概率的错误 token

这就是 RL 微调（RLFT）的动机所在。

---

## 二、MathGPT 的设计思路

### 2.1 为什么选择这个架构？

本项目基于 [nanochat](https://github.com/karpathy/nanochat) 框架，进行了 GTX 1080 适配。设计原则：

**极简主义**：
- 不使用外部推理框架（无 LangChain、无 vLLM）
- 所有组件自包含，便于理解和修改
- 代码总量 < 3000 行

**完整流程**：
- 从头预训练（不依赖已有权重）
- SFT → RL 完整管线
- 推理引擎支持工具调用

**GTX 1080 适配**：

```
硬件限制 → 设计决策
──────────────────────────────────────────────
SM 6.1，无 bfloat16 → float32 计算
VRAM 8 GB → depth=6, embd=384（73.5M 参数）
无 Flash Attention → PyTorch SDPA
无 torch.compile → 运行时检测，替换为恒等装饰器
```

### 2.2 工具调用设计（Calculator Tool）

模型通过特殊 token 调用计算器：

```
模型输出：...所以 Janet 每周储蓄 <|python_start|>16*7<|python_end|>
引擎计算：eval("16*7") = 112
强制注入：<|output_start|>112<|output_end|>
模型继续：112 美元...
```

这个设计让小模型（73.5M）也能精确计算数值，不依赖模型本身的"记忆"来做算术。

### 2.3 KV Cache 推理引擎

推理引擎（`engine.py`）采用批量生成 + KV 缓存复用：

1. **一次 prefill**：对提示词做一次前向传播，缓存所有层的 KV
2. **复制 KV Cache**：将 batch=1 的缓存广播到 batch=N，用于并行采样
3. **增量 decode**：每步只推理一个新 token，KV 缓存在 FA3 中原地更新

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

## 参考文献

- Mnih et al. (2013). *Playing Atari with Deep Reinforcement Learning*. DQN.
- Williams (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*. REINFORCE.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. PPO.
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback*. InstructGPT/RLHF.
- Rafailov et al. (2023). *Direct Preference Optimization*. DPO.
- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. GRPO.
- Karpathy (2024). [nanochat](https://github.com/karpathy/nanochat). 本项目基础框架.

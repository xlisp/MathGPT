# MathGPT 项目小白入门指南：从零理解 LLM 预训练、SFT 与强化学习

> 本文面向没有 LLM 训练背景的读者，逐步解释 MathGPT 这个项目"是什么、为什么、怎么做"，并把代码结构和每个训练阶段的原理讲透。
>
> 配套文档：`docs/RL_SFT_GRPO_INTRO.md`（SFT/GRPO 历史背景）、`reports/A800_TRAINING_REPORT_v2.md`（真实训练数据）。

---

## 一、这个项目在做什么？

一句话：**从零开始训练一个会做小学数学应用题的小型 ChatGPT。**

它完整复刻了大模型公司（OpenAI、Anthropic、DeepSeek）训练对话模型的三段式流水线，只是规模缩小了几个数量级，使其能在一张消费级显卡（GTX 1080）或一张 A800 上跑完：

```
原始网页文本                对话数据                  GSM8K 数学题
     │                        │                          │
     ▼                        ▼                          ▼
┌──────────┐  学会"说话"  ┌──────────┐  学会"聊天"  ┌──────────┐  学会"做对题"
│ 预训练    │ ──────────▶ │ SFT 微调  │ ──────────▶ │ RL 强化学习│
│ Base 模型 │             │ Chat 模型 │             │ Math 模型 │
└──────────┘             └──────────┘             └──────────┘
```

这和 ChatGPT/DeepSeek-R1 的训练路线是同构的，区别只在于：模型小（千万~亿级参数 vs 千亿级）、数据少、RL 算法用了最简化的 REINFORCE 而非完整的 PPO/GRPO。

核心框架来自 Karpathy 的 **nanochat**，本项目把它内嵌进来并针对老显卡做了兼容性修复（详见 README 的"兼容性说明"）。

---

## 二、目录结构导览（每个文件是干嘛的）

```
MathGPT/
├── nanochat/        ← "引擎室"：模型、优化器、推理、分词器
├── tasks/           ← "教材库"：训练和考试用的数据集
├── scripts/         ← "操作台"：训练、评估、聊天的入口脚本
├── math_gpt/        ← Web 聊天界面资源
├── docs/            ← 文档
├── reports/         ← 历次训练报告（非常值得读！）
└── runs/            ← 训练产物：分词器、各阶段 checkpoint
```

### 2.1 nanochat/ —— 模型核心

| 文件 | 作用 | 小白解释 |
|------|------|---------|
| `gpt.py` | GPT 模型定义 | Transformer 本体。`GPTConfig` 定义模型形状：`n_layer`（层数）、`n_embd`（隐藏维度）、`n_head`/`n_kv_head`（注意力头，支持 GQA 即多个 query 头共享 KV 头省显存）、`window_pattern`（滑动窗口注意力，"SSSL" 表示三层短窗口+一层全窗口，省计算）。还包含 RMSNorm、RoPE 位置编码等现代 LLM 标配组件。 |
| `optim.py` | MuonAdamW 优化器 | 混合优化器：矩阵参数用 **Muon**（一种对 2D 权重做正交化更新的新优化器，nanochat/模型竞速圈流行），embedding/输出层用 AdamW。注意训练脚本里三组学习率（`embedding-lr` / `unembedding-lr` / `matrix-lr`）就是对应这个设计。 |
| `tokenizer.py` | BPE 分词器 | 把文字切成 token（模型只认数字）。本项目自己训练了一个 vocab=32768 的 BPE 分词器，还定义了对话特殊 token：`<|user_start|>`、`<|assistant_end|>`、`<|python_start|>` 等。 |
| `engine.py` | 推理引擎 | 带 KV Cache 的高效生成器。**亮点是内置"计算器工具调用"状态机**：模型生成 `<|python_start|>12/60<|python_end|>` 时，引擎会真的执行这个表达式并把结果 `0.2` 喂回给模型继续生成——这就是最小化的 "Tool Use / Agent" 实现。 |
| `dataloader.py` / `dataset.py` | 预训练数据加载 | 下载并流式读取 ClimbMix 网页文本分片。 |
| `checkpoint_manager.py` | 检查点管理 | 保存/加载 `base`、`sft`、`rl` 三个阶段的模型权重。 |
| `common.py` | 设备/分布式初始化 | 自动检测 cuda/cpu/mps、初始化 DDP 多卡训练、精度选择（老卡 fp32，A800 bf16）。 |
| `core_eval.py` / `loss_eval.py` | 评估工具 | CORE 指标、BPB（bits per byte，越低越好的语言建模指标）。 |

### 2.2 tasks/ —— 数据集

| 文件 | 数据集 | 用途 |
|------|--------|------|
| `gsm8k.py` | GSM8K 小学数学应用题（~7.5K 训练 / 1.3K 测试） | **本项目的主角**。SFT 阶段当教材，RL 阶段当奖励来源，评估阶段当考卷。 |
| `smoltalk.py` | SmolTalk 通用对话 | SFT 阶段教模型"像个助手一样说话"。 |
| `mmlu.py` / `arc.py` | 学科知识 / 科学推理选择题 | SFT 数据混合 + 通用能力评估。 |
| `humaneval.py` | 代码生成题 | 评估用。 |
| `spellingbee.py` | 拼写任务 | SFT + 评估（模型很容易学会，常作 sanity check）。 |
| `common.py` | `Task` 基类与任务混合 | 定义统一接口：`get_example()` 给训练数据，`evaluate()` 判对错，`reward()` 给 RL 打分。 |

**GSM8K 数据长什么样**（理解这个就理解了一半项目）：

```
Question: Weng 每小时赚 $12，昨天她带了 50 分钟孩子，赚了多少钱？
Answer:  Weng 每分钟赚 12/60 = <<12/60=0.2>>0.2 美元
         50 分钟赚 0.2 x 50 = <<0.2*50=10>>10 美元
         #### 10
```

两个关键格式：
1. `<<表达式=结果>>` 是**计算器调用标记**。`tasks/gsm8k.py` 的 `get_example()` 会把它解析成 `python` / `python_output` 两种消息部件，对应推理时 engine.py 真实执行计算器。
2. `#### 10` 是**最终答案标记**。`extract_answer()` 用正则 `#### (\-?[0-9\.\,]+)` 抠出数字，和标准答案字符串比对——这就是 RL 奖励函数的全部逻辑：**对=1.0，错=0.0**。

### 2.3 scripts/ —— 训练流水线

按执行顺序：

| 顺序 | 脚本 | 阶段 |
|------|------|------|
| 1 | `nanochat.dataset`（模块） | 下载预训练数据 |
| 2 | `tok_train.py` | 训练 BPE 分词器 |
| 3 | `base_train.py` | **预训练** |
| 4 | `chat_sft.py` | **SFT 监督微调** |
| 5 | `train_rl.py` | **RL 强化学习** |
| — | `chat_cli.py` / `chat_web.py` | 和模型聊天（命令行 / Web UI with KaTeX） |
| — | `chat_eval.py` / `eval_report.py` | 评估、生成报告 |
| — | `full_train_a800_v2.sh` | 一键串起 SFT→RL→评估（含历次踩坑注释，强烈建议读） |

---

## 三、三个训练阶段的原理（小白版）

### 3.1 阶段一：预训练（Pretraining）——学会"接话"

**任务**：给模型海量网页文本，让它做一件极其单调的事——**预测下一个 token**。

```
输入: "中国的首都是北"     →  模型应输出: "京"
输入: "1 + 1 ="           →  模型应输出: " 2"
```

损失函数就是交叉熵（cross entropy）：模型对正确下一个 token 给的概率越高，loss 越低。在数十亿 token 上重复这个过程，模型被迫学会语法、常识、甚至简单算术——因为不学会这些就预测不准。

**本项目的实现**（`scripts/base_train.py`）：
- 数据：ClimbMix 网页文本（A800 版下载 100 个分片约 10GB）
- 模型：A800 版 d20（20 层 Transformer），1080 版 d6
- 衡量指标：**BPB**（bits per byte），v2 训练到 0.732
- 产物：`runs/base_checkpoints/` —— 这是一个只会"续写文本"的 Base 模型，你问它问题它不会回答，只会接着你的话往下编

### 3.2 阶段二：SFT（Supervised Fine-Tuning）——学会"对话"

**问题**：Base 模型不懂"现在是用户在提问、轮到你回答了"这个概念。

**解法**：用对话格式的数据继续训练。每条数据被渲染成带特殊 token 的序列：

```
<|user_start|>Weng每小时赚$12...<|user_end|><|assistant_start|>Weng每分钟赚<|python_start|>12/60<|python_end|><|python_output|>0.2<|...|>...#### 10<|assistant_end|>
```

训练目标仍然是预测下一个 token，但有一个关键技巧：**只在 assistant 部分计算 loss**（用户的话不需要模型学着说）。代码里通过把非 assistant 位置的 target 设为 `-1`（ignore index）实现。

**本项目的实现**（`scripts/chat_sft.py`）：
- 数据混合：SmolTalk 460K（通用对话）+ MMLU 500K + **GSM8K ×16 epochs ≈ 120K**（v2 故意把数学数据占比翻倍到 ~9%）+ SpellingBee 280K + 身份数据 2K
- v2 跑满 3000 步，Val BPB 0.49 → 0.354
- 产物：`runs/chatsft_checkpoints/` —— 现在模型会聊天、会按 GSM8K 格式答题、会调用计算器了，但准确率有限（greedy 解码约 16.7%）

**SFT 的本质局限**：它只是在"模仿"标准答案的写法。模型见过的解题路径全是人写的范例，它没机会从自己的错误中学习——这正是 RL 要解决的。

### 3.3 阶段三：RL（强化学习）——学会"做对题"

这是项目最有意思的部分（`scripts/train_rl.py`），用通俗语言走一遍每个训练步：

**第 1 步：采样 Rollouts（让模型自己做题）**

拿一道 GSM8K 题，让当前模型用 temperature=1.0 随机采样生成 N 条（v2 是 32 条）不同的解答。同一道题，模型可能 10 条做对、22 条做错。

```python
seqs_batch, masks_batch = engine.generate_batch(tokens, num_samples=..., temperature=1.0, ...)
```

**第 2 步：打分（奖励函数）**

每条解答抠出 `####` 后面的数字和标准答案比：对 → reward=1.0，错 → reward=0.0。注意这是**结果奖励（outcome reward）**——不看推理过程，只看最终答案。这也是 DeepSeek-R1 使用的 "RLVR"（可验证奖励强化学习）思路：数学题答案可机器验证，不需要人工标注或奖励模型。

**第 3 步：计算优势（Advantage）**

```python
advantages = rewards - rewards.mean()
```

如果这道题 32 条解答里 10 条对，平均 reward = 0.3125：
- 做对的解答：advantage = +0.6875（要鼓励）
- 做错的解答：advantage = −0.3125（要抑制）

用"组内平均"做基线（baseline）是关键设计——它让梯度只反映"这条解答比同组其他解答好多少"，大幅降低方差。**这正是 GRPO（Group Relative Policy Optimization，DeepSeek 提出）的核心思想**：不用单独训练一个 value 网络估计基线，直接用同一道题多条采样的平均分代替。

**第 4 步：策略梯度更新（REINFORCE）**

```python
logp   = -model(inputs, targets, loss_reduction='none')   # 每个 token 的 log 概率
pg_obj = (logp * advantages.unsqueeze(-1)).sum()           # logp × advantage
loss   = -pg_obj / num_valid_tokens                        # token 级归一化（DAPO 风格）
loss.backward()
```

直觉：**提高"好解答"中每个 token 的生成概率，降低"坏解答"的**。模型逐渐把概率质量从错误的推理路径搬到正确的推理路径上。

**和工业级 GRPO/PPO 的差异**（README 也提到了）：

| 组件 | 本项目 | 完整 GRPO/PPO | 缺失的后果 |
|------|--------|--------------|-----------|
| 基线 | 组内均值 ✅ | 组内均值（GRPO）/ value 网络（PPO） | — |
| KL 正则（防止偏离 SFT 模型太远） | ❌ 无 | 有（β·KL 惩罚） | 容易 mode collapse（见下） |
| PPO clipping（限制单步更新幅度） | ❌ 无 | 有 | 训练不稳定、易过冲 |
| 重要性采样比 | ❌ 无（纯 on-policy） | 有 | 每批数据只能用一次 |
| token 级归一化 | ✅ DAPO 风格 | DAPO 提出 | — |

所以这是一个"**带组基线的 REINFORCE**"，可以叫"简化版 GRPO"。代码极其干净（~350 行含日志），非常适合学习。

**真实训练曲线说明了什么**（来自 `reports/A800_TRAINING_REPORT_v2.md`）：

| Step | Pass@1 | Pass@16 | 解读 |
|------|--------|---------|------|
| 0 (SFT 起点) | 5.5% | 37.25% | SFT 模型"潜力"很大（采 16 次有 37% 概率至少对一次），但单次准确率低 |
| 90 | 9.25% | **43.0%** ← 峰值 | RL 早期：在保持多样性的同时提升准确率，最健康的阶段 |
| 232 | 4.75% | 28.5% | **Mode collapse**：模型收敛到少数解题模板，多样性崩塌，连 Pass@1 都跌了 |
| 480 | **14.75%** ← 峰值 | 33.5% | Pass@1 回升到峰值，但 Pass@16 永久受损 |

这条 U 型曲线是没有 KL 正则的 REINFORCE 的经典病症，也是本项目最有教学价值的实验结果：**RL 把 Pass@16 的"潜力"压缩成 Pass@1 的"确定性"，压缩过度就是灾难**。

---

## 四、几个贯穿全项目的关键概念

**Token 与分词器**：模型不认识文字，只认识 0~32767 的整数。"12/60" 可能被切成 ["12", "/", "60"] 三个 token。数学能力差的一个隐藏原因就是数字分词不友好。

**特殊 token 状态机**：对话结构（谁在说话）、工具调用（何时执行计算器）全靠 `<|...|>` 特殊 token 驱动，`engine.py` 里的生成循环就是一个围绕这些 token 的状态机。

**Pass@k**：采样 k 次，至少一次答对的比例。Pass@1 衡量"单次可靠性"，Pass@k（大 k）衡量"模型分布里有没有正确解法"。**RL 的本质就是把 Pass@k 的潜力搬运到 Pass@1**——RL 几乎不会教会模型全新的解法，只是放大已有的正确路径（这也是为什么 SFT 质量决定 RL 上限，v1→v2 的最大教训）。

**温度（temperature）**：t=0 贪心解码（每步选概率最高的 token，结果确定）；t=1 按概率随机采样（有多样性，RL rollout 必须用它来探索）。评估时两种口径会给出完全不同的数字（v2 报告里 16.67% vs 4.0% 的"矛盾"就是这个原因）。

**DDP（多卡训练）**：`train_rl.py` 里 `ddp_rank`/`all_reduce` 那些代码是为多 GPU 准备的——每张卡分到不同的题目做 rollout，梯度求平均后同步更新。单卡跑时这些逻辑自动退化为 no-op。

---

## 五、动手路线建议（按学习收益排序）

1. **跑通推理**：`python3 -m scripts.chat_web`，在浏览器里和已训练的模型聊天，直观感受小模型的能力边界。
2. **读 `tasks/gsm8k.py`**（150 行）：理解数据格式、奖励函数——RL 的"游戏规则"全在这。
3. **读 `scripts/train_rl.py`**（350 行）：对照本文第 3.3 节逐行理解，这是最精炼的 RLHF/RLVR 入门代码。
4. **读 `reports/` 三份训练报告**：v0→v1→v2 的踩坑史（SFT 步数 bug、RL 过拟合、mode collapse）比任何教科书都生动。
5. **改一行代码做实验**：比如给 reward 加格式分（有 `####` 标记 +0.1）、或给 advantage 除以组内标准差（变成标准 GRPO），观察训练曲线变化。

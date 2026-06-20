# 从模仿到自我进化：SFT、强化学习与 GRPO 的一条主线

> 本文不是术语词典，而是一条**因果链**：每一种技术都是为了修复前一种技术的内在矛盾而诞生的。我们沿着 `预训练 → SFT → DQN → 策略梯度 → PPO → RLHF → DPO → GRPO` 这条主线走一遍，并在每一步都对照 **MathGPT 项目的真实代码**（`scripts/chat_sft.py`、`scripts/train_rl.py`、`nanochat/tokenizer.py`、`tasks/gsm8k.py`），看看抽象的算法到底落在哪几行上。

读完你应该能回答三个问题：
1. SFT 到底在"监督"什么？为什么它只是"模仿"，且模仿有天花板？
2. 强化学习这条线为什么从 DQN 一路演化到 GRPO？每一步**解决了什么矛盾、又留下了什么新矛盾**？
3. 为什么 MathGPT 用的"简化版 REINFORCE"本质上就是 GRPO，而 GRPO 恰好是"为有标准答案的数学题量身定制"的？

---

## 第 0 幕：起点的矛盾——预训练模型"会说话，但不会帮忙"

预训练（Base 模型）只做一件事：在海量文本上预测下一个 token。它学到的是**语言的分布**——给定上文，什么样的下文"看起来像人写的"。

但这里有一个根本矛盾：

> **"像人写的"≠"对用户有用"。**

一个 Base 模型看到 "请帮我解方程 2x+5=17" 时，它可能续写成另一道练习题、续写成一篇博客、甚至续写成一段无关的对话——因为这些在训练语料里都"像人写的"。它没有"我应该扮演一个有帮助的助手、现在该我回答了"这个概念。

MathGPT 的 Base 模型（A800 上 depth=20、~700M 参数、2.6B tokens 预训练）就是典型：

```
"The capital of France is Paris"        ✅ 常识对
"The chemical symbol of gold is Au"     ✅ 事实对
"If 5*x + 3 = 13, then x is 13"         ❌ 数学推理错，而且根本没意识到要"解题"
```

**矛盾催生方案**：我们需要教模型"在什么场景、用什么格式、扮演什么角色去回答"。这就是 SFT。

---

## 第 1 幕：SFT——用"模仿"注入行为，以及模仿的天花板

### 1.1 SFT 的本质：把"对话"也当成"预测下一个 token"，但只在回答部分算账

SFT（Supervised Fine-Tuning）没有发明新的损失函数，它仍然是**交叉熵 + 预测下一个 token**。它的全部精巧之处在于一个动作：**用 loss mask 把"用户说的话"屏蔽掉，只对"助手该说的话"计算损失。**

直觉很简单：我们不希望模型学会"模仿用户怎么提问"，我们只希望它学会"在用户提问后，助手怎么回答"。

在 MathGPT 里，这个 mask 是在 `tokenizer.render_conversation` 里逐 token 打上的（`nanochat/tokenizer.py`）：

```python
# 用户消息：整段 mask=0（不监督）
add_tokens(user_start, 0)
add_tokens(value_ids, 0)
add_tokens(user_end, 0)

# 助手消息：mask=1（要监督，这是模型要学着生成的）
add_tokens(assistant_start, 0)        # 起始标记本身不监督
add_tokens(value_ids, 1)              # 助手正文 → 监督
...
# 工具调用：模型"决定调用"的部分要监督
add_tokens(python_start, 1)
add_tokens(value_ids, 1)              # <|python_start|> 16*7 <|python_end|>
add_tokens(python_end, 1)
# 但工具"返回的结果"不监督——那是环境在测试时给的，不是模型的决策
add_tokens(output_start, 0)
add_tokens(value_ids, 0)              # <|output_start|> 112 <|output_end|>
add_tokens(output_end, 0)

add_tokens(assistant_end, 1)          # 学会"在恰当的时候停下来"
```

这段代码藏着两个**极其重要的设计哲学**，后面 RL 还会复用：

1. **只监督"模型自己要产生的决策"**。计算器返回的 `112` 不是模型决定的，是 Python `eval` 算出来强制注入的，所以 mask=0。模型要学的是"何时调用计算器"和"拿到结果后怎么继续",而不是"背下 16*7=112"。
2. **`<|assistant_end|>` 要监督（mask=1）**。"知道什么时候闭嘴"本身是一种需要学习的能力——这一点在第 5 幕会变成 RL 过拟合的导火索。

到了训练脚本里，mask 被翻译成 PyTorch 交叉熵的 `ignore_index = -1`（`scripts/chat_sft.py`）：

```python
# targets 是 inputs 右移一位（预测下一个 token）
targets = batch_tensor[:, 1:]...
# mask=0 的位置（用户、BOS、工具输出、padding）全部设为 -1，不计入 loss
mask_targets = mask_tensor[:, 1:]...
targets[mask_targets == 0] = -1
```

于是 `loss = model(x, y)` 时，所有 `-1` 的位置被交叉熵自动忽略。**SFT loss 比 Base loss 低（项目里 ~0.98 vs ~2.50），不是因为 SFT 更强，而是它只预测"回答部分",任务天然更容易**——两者不可直接比较。

### 1.2 SFT 的工程细节：数据混合、打包、优化器

MathGPT 的 SFT 数据是一锅"混合菜"（`scripts/chat_sft.py`），比例由命令行控制：

```python
train_tasks = [
    SmolTalk(...),                                    # 460K 通用对话 → 学会"怎么对话"
    CustomJSON(identity), CustomJSON(identity),       # 身份认知，2 个 epoch
    *[MMLU(...) for _ in range(args.mmlu_epochs)],    # 学科知识，默认 ×5
    *[GSM8K(...) for _ in range(args.gsm8k_epochs)],  # 数学+工具调用，默认 ×16
    SimpleSpelling(200000), SpellingBee(80000),       # 拼写/计数
]
```

`gsm8k_epochs=16` 意味着数学数据被反复看 16 遍——**这是把通用模型"掰"向数学的关键旋钮**。

两个值得一提的工程点：

- **best-fit 打包不丢 token**：`sft_data_generator_bos_bestfit` 把长短不一的对话用"最佳适配"塞进定长 2048 的行，塞不下就 padding（而不是裁切），padding 位置 mask=0。这样数据利用率高，又不会把一句话拦腰切断。
- **优化器是 MuonAdamW**（`nanochat/gpt.py: setup_optimizer`）：矩阵参数走 Muon（基于 Newton-Schulz 正交化），embedding / lm_head / 标量参数走 AdamW，学习率还按 `1/√(dmodel/768)` 缩放。SFT 直接继承预训练的优化器动量，只是把学习率重置成一个小比例（`init_lr_frac`）继续走。

### 1.3 SFT 的天花板：四个无法回避的矛盾

SFT 很有效，但它是**模仿学习**（imitation learning），而模仿有结构性的上限：

| 矛盾 | 含义 | 后果 |
|------|------|------|
| **只能模仿，不能超越** | 损失函数让模型逼近训练数据的分布 | 数据里没有的解法、比标注更好的解法，模型学不到 |
| **不知道对错** | 交叉熵只关心"和参考答案的 token 像不像" | 一个推理过程哪怕算错了，只要措辞像，loss 照样低 |
| **token 一视同仁** | 每个 token 的 loss 权重相同 | "关键的那一步推理"和"填充的连接词"被平等对待 |
| **数学尤其残酷** | 答案差一个数字就是全错，但 SFT 允许小概率采样到错 token | 推理链一步错、步步错 |

把这四条压成一句话：

> **SFT 教会了模型"怎么说"，但没教会它"说得对不对"——因为它从头到尾没有一个"对错信号"。**

我们需要一个能告诉模型"这次答对了 / 答错了"的信号，并据此调整。这正是强化学习要提供的东西。于是主线从"监督学习"切换到"强化学习"。

---

## 第 2 幕：强化学习的第一次尝试——DQN，以及它为什么救不了语言模型

### 2.1 DQN 的思路：学一个"打分表" Q(s, a)

2013-2015 年，DeepMind 的 DQN 让强化学习第一次"出圈"——直接从 Atari 游戏画面学会打游戏。它的核心是**值函数（value-based）**：学一个函数 `Q(s, a)`，表示"在状态 s 下采取动作 a，未来能拿到的总回报"。

```
Q(s, a) = 即时奖励 r + γ · max_{a'} Q(s', a')   （Bellman 方程）
```

有了 Q，决策就是"在当前状态下，选 Q 值最大的动作"。DQN 还贡献了两个稳定训练的技巧：经验回放（打破样本时序相关性）和目标网络（用滞后副本算目标，防止自举发散）。

### 2.2 DQN 撞上语言模型的墙

DQN 在游戏里能用，是因为 Atari 的动作空间很小（上下左右开火，十几个离散动作）。决策时要算 `max_{a'} Q(s', a')`——**遍历所有动作取最大**。

现在把它搬到语言模型上：

> 语言模型每一步的"动作"是**从 32768 个 token 里选一个**。一句话几十上百步，每步都要对 32768 个动作算 Q 再取 max。

这就是 DQN 的死穴——**它需要枚举动作空间**。在 vocab=32768、序列长度上百的设定下，这条路在计算上根本走不通。

**矛盾催生方案**：我们不要去学"每个动作值多少分"（那要枚举），我们直接去学"该以多大概率选每个动作"——也就是直接优化**策略 π(a|s)**。这把主线从 value-based 推向 policy-based。

---

## 第 3 幕：策略梯度 / REINFORCE——终于能用在文本上了，但方差大到发抖

### 3.1 直接优化策略：每个生成的 token 就是一个动作

策略梯度（Policy Gradient）不学 Q，直接对策略本身求梯度：

```
∇J(θ) = E_{τ~π_θ} [ Σ_t ∇log π_θ(a_t | s_t) · G_t ]
```

`G_t` 是从第 t 步往后的累积回报。翻译成大白话：

> **如果一条轨迹最终拿到了正回报，就提高它沿途每个动作的概率；如果是负回报，就压低。**

这就是 **REINFORCE**（Williams, 1992）。它天然契合语言模型：把"生成一句回答"看成一条轨迹，每个 token 是一个动作，最后用一个标量奖励评价整句话的好坏。**MathGPT 的 RL 训练就建立在这个最朴素的算法之上。**

### 3.2 REINFORCE 的矛盾：方差大，且会"训崩"

朴素 REINFORCE 有两个老毛病：

1. **方差极大**：`G_t` 是采样出来的随机量，同一个策略跑两次可能得到天差地别的梯度，训练像在抖。
2. **步子迈大了会崩**：一次更新如果把策略推得太远，下一轮采样的数据全来自这个"崩坏"的新策略，再也回不来了。

针对第 1 个毛病，标准解法是**减一个基线（baseline）**：

```
A_t = G_t - b(s)     # 优势函数 = 回报 - 基线
```

只要基线 `b` 不依赖于动作，它就不改变梯度的期望，却能大幅降低方差。直觉是：不要问"这个动作的绝对回报是多少",要问"**它比平均水平好多少**"。这个"比平均好多少"的思想，会在第 7 幕的 GRPO 里以最朴素的形式复活。

针对第 2 个毛病（步子太大），需要"限制每次更新的幅度"——这就引出了 PPO。

---

## 第 4 幕：PPO——给策略更新套上"安全带"

PPO（Proximal Policy Optimization, 2017）解决"步子太大会崩"的办法是：**用重要性采样比衡量新旧策略的差距，并把它裁剪（clip）在一个小区间内。**

```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)            # 新旧策略概率比
L_PPO = E[ min( r_t·A_t , clip(r_t, 1-ε, 1+ε)·A_t ) ]
```

`clip` 的含义：如果新策略想把某个动作的概率推得离旧策略太远（比值超出 `[1-ε, 1+ε]`），就把收益截断，不再奖励这种"激进"的更新。这相当于给优化过程套了一条安全带，让 RL 训练稳定下来。

PPO 稳、通用、效果好，于是它顺理成章成为下一幕"对齐人类偏好"的主力算法。但要把 PPO 用起来，还缺一样东西：**奖励从哪来？** 打游戏有分数，写一句话的"奖励"是什么？这就是 RLHF 要回答的。

---

## 第 5 幕：RLHF——把"人类偏好"变成奖励信号，以及它的沉重代价

### 5.1 三步走：SFT → 奖励模型 → PPO

InstructGPT / ChatGPT（OpenAI, 2022）用 RLHF（Reinforcement Learning from Human Feedback）第一次把 RL 大规模用在 LLM 对齐上：

```
Step 1  SFT：人类示范，模型学会基本回答格式（就是我们的第 1 幕）
Step 2  奖励模型 RM：让人类对同一问题的多个回答排序，训练一个"打分器"
                     输入(问题, 回答) → 输出一个标量分数，代理人类的主观偏好
Step 3  PPO：用 RM 当奖励来源，用 PPO 优化策略，
             同时加 KL 惩罚，防止策略偏离 SFT 模型太远（否则会胡说八道去骗高分）
```

RLHF 最大的贡献，是解决了"如何把**主观的、说不清的**人类偏好变成可优化的训练信号"——靠的是一个学出来的奖励模型。

### 5.2 RLHF 的矛盾：贵、复杂、还会被"钻空子"

| 矛盾 | 细节 |
|------|------|
| **要 4 个模型同时在显存里** | policy、reference（算 KL）、reward、value（critic，给 PPO 估优势）|
| **奖励模型要人工标注** | 大量"哪个回答更好"的对比数据，成本高 |
| **奖励模型只是"代理"** | 它是对人类偏好的近似，模型会去**钻它的漏洞**（reward hacking）：生成 RM 打高分但人类其实不喜欢的东西 |
| **实现复杂** | 四模型协同 + KL + clip + value 估计，工程门槛高 |

这套东西对"开放式对话对齐"是值得的，但它催生了两个简化方向：

- 既然奖励模型是麻烦的根源，**能不能不要奖励模型？** → DPO
- 既然 value/critic 网络是另一个麻烦，**对于"有标准答案"的任务，能不能连 critic 也不要？** → GRPO

---

## 第 6 幕：DPO——绕过奖励模型的"抄近路"

DPO（Direct Preference Optimization, 2023）的洞察是：既然最终目标是让模型在偏好对 `(回答_好, 回答_坏)` 上"偏向好的",**那干脆把这个目标直接写成一个分类式的损失函数，跳过"训练奖励模型 + 跑 PPO"这两步**。

DPO 用一个闭式的损失，直接在偏好对上提高"好回答"相对"坏回答"的对数概率比，并用 reference 模型隐式地起到 KL 约束的作用。它把 RLHF 的三步压成一步，省掉了奖励模型和在线采样。

但 DPO 仍然有它的边界：

- 它需要**成对的偏好数据**（还是要标注，只是不用单独训 RM）。
- 它是**离线**的——在固定数据集上学，**没有"探索"**：模型不会主动去尝试新解法再根据结果强化。

对很多任务这够用了。但对数学这种**有客观对错、且希望模型自己探索解题路径**的任务，有一条更直接的路：根本不需要人类偏好，因为"答案对不对"机器自己就能判。这就是 GRPO 的舞台。

---

## 第 7 幕：GRPO——回到 REINFORCE 的朴素，却专为"可验证任务"而生

### 7.1 核心洞察：用"同组内的相对表现"代替 critic

GRPO（Group Relative Policy Optimization, DeepSeek-Math 2024）的思路优雅得让人想起第 3 幕的基线：

> 对同一道题，采样 G 条回答，**用这 G 条的平均分当基线**。高于平均的就是正优势，低于平均的就是负优势。

```
对问题 q 采样 {o_1, ..., o_G}，得到奖励 {r_1, ..., r_G}
A_i = (r_i - mean(r)) / std(r)      # 组内相对优势，不需要任何 value 网络！
```

一句话：**PPO/RLHF 用一个专门的 critic 网络去估"基线该是多少",GRPO 说——不用估，同组的平均分就是最好的基线。** 这一下就砍掉了 critic（少一个大模型）。再加上数学任务的奖励是**规则判定的**（答案对=1，错=0），又砍掉了奖励模型（不用人工标注）。

| 维度 | PPO / RLHF | GRPO |
|------|-----------|------|
| Critic（value 网络）| 需要 | **不需要**（组内均值当基线）|
| 奖励模型 | 需要（人工标注训练）| **不需要**（规则奖励）|
| 同时在显存的模型 | 4 个 | 2 个（policy + 可选 reference）|
| 适用场景 | 开放式对话对齐 | **有明确对错的任务**（数学、代码）|

GRPO 完整版的损失仍保留 PPO 的 clip 和对 reference 的 KL 惩罚：

```
L_GRPO = -E[ min(r_t·A, clip(r_t,1-ε,1+ε)·A) - β·KL(π_θ || π_ref) ]
```

### 7.2 为什么 GRPO 这条主线"绕了一圈又回来了"

值得停下来体会这条主线的形状：

```
DQN（学Q，要枚举动作）
  ↓ 文本动作空间太大，枚举不了
REINFORCE（直接优化策略，但方差大、会崩）
  ↓ 加基线降方差 + 限制步长
PPO（clip 安全带，稳了）
  ↓ 需要奖励来源
RLHF（奖励模型 + PPO + KL，但要4个模型、会reward hacking）
  ↓ 砍掉奖励模型
DPO（直接在偏好对上优化，但离线、无探索、仍需标注）
  ↓ 对"可验证任务"，连critic和RM都不需要
GRPO ≈ 带组内基线的 REINFORCE
```

**绕了一大圈，GRPO 在算法骨架上几乎回到了最朴素的 REINFORCE+baseline**——只不过它精准地砍掉了 RLHF 那套重型机器里、对"有标准答案的任务"而言纯属冗余的部分（critic、奖励模型）。这就是"为合适的问题选合适的工具"的典范。

---

## 第 8 幕：落到代码——MathGPT 的"简化版 GRPO"到底是哪几行

MathGPT 的 `scripts/train_rl.py` 注释自称 "REINFORCE-style policy gradient (simplified GRPO)"。我们把第 7 幕的公式和代码一一对上。

### 8.1 采样一组轨迹（对应"对同一题采 G 条回答"）

```python
# 把对话渲染到"该助手开口"的位置为止（render_for_completion 不含答案）
tokens = tokenizer.render_for_completion(conversation)
prefix_length = len(tokens)

# 对同一道题生成 num_samples 条 rollout（这就是 GRPO 的"组 group"）
seqs_batch, masks_batch = engine.generate_batch(
    tokens, num_samples=args.device_batch_size,
    max_tokens=args.max_new_tokens, temperature=args.temperature, ...)
```

### 8.2 规则奖励（对应"砍掉奖励模型"）

奖励不是学出来的，是一个正则表达式判出来的（`tasks/gsm8k.py`）——这正是 GRPO 适合数学的根本原因：

```python
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")   # GSM8K 答案写在 #### 后面

def evaluate(self, conversation, assistant_response):
    ref_num  = extract_answer(last_text_part)      # 标准答案的数字
    pred_num = extract_answer(assistant_response)  # 模型答案的数字
    return int(pred_num == ref_num)                # 数字匹配 → 1，否则 0

def reward(self, conversation, assistant_response):
    return float(self.evaluate(conversation, assistant_response))  # 1.0 / 0.0
```

### 8.3 组内相对优势（对应"砍掉 critic，用组均值当基线"）

整个 GRPO 的"灵魂"在 MathGPT 里就是**一行**：

```python
rewards    = torch.tensor(rewards, ...)        # 这一组 G 条回答的奖励
advantages = rewards - rewards.mean()          # REINFORCE with baseline = GRPO 的组内基线
```

`rewards.mean()` 就是基线。答对的 advantage 为正（被强化），答错的为负（被抑制），全对或全错时 advantage 全 0（无梯度）。这就是第 3 幕的基线思想 + 第 7 幕的"组内均值",合体成一行。

> **与完整 GRPO 的差异**：MathGPT 简化掉了 (1) 除以 `std` 的归一化、(2) PPO 的 clip、(3) 对 reference 的 KL 惩罚。这让实现极简，但也埋下了第 9 幕的过拟合隐患。

### 8.4 策略梯度损失 + 只对"模型自己生成的 token"算账

```python
logp      = -model(inputs, targets, loss_reduction='none').view_as(inputs)  # log π
pg_obj    = (logp * advantages.unsqueeze(-1)).sum()    # Σ logπ · A
num_valid = (targets >= 0).sum().clamp(min=1)
pg_obj    = pg_obj / (num_valid * num_passes * examples_per_rank)
loss      = -pg_obj                                     # 取负：最小化loss = 最大化好回答概率
loss.backward()
```

注意 `targets` 里那些被设成 `-1` 的位置（来自和 SFT 同源的 mask 逻辑）——**prompt 和工具注入的输出不参与梯度**：

```python
targets[mask_ids[:, 1:] == 0] = -1  # ignore prompt tokens in the loss
```

这就是第 1 幕"只监督模型自己的决策"的哲学在 RL 里的延续：模型只为**自己采样出来的 token**负责。

### 8.5 关键认知：RL 的 loss 不是越低越好

```
Ex 26 | loss: -0.001862 | reward: 0.125    # 部分对部分错 → 有梯度，最理想
Ex 25 | loss: -0.000000 | reward: 0.000    # 16条全错 → advantage全0 → 无梯度
Ex 27 | loss: -0.000326 | reward: 0.969    # 几乎全对 → advantage趋0 → 梯度也小
```

Base/SFT 的 loss 越低代表预测越准；但 **RL 的 loss≈0 往往代表"这道题没有产生任何学习信号"**（全对或全错）。理想状态是 loss 有正有负地波动、同时 reward 整体上升。

---

## 第 9 幕：理论照进现实——MathGPT 训练里 GRPO 真实的样子

主线讲完了，但真实训练会狠狠地提醒你"理论很美，工程很糙"。MathGPT 的 A800 训练（详见 `reports/A800_TRAINING_REPORT_v2.md`）暴露了简化版 GRPO 的几个真实现象，正好回扣前面每一幕：

- **回扣第 1 幕（SFT 是地基）**：SFT 充分训练后，RL 的初始 reward 从 0.07 飙到 0.41（5.9x）。因为 reward 全 0 时 advantage=0、梯度为 0——**SFT 太弱会让 GRPO 根本启动不了**。这是"SFT 是地基"最量化的证据。
- **回扣第 7 幕（砍掉 KL 的代价）**：MathGPT 省掉了对 reference 的 KL 惩罚，结果出现 **mode collapse**——Pass@16 从 43% 跌到 31.5%，模型探索多样性崩塌。这恰恰说明 RLHF/GRPO 里那个 KL 项不是装饰，而是"防止策略跑偏"的保险。
- **回扣第 5 幕（reward hacking 的影子）**：模型学会生成更短的回答、提前吐 `<|assistant_end|>`，因为短回答更容易凑出 `#### number` 拿到规则奖励——这是规则奖励版本的"钻空子"。序列长度从 222 掉到 129 就是证据。
- **train reward 上升 ≠ 模型变好**：train reward 一路涨到 0.45，但 eval Pass@1 在 step 120 后就停滞甚至下降。最优 checkpoint 出现在 17% 处而非终点。

把这些现象和主线对照，你会发现：**GRPO 砍掉的每一样东西（critic、奖励模型、KL、clip），都在某个训练故障里以"它的缺席"被重新证明了价值。** 这也是为什么完整 GRPO 仍保留 KL 和 clip——简化是有代价的，要根据任务权衡。

---

## 尾声：一句话串起整条主线

> **预训练让模型"会说话"；SFT 用模仿教它"按格式扮演助手",但模仿不知对错；强化学习引入"对错信号"——DQN 因动作空间太大出局，REINFORCE 能用却抖，PPO 加安全带稳住，RLHF 用奖励模型把人类偏好接进来但太重，DPO 抄近路省掉奖励模型，GRPO 则对"有标准答案的任务"连 critic 都省掉、用组内均值当基线，几乎回到 REINFORCE 的朴素。MathGPT 用的就是这条主线的终点：一行 `advantages = rewards - rewards.mean()`，配上一个正则表达式判分的规则奖励，让一个 700M 的小模型在 GSM8K 上学会了自己探索解题。**

理解了这条因果链，你就不会把这些算法当成孤立的名词，而会看到它们是同一个问题在不同约束下的连续解答。

---

## 代码索引（按主线顺序）

| 主线环节 | 文件 | 关键位置 |
|---------|------|---------|
| SFT loss mask（监督谁）| `nanochat/tokenizer.py` | `render_conversation`，`add_tokens(.., 0/1)` |
| SFT 训练循环 / 数据混合 / 打包 | `scripts/chat_sft.py` | `train_tasks`、`sft_data_generator_bos_bestfit` |
| 优化器 MuonAdamW | `nanochat/gpt.py` / `nanochat/optim.py` | `setup_optimizer` |
| RL 采样一组轨迹 | `scripts/train_rl.py` | `get_batch`，`engine.generate_batch` |
| 规则奖励 | `tasks/gsm8k.py` | `evaluate` / `reward` |
| GRPO 组内基线（灵魂一行）| `scripts/train_rl.py` | `advantages = rewards - rewards.mean()` |
| 策略梯度损失 + mask | `scripts/train_rl.py` | `pg_obj`、`targets[mask==0] = -1` |
| Pass@k 评估 | `scripts/train_rl.py` | `run_gsm8k_eval` |

## 参考文献

- Mnih et al. (2013). *Playing Atari with Deep Reinforcement Learning*. DQN.
- Williams (1992). *Simple Statistical Gradient-Following Algorithms*. REINFORCE / baseline.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. PPO.
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback*. InstructGPT / RLHF.
- Rafailov et al. (2023). *Direct Preference Optimization*. DPO.
- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning*. GRPO.
- 配套文档：`docs/RL_SFT_GRPO_INTRO.md`（技术总览 + 训练复盘 + CoT 路线图）。
- 训练实录：`reports/A800_TRAINING_REPORT_v1.md`、`reports/A800_TRAINING_REPORT_v2.md`。

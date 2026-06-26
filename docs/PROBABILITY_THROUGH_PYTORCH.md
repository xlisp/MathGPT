# 大学4年没讲明白的概率论，被一段 PyTorch 代码讲透了

> 大学里的概率论，大概是这样的：老师在黑板上写下一行公式，
> $P(A\mid B)=\dfrac{P(AB)}{P(B)}$，然后开始证明，下课。
> 四年下来，你会背贝叶斯、会算泊松、会做卷积，但你心里始终有个声音：
> **"这玩意儿到底有什么用？"**
>
> 直到有一天，你打开一个训练大模型的项目，发现整座 Transformer
> ——从生成一个字，到强化学习自我提升——**全是概率论**。
> 而且不是抽象的公式，是一行行能跑、能 debug、能可视化的 PyTorch 代码。
>
> 这篇文章，我们就用本项目 [MathGPT](../README.md) 的真实代码，
> 把概率论从"考试科目"还原成"工程直觉"，再一路延伸到**怎么 debug 大模型**、
> **怎么预测微信群消息量**，以及**生活中处处可用的概率思维**。

本文配套的所有图都可以用 `docs/images/prob/` 里的脚本复现。我们由浅入深，分七层：

1. softmax：把任意数字变成概率
2. temperature：同一个分布，不同的"自信程度"
3. top-k：砍掉概率分布的长尾
4. 交叉熵：损失函数其实就是"惊讶程度"
5. 条件概率：Transformer 本质就是一台条件概率机器
6. 策略梯度：REINFORCE 用"期望"教模型自我提升
7. 落地：debug Transformer / 预测微信泊松流量 / 生活中的概率思维

---

## 0. 一个被忽略的真相：神经网络的输出从来不是答案，是分布

先记住一句话，它是后面所有内容的地基：

> **语言模型每走一步，吐出来的不是"下一个字"，而是"整个词表上的一个概率分布"。**

在本项目里，模型最后一层就是把隐藏向量投影成 `vocab_size` 个数（叫 **logits**）：

```python
# nanochat/gpt.py  (forward)
logits = self.lm_head(x)                     # (B, T, vocab_size) 每个 token 一个分数
logits = logits[..., :self.config.vocab_size]
logits = logits.float()
softcap = 15
logits = softcap * torch.tanh(logits / softcap)  # 把 logits 平滑压到 [-15, 15]
```

注意：这时候 `logits` 还只是一堆任意实数，可能是 `8.0, 6.5, -3.2, ...`。
它**不是概率**——没有非负、不归一、加起来也不等于 1。

把它变成概率，只需要一个函数：**softmax**。这就是第一层。

---

## 1. softmax：把任意数字变成"概率"

概率分布必须满足两条：每一项 ≥ 0，所有项加起来 = 1。
softmax 用一招同时搞定：先 `exp`（保证为正），再除以总和（保证归一）：

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

本项目里它就藏在采样函数里：

```python
# nanochat/engine.py  sample_next_token()
probs  = F.softmax(vals, dim=-1)                       # logits -> 概率分布
choice = torch.multinomial(probs, num_samples=1, generator=rng)  # 按概率抽一个
```

这两行就是大模型"写字"的核心：
**softmax 把分数变成概率，`multinomial` 按这个概率掷一次骰子。**

大学里 softmax 通常只在"逻辑回归"那一节露个脸，老师不会告诉你：
**你每天用的 ChatGPT，每吐一个字，背后都在做一次 softmax + 掷骰子。**

> 🔑 **概率思维落地 #1**：softmax 不只是公式，它是"**把直觉打分变成可比较的概率**"的通用工具。
> 你给三家公司 offer 打分 `8 / 6.5 / 5`？做个 softmax，你就知道"如果让你随机选，你有多大概率去第一家"。
> 分差越大越笃定，分差越小越纠结——而这个"纠结程度"，下一节就能量化。

---

## 2. temperature：同一个分布，不同的"自信程度"

注意上面代码里有一行 `vals = vals / temperature`。这个 `temperature`（温度）
是理解概率分布"形状"的最好教具。它不改变谁排第一，只改变**分布有多尖**：

```python
# nanochat/engine.py
if temperature == 0.0:
    return torch.argmax(logits, dim=-1, keepdim=True)  # T=0：永远选最大，完全确定
...
vals  = vals / temperature   # T<1 放大差距(更自信)，T>1 抹平差距(更随机)
probs = F.softmax(vals, dim=-1)
```

同样的 logits `[8.0, 6.5, 5.0, ...]`，在不同温度下变成完全不同的分布：

![temperature 对概率分布的影响](images/prob/fig1_temperature.png)

- **T=0.3**：几乎 100% 选 "72"，模型"斩钉截铁"。
- **T=1.0**：第一名约 77%，但偶尔会选别的，有了"创造性"。
- **T=2.0**：分布被抹平，第一名只剩 ~49%，模型开始"胡言乱语"。

本项目推理默认 `temperature=0.6`（见 README 参数表）——
**比 1 略冷，保证数学题答案稳定，又不至于死板。**

> 🔑 **概率思维落地 #2**：temperature 就是"**我有多确定**"的旋钮。
> 做选择时，"高温"是头脑风暴（多探索几个方案），"低温"是拍板执行（选最优）。
> 调试模型答非所问时，第一反应应该是：**温度是不是太高了？**

---

## 3. top-k：砍掉概率分布的长尾

词表有三万多个 token，softmax 之后，绝大多数 token 都有一点点（非零的）概率。
如果完全按这个分布抽样，偶尔会抽到一个排名第 8000、概率 0.0001 的"垃圾词"，
答案就崩了。**top-k 采样**：只保留概率最高的 k 个，其余全部置零再重新归一化。

```python
# nanochat/engine.py  sample_next_token()
if top_k is not None and top_k > 0:
    k = min(top_k, logits.size(-1))
    vals, idx = torch.topk(logits, k, dim=-1)   # 只留前 k 名
    vals  = vals / temperature
    probs = F.softmax(vals, dim=-1)             # 在这 k 个里重新分配 100% 概率
    choice = torch.multinomial(probs, num_samples=1, generator=rng)
    return idx.gather(1, choice)
```

![top-k 截断长尾](images/prob/fig2_topk.png)

左边是完整分布（长尾里全是噪声），右边是 `top-k=8` 之后：红线右边的尾巴被一刀切掉，
概率质量重新集中到靠谱的候选上。本项目默认 `top_k=50`。

> 🔑 **概率思维落地 #3**：top-k 是"**只在靠谱的选项里随机**"。
> 选餐厅别在全城 5000 家里随机，先按评分取 top 10，再随便挑一家——
> 既保留惊喜，又避免踩雷。这就是 top-k 的人生版。

---

## 4. 交叉熵：损失函数其实就是"惊讶程度"

前面讲的是"模型怎么用概率生成"。现在反过来问：**模型怎么学？**

预训练的目标只有一句话：**让正确的下一个字，获得尽可能高的概率。**
怎么把"概率高不高"变成一个能优化的数字？答案是**交叉熵 = 负对数似然**：

$$
\text{loss} = -\log p(\text{正确的下一个字})
$$

本项目里就是一行：

```python
# nanochat/gpt.py  forward()
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
    ignore_index=-1,        # -1 的位置不计损失（比如 prompt 部分）
    reduction=loss_reduction,
)
```

为什么是 `-log(p)`？看这条曲线就懂了：

![交叉熵就是惊讶程度](images/prob/fig3_cross_entropy.png)

- 模型给正确答案 `p=0.9` → loss ≈ 0.1（"我早就知道"，几乎不惊讶，几乎不罚）
- 给 `p=0.5` → loss ≈ 0.69（"半信半疑"）
- 给 `p=0.02` → loss ≈ 3.9（"竟然是这个？！"，巨大的惊讶，巨大的惩罚）

**训练，就是不断减少模型对真实世界的"惊讶程度"。**
这就是为什么交叉熵也叫"困惑度"（perplexity）的来源——困惑越少，模型越懂世界。

大学里"最大似然估计 (MLE)"和"交叉熵"是两章分开讲的，
你可能从没意识到：**最小化交叉熵 ≡ 最大化似然 ≡ 让训练数据在模型眼里最不意外。**
三个名字，一件事。

> 🔑 **概率思维落地 #4**：用"惊讶程度"评估你的判断质量。
> 你预测某事 90% 会发生，结果没发生 → 你应该非常惊讶，并大幅修正认知（大 loss）。
> 你说 50% → 无论结果如何都别太得意（中等 loss）。
> **一个好的预测者，是长期"惊讶总量"最小的人。** 这正是 cross-entropy 在做的事。

---

## 5. 条件概率：Transformer 本质就是一台条件概率机器

现在我们触到了那张被反复念叨、却从没真正"用过"的公式：**条件概率 $P(A\mid B)$。**

大学里它是个抽奖、摸球的玩具。但在大模型里，它是**全部**。
一句话的概率，被链式法则拆成一连串条件概率的乘积：

$$
P(w_1 w_2 \cdots w_n) = \prod_{t=1}^{n} P(w_t \mid w_1, \dots, w_{t-1})
$$

![自回归就是条件概率链](images/prob/fig4_conditional.png)

模型每一步算的，就是 **"给定前面所有字，下一个字是什么"** 的条件分布。
在代码里，这个"给定前面所有字"是靠 **KV Cache** 实现的——
它把"历史"（条件 $B$）缓存下来，新 token 只需在这个条件下计算：

```python
# nanochat/engine.py  generate()
# 把已经算过的 K/V（也就是"条件"，前文上下文）缓存起来
logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
# 在"给定前文"的条件下，对下一个 token 采样
next_ids = sample_next_token(logits, rng, temperature, top_k)
```

`[:, -1, :]` 这个切片很关键：我们只取**最后一个位置**的 logits，
因为我们要的就是 $P(\text{next}\mid \text{前面全部})$。

**这就是"GPT"里的 G（Generative）和 P（Pre-trained）背后的概率内核。**
你大学里背的链式法则 $P(ABC)=P(A)P(B\mid A)P(C\mid AB)$，
正是 ChatGPT 写出每一句话的方式。

### 条件概率，还藏在工具调用里

本项目的模型会算数学题时调用计算器，这本身就是一个**条件概率的状态机**：
当模型采样出 `<|python_start|>`，后续 token 的分布就被"条件化"到了写表达式的模式：

```python
# nanochat/engine.py  generate()  —— 工具调用状态机
if next_token == python_start:
    state.in_python_block = True            # 进入"写代码"的条件
elif next_token == python_end and state.in_python_block:
    expr   = self.tokenizer.decode(state.python_expr_tokens)
    result = use_calculator(expr)           # 算出真实结果
    if result is not None:
        state.forced_tokens.append(output_start)
        state.forced_tokens.extend(self.tokenizer.encode(str(result)))
        state.forced_tokens.append(output_end)  # 强制把结果"喂"回上下文
```

被 `use_calculator` 算出的真实结果会作为**新的条件**注入上下文，
后面所有 token 都在"已知 12×6=72"的条件下生成——
**这就是检索增强 (RAG)、工具调用的概率本质：不断给条件概率添加新的、可靠的条件。**

> 🔑 **概率思维落地 #5（这条最值钱）**：条件概率是"**新信息如何改变你的判断**"。
> - "这个项目能成吗？" → 无条件，瞎猜。
> - "已知团队做过三个同类项目、且拿到了头部客户，这个项目能成吗？" → 条件概率，靠谱多了。
>
> 生活里每一条新信息，都是在给你的判断"加一个条件"。
> 学会问 **"在已知 X 的条件下"**，你就掌握了概率论里唯一真正改变命运的那个公式。

---

## 6. 策略梯度：用"期望"教模型自我提升

最后一层，也是本项目的灵魂：**强化学习 (RL)**。这里概率论从"描述世界"升级为"改变世界"。

预训练让模型学会"像人一样说话"，但说得对不对，没人管。
RL 阶段，我们让模型**对同一道数学题采样多个答案，对的奖励 1，错的奖励 0**，
然后调整概率分布：**让"答对的路径"概率上升，"答错的路径"概率下降。**

```python
# scripts/train_rl.py  get_batch()
# 1) 对同一道题采样 num_samples 个答案（每个都是一次随机游走）
seqs_batch, _ = engine.generate_batch(tokens, num_samples=args.device_batch_size,
                                       temperature=args.temperature, top_k=args.top_k, ...)
# 2) 给每个答案打分：对=1.0, 错=0.0
rewards = [train_task.reward(conversation, tokenizer.decode(s[prefix_length:]))
           for s in generated_seqs]
# 3) 优势 = 奖励 - 平均奖励（这一步是 REINFORCE 的精髓）
rewards    = torch.tensor(rewards, dtype=torch.float, device=device)
advantages = rewards - rewards.mean()
```

为什么要减去平均值 `rewards.mean()`？这就是大学里**期望 $E[X]$** 真正的用武之地：

![REINFORCE 的优势函数](images/prob/fig5_advantage.png)

把平均奖励当作"基准线 (baseline)"，**比平均好的答案优势为正（往上推），比平均差的为负（往下压）**。
这正是"相对于期望的偏离"——你大学里学的方差、期望，在这里变成了"教模型变聪明"的方向盘。

更新的目标函数，就是大学概率论里那个 $E[f(X)]$ 的策略梯度版：

```python
# scripts/train_rl.py  训练循环
# 策略梯度目标：最大化 E[logp * advantage]
logp   = -model(inputs, targets, loss_reduction='none').view_as(inputs)  # 每个 token 的对数概率
pg_obj = (logp * advantages.unsqueeze(-1)).sum()
loss   = -pg_obj
loss.backward()
```

一行 `logp * advantage` 就讲完了 REINFORCE：

> **正确答案里的每个字，提高它的概率；错误答案里的每个字，降低它的概率；
> 提高/降低的力度，正比于这个答案比平均"好/坏"多少。**

这是一个干净的 REINFORCE 实现（见 [README 的 RL 原理](../README.md#rl-训练原理)），
无 KL 正则、无 PPO clipping。整个"大模型对齐"的魔法，剥到底层，
就是大学概率论里的**期望、采样、对数似然**这三件套。

> 🔑 **概率思维落地 #6**：期望思维 = "**不看单次结果，看长期平均**"。
> 一次决策的好坏别用单次结果判断（你可能只是运气好/坏），
> 要问"这个策略重复 1000 次，期望收益是正还是负？"。
> 减 baseline 的智慧是：**别和绝对值较劲，和'平均水平'比**——这才是进步的方向。

---

## 7. 落地三连：debug 大模型 / 预测微信流量 / 生活概率思维

> 学概率，"**用到生活中形成映射反应，才不会被忘掉**"。下面把上面六层接回真实世界。

### 7.1 如何 debug 一个 Transformer：把概率画出来

模型答错了，怎么 debug？传统程序你能打断点看变量，但 Transformer 是概率机器，
**最有效的调试手段是把"概率分布"可视化出来。** 几个实战招式：

**① 画 per-token 熵 (entropy)，找到模型"心虚"的地方。**
熵 $H=-\sum p\log p$ 衡量分布有多"散"。熵高 = 模型很不确定（容易出错的决策点）；
熵≈0 = 模型很笃定（或是被强制注入的工具结果）。

![用逐 token 熵调试 Transformer](images/prob/fig7_entropy.png)

```python
# 调试片段：在 engine.generate 的采样处加几行，记录每步分布的熵
import torch.nn.functional as F
probs   = F.softmax(logits / temperature, dim=-1)
entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1)   # 每个 token 的熵
# 把 entropy 存下来画成上面那张图：尖峰处就是模型"拿不准"的关键决策点
```

熵的尖峰，往往就是模型推理链断裂、答案开始跑偏的地方——**先去看那里。**

**② 对比 temperature 扫描。** 答案不稳定？把 `temperature` 从 0 扫到 1.5，
看答案在什么温度下开始崩——崩得早说明模型对这道题本就没把握（见第 2 节）。

**③ 看 top-k 候选词。** 把每步的 `torch.topk(logits, 5)` 打印出来，
如果正确答案根本不在前 5，说明问题在**模型能力/训练**，而不是采样策略。

**④ 用 pass@k 量化"运气成分"。** 本项目评估时正是这么做的：

```python
# scripts/train_rl.py  —— pass@k：采样 k 次，至少对一次的比例
for k in range(1, args.device_batch_size + 1):
    passk[k-1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
```

`pass@1` 低但 `pass@8` 高 → 模型其实"会做"，只是采样没采到（调采样/RL 能救）；
`pass@8` 也低 → 模型是真不会（得加训练数据/调模型）。**这是一条极其重要的诊断分界线。**

### 7.2 预测微信群流量：泊松分布

> "预测微信流量泊松分布"——这是把概率论用回生活最漂亮的例子。

**泊松分布**专门描述"单位时间内，独立随机事件发生的次数"：
群消息数、客服来电数、网站请求数、地铁站到人数……都近似泊松。

$$
P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

$\lambda$ 是"平均每小时几条消息"。下图是不同活跃度的群的消息分布：

![微信群消息的泊松分布](images/prob/fig6_poisson.png)

这就是"**概率编程**"最朴素的样子——几行代码就能拟合并预测：

```python
import numpy as np
from scipy import stats

# 过去 7 天，每小时的微信群消息数（你的真实数据）
counts = np.array([3, 5, 2, 8, 6, 4, 7, 5, 9, 1, 4, 6, ...])

lam = counts.mean()                       # 最大似然估计：λ̂ 就是样本均值
print(f"平均每小时 {lam:.1f} 条")

# 预测：下一个小时消息数 ≥ 10 的概率（该不该开免打扰？）
p_busy = 1 - stats.poisson.cdf(9, lam)
print(f"下一小时爆群(≥10条)的概率：{p_busy:.1%}")
```

注意 `lam = counts.mean()`——**泊松的 λ 的最大似然估计就是样本均值**，
和第 4 节"交叉熵 ≡ 最大似然"是同一套思想。概率论的内核，到处都在复用。

更进一步，用概率编程框架（如 PyMC）还能做**贝叶斯估计**，
不仅给出 λ 的点估计，还给出"我对这个 λ 有多确定"的整条后验分布——
这又回到了第 5 节的条件概率：**用观测数据，把先验更新成后验。**

### 7.3 生活中处处的概率思维

把这篇文章的六层，翻译成日常决策的反射动作：

| 概率概念 | 代码里 | 生活里的"映射反应" |
|---------|--------|------------------|
| softmax | logits → 概率 | 把模糊的"打分/好感"变成可比较的概率 |
| temperature | 控制分布尖锐度 | 探索期调高温（多试），执行期调低温（拍板） |
| top-k | 砍掉长尾 | 只在靠谱的少数选项里随机，别在全集里碰运气 |
| 交叉熵 / 似然 | `-log(p)` | 做长期"惊讶最小"的预测者，错了就大幅修正 |
| **条件概率** | KV Cache / 工具调用 | 永远问"在已知 X 的条件下"，让新信息更新判断 |
| 期望 / 优势 | `reward - mean` | 看长期期望而非单次结果，和平均水平比 |
| 泊松 | `λ = mean` | 预测"单位时间内随机事件次数"：排队、消息、故障 |

> "**条件概率就很有用**"——这句大白话其实是整个概率论里最实用的一句。
> 贝叶斯定理 $P(A\mid B)=\dfrac{P(B\mid A)P(A)}{P(B)}$ 不是用来考试的，
> 它是你每次"看到新证据、更新旧看法"时，大脑本该执行的运算。
> 医生看到化验单更新诊断、投资人看到财报更新估值、你看到对方已读不回更新心情——
> **全是条件概率。**

---

## 结语：概率思维，概率编程

回到开头那个问题——大学四年的概率论为什么没讲明白？

因为它把概率教成了"**计算**"：摸球、抽签、求积分。
而概率真正的样子是"**建模与决策**"：
- 把不确定的世界，建模成一个**分布**（softmax / 泊松）；
- 用**数据**去拟合、更新这个分布（最大似然 / 贝叶斯 / 交叉熵）；
- 在分布上做**采样与决策**（temperature / top-k / 期望最大化）。

这三步，就是这个 MathGPT 项目从预训练、到 SFT、到 RL 的全过程，
也是 ChatGPT 写出每一个字的全过程，更是你做每一个理性决策的全过程。

> **概率思维，是把"我觉得"换成"有多大概率"；**
> **概率编程，是把这个判断写成几行能跑、能验证、能 debug 的代码。**

当你能在一段 PyTorch 里同时看见 softmax、条件概率、期望和最大似然，
并且能把它们映射回"调温度、减 baseline、预测群消息"这些具体动作时——
概率论才算真正长进了你的直觉里，**再也忘不掉。**

---

### 附：复现本文所有图

本文 7 张图均由 matplotlib 生成，逻辑与本项目代码一致（softmax / top-k / 交叉熵 / 优势函数 / 泊松 / 熵）。
配套脚本见 `docs/images/prob/`，核心数学就是上文每段贴出的那几行。

### 延伸阅读（本项目内）

- [README.md](../README.md) — MathGPT 总览与 RL 训练原理
- [docs/RL_SFT_GRPO_INTRO.md](RL_SFT_GRPO_INTRO.md) — SFT / GRPO / RL 的演进
- [docs/SFT_TO_GRPO_NARRATIVE.md](SFT_TO_GRPO_NARRATIVE.md) — 从 SFT 到 GRPO 的叙事
- 关键源码：`nanochat/engine.py`（采样/工具调用）、`nanochat/gpt.py`（交叉熵）、`scripts/train_rl.py`（策略梯度）

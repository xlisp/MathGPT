# MathGPT 下一步改进建议：如何训练出更强的数学模型

> 基于对当前代码（`scripts/train_rl.py`、`scripts/chat_sft.py`、`tasks/gsm8k.py`）和 v0/v1/v2 三轮训练报告的分析。
> 当前状态：d20 模型，SFT greedy ~16.7%，RL 峰值 Pass@1 = 14.75% (step 480)，Pass@16 峰值 = 43.0% (step 90)，存在明显 mode collapse。
>
> 建议按 ROI（投入产出比）排序，分为四个层级：算法修复 → 数据增强 → 训练策略 → 规模扩展。

---

## 〇、先解决已知 Bug（零成本，必做）

### 0.1 `--num-epochs=1` 没有生效

v2 报告明确指出：脚本实际跑了 3 epoch（698 步）而非配置的 1 epoch。先排查 `full_train_a800_v2.sh` 实际执行的命令 / 参数传递链，确认 `train_rl.py` 收到的 `args.num_epochs`。v2 的两个峰值（step 90 和 step 480）之间隔着一个深谷，说明训练调度本身就是最大的性能变量。

### 0.2 评估口径统一

v2 中 ChatCORE (t=0, 24 题) 与 Pass@k (t=1.0, 100/400 题) 数字差异巨大造成误判。建议固定一个主指标：**GSM8K test 全量 1319 题，greedy (t=0) Pass@1**，所有版本可比。RL 期间的 pass@k eval 可保留 t=1.0 但题数提高（400→全量），消除 24 题口径的噪声。

---

## 一、RL 算法层改进（改动小、收益大）

### 1.1 加入 KL 正则 —— 治疗 mode collapse 的特效药 ⭐⭐⭐

v2 最大病症：Pass@16 从 43% 崩到 28.5%，多样性永久受损。根因是纯 REINFORCE 没有任何"别离 SFT 模型太远"的约束。

最小实现（在 `train_rl.py` 的 loss 里加一项）：

```python
# 训练前冻结一份 SFT 模型作 ref_model（eval 模式，no_grad）
with torch.no_grad():
    ref_logp = -ref_model(inputs, targets, loss_reduction='none').view_as(inputs)

# k3 估计器（GRPO 论文用法，无偏且方差低）
log_ratio = ref_logp - logp
kl = (log_ratio.exp() - 1) - log_ratio          # k3: e^x - 1 - x >= 0
loss = -(pg_obj - kl_beta * (kl * valid_mask).sum() / num_valid)
```

- 代价：显存翻倍（多一份冻结模型）或用 LoRA-only 训练规避；80GB A800 上 d20 完全装得下。
- 建议 `kl_beta` 从 0.01~0.05 网格搜索。
- 预期效果：Pass@16 不再崩塌，可以安全地训更多步，让 Pass@1 持续爬升而不是 U 型震荡。

### 1.2 升级为标准 GRPO：advantage 除以组内标准差 ⭐⭐⭐

当前 `advantages = rewards - rewards.mean()`，一行改动即成标准 GRPO：

```python
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
```

作用：当一道题 32 条只对 1 条（或只错 1 条）时，原始 advantage 很小，梯度信号弱；归一化后这些"边缘题"（恰恰是模型能力边界、最值得学的题）获得与中等难度题同等的梯度权重。

### 1.3 过滤零梯度组（DAPO 的 Dynamic Sampling）⭐⭐⭐

全对或全错的组 advantage 全为 0，白白浪费一次 forward/backward。统计上 SFT 起点 Pass@1≈5.5% 时，**大量题目是 32 条全错**——这部分计算完全无效。

```python
if rewards.std() == 0:   # 全对或全错
    continue             # 跳过，重采下一题补足 batch
```

DAPO 论文证明这一项单独就能显著加速收敛。配合"题目难度课程"（见 2.3）效果更好。

### 1.4 加入 PPO-style clipping，允许 off-policy 复用

当前纯 on-policy：每批 rollout 只能更新一次。加入 ratio clipping 后同一批数据可以做 2~4 个 epoch 的更新，rollout（训练中最贵的部分，占 RL 时间大头）利用率翻倍：

```python
ratio = (logp - old_logp.detach()).exp()
pg = torch.min(ratio * adv, ratio.clamp(1-eps, 1+eps) * adv)
```

注意 v2 RL 用了 15.6 小时，其中绝大部分是采样而非反向传播——这条是降低墙钟时间的关键。

### 1.5 奖励塑形（reward shaping）—— 谨慎使用 ⭐

当前 reward 是 {0, 1} 硬二值。可叠加小额格式分：

- 输出包含 `####` 标记：+0.1（治疗 v1/v2 报告中频发的"截断不出答案"问题）
- 正确使用 `<|python_start|>` 计算器且执行成功：+0.05
- 超长未终止（撞到 max-new-tokens）：−0.1

注意上限：塑形分必须远小于正确性分（1.0），否则模型会学会"刷格式不解题"。`tasks/gsm8k.py` 的 `reward()` 注释本身就预留了这个扩展点。

---

## 二、数据层改进（中等成本、上限最高）

### 2.1 SFT 阶段引入更强的数学 CoT 数据 ⭐⭐⭐

RL 只能放大 SFT 已有的正确路径（v2 实验铁证：SFT reward 起点 0.07→0.41 直接决定了 RL 成败）。所以**提升 SFT 数学数据的质量和数量是上限最高的一招**：

| 数据集 | 规模 | 说明 |
|--------|------|------|
| **MetaMathQA** | 395K | GSM8K/MATH 的改写增强（rephrasing、self-verification、FOBAR 逆向题），与现有 pipeline 格式最接近，首选 |
| **OpenMathInstruct-2** | 14M | NVIDIA 出品，Llama-3.1 合成，质量高，可抽样 200K~1M |
| **NuminaMath-CoT** | 860K | 竞赛题为主，可作为进阶混入（比例放低，防止过难） |
| **GSM8K-RFT / 自蒸馏** | 自产 | 见 2.2 |

实现成本低：`tasks/customjson.py` 已支持自定义 JSONL，只需把数据转成 conversation 格式。建议把 SFT 数学占比从当前 ~9% 提到 25~40%（本项目目标就是数学专精，不必过度顾忌通用能力）。

### 2.2 拒绝采样自蒸馏（Rejection Sampling / RFT）⭐⭐⭐

这是 Llama 2/3、Qwen-Math 都在用的"免费午餐"，和现有代码天然契合：

1. 用当前最优 RL checkpoint (step 480) 对 GSM8K **训练集**每题采样 32~64 条（`engine.generate_batch` 现成）
2. 用 `task.reward()` 筛出答对的解答，按多样性去重（如按解题路径 n-gram 去重）
3. 把这些"模型自己写的正确解答"作为新 SFT 数据，重新 SFT 一轮
4. 在新 SFT 模型上再做 RL

Pass@16=43% 意味着约 43% 的题能采出至少一条正确解答——这些数据比人写的 GSM8K 原始答案更贴近模型自己的分布，SFT 学起来更高效。**这等于把 RL 浪费掉的 Pass@k 潜力回收成 Pass@1。** 形成 SFT→RL→蒸馏→SFT 的迭代循环（即 ReST / Iterative RFT）。

### 2.3 难度课程（Curriculum）⭐⭐

GSM8K 题目难度差异大（v2 案例分析：1-2 步加减法 7/16 对，比例题 0/16 对）。建议：

1. 离线预处理：用 SFT 模型对全训练集采样 8 条，按通过率给每题打难度标签
2. RL 训练顺序：先训通过率 30~70% 的"学习区"题目，逐步纳入难题
3. 全错的题（0/32）放到模型变强后再回炉——它们当前提供零梯度（见 1.3）

### 2.4 测试集污染检查（hygiene）

自蒸馏和外部数据引入后，务必对 GSM8K test 做 n-gram 去重检查，否则评估数字失真。

---

## 三、训练策略改进

### 3.1 RL 调度：基于 eval 的早停 + 最优 checkpoint 自动选择 ⭐⭐

v2 暴露的问题：峰值在 step 90/480，但训练跑到 698 步才停。建议在 `train_rl.py` 加：

- 维护 `best_pass1`，eval 后若刷新则另存 `model_best.pt`
- 连续 N 次 eval（如 4 次）无提升则自动早停
- eval 集固定 seed 固定题目，保证可比

### 3.2 长度与截断管理 ⭐⭐

三轮报告都出现"截断"问题。两个方向：

- `max-new-tokens` 768 → 1024（A800 显存足够），并在 reward 中对撞上限的样本给轻微负分（见 1.5），让模型学会收敛到 `####`
- 监控 `sequence_length` 曲线：若 RL 中 seq_len 持续上升而 reward 不升，说明模型在"用废话拖延"，是 reward hacking 前兆

### 3.3 温度调度

rollout 用 t=1.0 探索没问题，但可以试**双温采样**：每题 32 条中 24 条 t=1.0 + 8 条 t=0.7，兼顾探索与利用。eval 一律 t=0 greedy（主指标）。

### 3.4 SFT 阶段的 NEFTune / 学习率微调

低成本试验：SFT 加 NEFTune 噪声（embedding 加均匀噪声）在小模型上常有 1-3pp 免费提升；SFT 末期 LR 衰减到 0 而非保持线性（检查 `chat_sft.py` 当前调度）。

---

## 四、模型与算力层（成本最高，最后考虑）

### 4.1 扩大模型 ⭐⭐

d20（~560M 级别参数量级）在 GSM8K 上的天花板有限。文献参考：同等数据下模型规模对数学推理的影响是超线性的。A800-80GB 可支撑：

- **d26 / d32**：预训练时间按比例增加（v1 base 训练 5000 步的成本可查 v1 报告），但 SFT+RL 成本增幅有限
- 或者**换底座**：放弃从零预训练，加载开源小模型（如 Qwen2.5-0.5B/1.5B-base）做 SFT+RL。代价是要适配 tokenizer 和模型结构（工程量中等），收益是预训练质量碾压自训的 10GB 数据。**如果目标是"更强的数学模型"而非"全程从零"，这是性价比最高的一条路**——Qwen2.5-1.5B 底座 + 本项目的 SFT/RL pipeline，GSM8K 上 60%+ 是合理预期。

### 4.2 扩大预训练数学密度

若坚持从零预训练：在 ClimbMix 基础上混入 OpenWebMath / FineMath 等数学网页语料（10~30%），base 模型的数学先验直接决定下游上限。

### 4.3 推理时增强（不训练也能涨分）

- **Self-consistency / majority voting**：采样 k 条取多数答案，Pass@16=43% 的模型用 maj@16 估计能到 25~30%，作为产品功能几乎免费
- **强制工具调用**：prompt 引导模型对所有算术使用计算器（engine 已支持），消除纯计算错误

---

## 五、建议的执行路线图

| 阶段 | 内容 | 预期 GSM8K greedy Pass@1 |
|------|------|--------------------------|
| 当前 | d20, SFT + 简化 REINFORCE | ~15% (step 480) |
| **第 1 轮**（1~2 天） | Bug 修复 + GRPO 标准化（1.2）+ 零梯度过滤（1.3）+ KL 正则（1.1）+ 早停（3.1） | 18~22% |
| **第 2 轮**（3~5 天） | MetaMathQA 加入 SFT（2.1）+ 重新 SFT→RL | 25~35% |
| **第 3 轮**（1 周） | 拒绝采样自蒸馏迭代 ×2（2.2）+ 课程（2.3） | 30~40% |
| **第 4 轮**（可选） | 换 Qwen2.5-1.5B 底座复用全 pipeline | 55~70% |

每轮只改一类变量，保留 v2 报告的对照实验风格（这是本项目目前做得最好的地方，务必延续）。

---

## 附：每条建议对应的代码改动位置

| 建议 | 文件 | 位置 |
|------|------|------|
| KL 正则 / GRPO std / clipping | `scripts/train_rl.py` | `get_batch()` 的 advantage 计算 + 训练循环的 loss 计算 |
| 零梯度过滤 | `scripts/train_rl.py` | `get_batch()` yield 之前 |
| 奖励塑形 | `tasks/gsm8k.py` | `reward()` 方法（注释已预留扩展点） |
| 新 SFT 数据 | `tasks/customjson.py` + `scripts/chat_sft.py` | 数据混合配置 |
| 自蒸馏采样脚本 | 新建 `scripts/distill_rft.py` | 复用 `engine.generate_batch` + `task.reward` |
| 早停/最优保存 | `scripts/train_rl.py` | eval 块之后 |
| maj@k 投票 | `scripts/chat_eval.py` 或 `engine.py` | 评估/推理路径 |

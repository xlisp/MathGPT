# MathGPT 下一步改进建议：如何训练出更强的数学模型

> 基于对当前代码（`scripts/train_rl.py`、`scripts/chat_sft.py`、`tasks/gsm8k.py`）、v0/v1/v2 三轮训练报告，以及 `runs/tb_logs/` 训练日志的取证分析。
> 当前状态：d20 模型，SFT greedy ~16.7%；RL 阶段的"峰值数据"因实验污染（见第〇节）暂不可信，需重新评估。
>
> 建议按 ROI（投入产出比）排序，分为五个层级：实验取证 → 算法修复 → 数据增强 → 训练策略 → 规模扩展。

---

## 〇、先重建实验真相（零训练成本，最高优先级）⚠️

### 0.1 v2 的 RL 评估表拼接了两次不同的训练 —— 现有结论不可信

**证据链**：

1. `runs/tb_logs/train_rl/default/` 下存在**两个 TensorBoard event 文件，来自两个不同进程**（PID 225591 / 264449）→ RL 实际跑了两次，且都写入同一个 checkpoint 目录 `chatrl_checkpoints/math_d20/`。
2. 报告评估了 17 个 checkpoint：30, 60, ..., 210, **232**, 240, 300, ..., 660, 698。算术严丝合缝：
   - 一次 **3-epoch 跑**（699 步，`save-every=60`，疑似手动执行了 v1 配置的命令）→ 12 个 checkpoint：60, 120, 180, 240, ..., 698
   - 一次 **1-epoch 跑**（233 步，`save-every=30`，真正的 v2 配置）→ 8 个 checkpoint：30, 60, ..., 210, 232
   - 合并后 3 个碰撞步（60/120/180）被后跑覆盖 → **12 + 8 − 3 = 17** ✓
   - 这同时解释了"232 之前间隔 30、之后间隔 60"和"232 不是 30 的倍数"两个反常（233 步 0-indexed 末步即 232）。

**后果**：

- v2 报告的"U 型轨迹 / Epoch 1 末崩溃 / Epoch 3 反弹"叙事是**拼接伪影**——曲线在 step 232/240 处从一次训练跳到了另一次训练。
- "Step 90 = Pass@16 全局最优"与"Step 480 = Pass@1 全局最优"来自**两次配置不同的训练**，不可比，不能作为调参依据。
- "为什么 `--num-epochs=1` 没生效"的谜题反转：它生效了（232 步那次），是另一次 3-epoch 跑污染了 checkpoint 目录。

**待办**：

1. 在 `train_rl.py` 的 `save_checkpoint` 路径中加入 run 名/时间戳（如 `chatrl_checkpoints/math_d20_{run}_{timestamp}/`），并把完整 `user_config` + git commit hash 写入 checkpoint meta——保证每个 checkpoint 永远可审计。
2. 重新只评估能确认归属的 checkpoint（30~232 这条 1-epoch 链是干净的），重写 v2 结论。
3. 在干净基础上重跑一次受控的 RL（哪怕配置不变），拿到第一条可信的完整曲线。

### 0.2 评估噪声大于结论的效应量

400 题、t=1.0 的 Pass@1，二项分布标准误 ≈ √(0.13×0.87/400) ≈ **1.7pp**。报告中 step 480 (14.75%) vs step 360 (12.0%) vs step 698 (12.75%) 的差距均在 1~2 个标准误内——"全局最优在 480"很可能只是在噪声中挑了最大值。

**修复**：checkpoint 选择和版本对比一律使用 **GSM8K test 全量 1319 题、t=0 greedy 的 Pass@1** 作为主指标（确定性、无采样噪声、样本量最大）。t=1.0 的 Pass@k 保留用于监控多样性/mode collapse，但不用于选点。

### 0.3 loss 归一化与 DAPO 声明不符（代码级）

`train_rl.py` 当前：

```python
pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
```

`num_valid` 是**当前 pass**（device_batch_size 条序列）的有效 token 数，乘以 `num_passes` 后得到的是"每个 pass 等权"，而非 DAPO 的"全 batch token 等权"。RL 中错误答案往往更短，短序列 pass 的每个 token 被系统性上调权重。修复：先对整个 example 的所有 pass 求和 `pg_obj` 与 `num_valid`，最后统一相除：

```python
# 累积阶段
pg_sum    += (logp * advantages.unsqueeze(-1) * (targets >= 0)).sum()
valid_sum += (targets >= 0).sum()
# 该 example 所有 pass 结束后
loss = -(pg_sum / valid_sum.clamp(min=1)) / examples_per_rank
loss.backward()   # 注意：需把 backward 移到 pass 循环外，或保持逐 pass backward 但用全局 valid_sum（需先预扫一遍长度）
```

同时更新 README 中"DAPO 风格 token 级归一化"的描述，或修代码使其名实相符。

---

## 一、RL 算法层改进（改动小、收益大）

> 前提：第〇节完成后，在干净的基线上逐项做对照实验。

### 1.1 加入 KL 正则 —— 治疗 mode collapse 的特效药 ⭐⭐⭐

干净的 1-epoch 链（step 30→232）显示 Pass@16 从 43% 跌至 28.5%——即使排除拼接伪影，**单次训练内的多样性崩塌依然真实存在**。根因是纯 REINFORCE 没有任何"别离 SFT 模型太远"的约束。

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

- 代价：显存翻倍（多一份冻结模型）；80GB A800 上 d20 完全装得下。
- 建议 `kl_beta` 从 0.01~0.05 网格搜索。
- 预期效果：Pass@16 不再崩塌，可以安全地训更多步，让 Pass@1 持续爬升。

### 1.2 升级为标准 GRPO：advantage 除以组内标准差 ⭐⭐⭐

当前 `advantages = rewards - rewards.mean()`，一行改动即成标准 GRPO：

```python
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
```

作用：当一道题 32 条只对 1 条（或只错 1 条）时，原始 advantage 很小，梯度信号弱；归一化后这些"边缘题"（恰恰是模型能力边界、最值得学的题）获得与中等难度题同等的梯度权重。

### 1.3 过滤零梯度组（DAPO 的 Dynamic Sampling）⭐⭐⭐

全对或全错的组 advantage 全为 0，白白浪费一次 forward/backward。SFT 起点 Pass@1≈5.5% 时，**大量题目是 32 条全错**——这部分计算完全无效。

```python
if rewards.std() == 0:   # 全对或全错
    continue             # 跳过，重采下一题补足 batch
```

DAPO 论文证明这一项单独就能显著加速收敛。配合"题目难度课程"（见 2.3）效果更好。

### 1.4 加入 PPO-style clipping，允许 off-policy 复用

当前纯 on-policy：每批 rollout 只能更新一次。加入 ratio clipping 后同一批数据可以做 2~4 个 epoch 的更新，rollout（RL 墙钟时间的大头）利用率翻倍：

```python
ratio = (logp - old_logp.detach()).exp()
pg = torch.min(ratio * adv, ratio.clamp(1-eps, 1+eps) * adv)
```

### 1.5 奖励塑形（reward shaping）—— 谨慎使用 ⭐

当前 reward 是 {0, 1} 硬二值。可叠加小额格式分：

- 输出包含 `####` 标记：+0.1（治疗报告中频发的"截断不出答案"问题）
- 正确使用 `<|python_start|>` 计算器且执行成功：+0.05
- 超长未终止（撞到 max-new-tokens）：−0.1

注意上限：塑形分必须远小于正确性分（1.0），否则模型会学会"刷格式不解题"。`tasks/gsm8k.py` 的 `reward()` 注释本身就预留了这个扩展点。

---

## 二、数据层改进（中等成本、上限最高）

### 2.1 SFT 阶段引入更强的数学 CoT 数据 ⭐⭐⭐

RL 只能放大 SFT 已有的正确路径（v1→v2 实验铁证：SFT reward 起点 0.07→0.41 直接决定了 RL 成败）。所以**提升 SFT 数学数据的质量和数量是上限最高的一招**：

| 数据集 | 规模 | 说明 |
|--------|------|------|
| **MetaMathQA** | 395K | GSM8K/MATH 的改写增强（rephrasing、self-verification、FOBAR 逆向题），与现有 pipeline 格式最接近，首选 |
| **OpenMathInstruct-2** | 14M | NVIDIA 出品，Llama-3.1 合成，质量高，可抽样 200K~1M |
| **NuminaMath-CoT** | 860K | 竞赛题为主，可作为进阶混入（比例放低，防止过难） |
| **GSM8K-RFT / 自蒸馏** | 自产 | 见 2.2 |

实现成本低：`tasks/customjson.py` 已支持自定义 JSONL，只需把数据转成 conversation 格式。建议把 SFT 数学占比从当前 ~9% 提到 25~40%（本项目目标就是数学专精，不必过度顾忌通用能力）。

### 2.2 拒绝采样自蒸馏（Rejection Sampling / RFT）⭐⭐⭐

这是 Llama 2/3、Qwen-Math 都在用的"免费午餐"，和现有代码天然契合：

1. 用确认归属后的最优 RL checkpoint 对 GSM8K **训练集**每题采样 32~64 条（`engine.generate_batch` 现成）
2. 用 `task.reward()` 筛出答对的解答，按多样性去重（如按解题路径 n-gram 去重）
3. 把这些"模型自己写的正确解答"作为新 SFT 数据，重新 SFT 一轮
4. 在新 SFT 模型上再做 RL

Pass@16≈43% 意味着约 43% 的题能采出至少一条正确解答——这些数据比人写的 GSM8K 原始答案更贴近模型自己的分布，SFT 学起来更高效。**这等于把 RL 浪费掉的 Pass@k 潜力回收成 Pass@1。** 形成 SFT→RL→蒸馏→SFT 的迭代循环（即 ReST / Iterative RFT）。

### 2.3 难度课程（Curriculum）⭐⭐

GSM8K 题目难度差异大（案例分析：1-2 步加减法 7/16 对，比例题 0/16 对）。建议：

1. 离线预处理：用 SFT 模型对全训练集采样 8 条，按通过率给每题打难度标签
2. RL 训练顺序：先训通过率 30~70% 的"学习区"题目，逐步纳入难题
3. 全错的题（0/32）放到模型变强后再回炉——它们当前提供零梯度（见 1.3）

### 2.4 测试集污染检查（hygiene）

自蒸馏和外部数据引入后，务必对 GSM8K test 做 n-gram 去重检查，否则评估数字失真。

---

## 三、训练策略改进

### 3.1 实验管理：可审计的 checkpoint + 早停 ⭐⭐⭐（与 0.1 配套）

- checkpoint 目录含 run 名/时间戳，meta 写入完整配置 + git hash
- 维护 `best_pass1`（基于 0.2 的 greedy 全量主指标），刷新则另存 `model_best.pt`
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

d20 在 GSM8K 上的天花板有限。A800-80GB 可支撑：

- **d26 / d32**：预训练时间按比例增加，但 SFT+RL 成本增幅有限
- 或者**换底座**：放弃从零预训练，加载开源小模型（如 Qwen2.5-0.5B/1.5B-base）做 SFT+RL。代价是要适配 tokenizer 和模型结构（工程量中等），收益是预训练质量碾压自训的 10GB 数据。**如果目标是"更强的数学模型"而非"全程从零"，这是性价比最高的一条路**——Qwen2.5-1.5B 底座 + 本项目的 SFT/RL pipeline，GSM8K 上 60%+ 是合理预期。

### 4.2 扩大预训练数学密度

若坚持从零预训练：在 ClimbMix 基础上混入 OpenWebMath / FineMath 等数学网页语料（10~30%），base 模型的数学先验直接决定下游上限。

### 4.3 推理时增强（不训练也能涨分）

- **Self-consistency / majority voting**：采样 k 条取多数答案，Pass@16≈43% 的模型用 maj@16 估计能到 25~30%，作为产品功能几乎免费
- **强制工具调用**：prompt 引导模型对所有算术使用计算器（engine 已支持），消除纯计算错误

---

## 五、建议的执行路线图

| 阶段 | 内容 | 预期 GSM8K greedy Pass@1 |
|------|------|--------------------------|
| 当前 | d20, SFT + 简化 REINFORCE（评估数据被两次训练污染） | 真实值待重测 |
| **第 0 轮**（半天） | 取证与基建：checkpoint 可审计化（0.1/3.1）+ 统一 greedy 全量主指标（0.2）+ 归一化修复（0.3）+ 重跑一次受控 RL 拿干净基线 | 建立可信基线 |
| **第 1 轮**（1~2 天） | GRPO 标准化（1.2）+ 零梯度过滤（1.3）+ KL 正则（1.1）+ 早停 | 基线 +3~6pp |
| **第 2 轮**（3~5 天） | MetaMathQA 加入 SFT（2.1）+ 重新 SFT→RL | 25~35% |
| **第 3 轮**（1 周） | 拒绝采样自蒸馏迭代 ×2（2.2）+ 课程（2.3） | 30~40% |
| **第 4 轮**（可选） | 换 Qwen2.5-1.5B 底座复用全 pipeline | 55~70% |

每轮只改一类变量。v2 报告的对照实验风格值得延续，但**结论必须建立在可审计的单次训练数据上**——这是本轮取证的最大教训。

---

## 附：每条建议对应的代码改动位置

| 建议 | 文件 | 位置 |
|------|------|------|
| checkpoint 可审计化 | `scripts/train_rl.py` | `save_checkpoint` 调用处（目录名 + meta 写入 user_config/git hash） |
| 主指标统一 | `scripts/eval_report.py` / `train_rl.py` eval 块 | t=0 greedy 全量 1319 题 |
| 归一化修复 | `scripts/train_rl.py` | 训练循环的 loss 计算（per-pass → 全局 token 归一化） |
| KL 正则 / GRPO std / clipping | `scripts/train_rl.py` | `get_batch()` 的 advantage 计算 + loss 计算 |
| 零梯度过滤 | `scripts/train_rl.py` | `get_batch()` yield 之前 |
| 奖励塑形 | `tasks/gsm8k.py` | `reward()` 方法（注释已预留扩展点） |
| 新 SFT 数据 | `tasks/customjson.py` + `scripts/chat_sft.py` | 数据混合配置 |
| 自蒸馏采样脚本 | 新建 `scripts/distill_rft.py` | 复用 `engine.generate_batch` + `task.reward` |
| maj@k 投票 | `scripts/chat_eval.py` 或 `engine.py` | 评估/推理路径 |

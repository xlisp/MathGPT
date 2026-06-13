# train_rl.py 接入 GRPO 的最小改动（diff 说明）

原版关键缺陷在 `get_batch()` 末尾这一行：

```python
advantages = rewards - rewards.mean()   # 只去均值 = REINFORCE，不是 GRPO
```

以及训练循环里直接 `loss = -pg_obj`，**无 KL、无跳过退化组**。下面是最小接入。

## 1. 顶部新增 import

```python
from improved.rl_grpo_core import compute_grpo_advantages, group_is_degenerate, compute_kl_penalty
```

## 2. get_batch() 里替换 advantage 计算

```python
# 原：advantages = rewards - rewards.mean()
# 新：组内标准化（num_samples 即组大小）
advantages = compute_grpo_advantages(rewards, group_size=args.num_samples)

# 跳过"全对/全错"的退化组：无学习信号，省算力（DAPO dynamic sampling）
if group_is_degenerate(rewards):
    continue   # 该题不产出梯度，直接采下一题
```

## 3. 加载一份冻结 reference policy（KL 用，开关可控）

```python
# 紧跟 model 加载之后
ref_model = None
if args.kl_coef > 0.0:
    ref_model, _, _ = load_model(args.source, device, phase="eval",
                                 model_tag=args.model_tag, step=args.model_step)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()
```

新增两个 CLI 参数：

```python
parser.add_argument("--kl-coef", type=float, default=0.0, help=">0 开启对 ref policy 的 KL 约束")
parser.add_argument("--clip-eps", type=float, default=0.0, help=">0 开启 PPO-style ratio clip")
```

## 4. loss 段加上 KL（在现有 pg_obj 之后）

```python
logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
pg_obj = (logp * advantages.unsqueeze(-1)).sum()
num_valid = (targets >= 0).sum().clamp(min=1)
pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
loss = -pg_obj

if ref_model is not None:                      # KL 正则，防策略跑飞
    with torch.no_grad():
        logp_ref = -ref_model(inputs, targets, loss_reduction='none').view_as(inputs)
    mask = (targets >= 0).float()
    loss = loss + args.kl_coef * compute_kl_penalty(logp, logp_ref, mask)

loss.backward()
```

## 5. 其它建议的默认值调整（一行一句话理由）

| 参数 | 原值 | 建议 | 理由 |
|---|---|---|---|
| `--max-new-tokens` | 256 | 512~768 | 多步推理 256 token 常被截断，长度是数学能力关键 |
| `--num-samples` | 16 | 16~32 | 组越大，GRPO 的 std baseline 越稳，方差越低 |
| `--kl-coef` | （无） | 0.0→0.01 | 先无约束跑通，发现退化/复读再开 KL |
| `--temperature`(eval) | 1.0 | 0.0 + maj@k | 评估用贪心测能力、用多数投票测上限，两套指标 |

> 全部改动都带开关、默认值保持原行为，可单独消融、可一键回滚 —— 这就是"敏捷"。

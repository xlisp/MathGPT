"""
GRPO 核心组件 —— 把 train_rl.py 的"简化 REINFORCE"升级为真正的 GRPO。

对应面试里讲的"算法层"。三个独立、可单测的函数，方便逐个接入、逐个消融：

  compute_grpo_advantages : 组内 (reward - mean) / (std + eps)，这才是 GRPO
                            和 REINFORCE 的关键区别 —— 除以 std 做方差归一，
                            不同难度的题梯度尺度一致，训练更稳。
  group_is_degenerate     : 一道题 N 条采样全对或全错 -> std≈0 -> advantage 全 0，
                            没有学习信号。跳过这种组（DAPO 的 dynamic sampling），
                            把算力留给"有对有错"的题，单步有效样本更多。
  compute_kl_penalty      : 对 frozen reference policy 的逐 token KL，防止策略
                            跑飞 / 输出退化。原版纯 REINFORCE 无任何约束。

这些函数只依赖 torch，不依赖项目内部模块，可直接 import 进 train_rl.py。
"""

import torch


def compute_grpo_advantages(rewards, group_size, eps=1e-4, normalize_std=True):
    """
    rewards: 形如 [num_problems * group_size] 的 1D 张量，按题分组连续排列。
    返回同形状的 advantages。

    REINFORCE(原版): adv = r - mean(r)            # 仅去均值
    GRPO(本函数):    adv = (r - mean_g) / (std_g + eps)  # 组内标准化
    """
    assert rewards.dim() == 1 and rewards.numel() % group_size == 0
    r = rewards.view(-1, group_size)                      # [G, group_size]
    mean = r.mean(dim=1, keepdim=True)
    adv = r - mean
    if normalize_std:
        std = r.std(dim=1, keepdim=True)
        adv = adv / (std + eps)
    return adv.view(-1)


def group_is_degenerate(rewards_group, tol=1e-6):
    """组内奖励几乎全相同 -> 无优势信号 -> 应跳过该题的梯度。"""
    return float(rewards_group.max() - rewards_group.min()) < tol


def compute_kl_penalty(logp_policy, logp_ref, mask):
    """
    k3 估计量（无偏、低方差）的逐 token KL：kl = exp(Δ) - Δ - 1, Δ = logp_ref - logp_policy。
    mask: 1 表示参与 loss 的生成 token，0 表示 prompt/padding。
    返回标量（按有效 token 平均）。
    """
    delta = (logp_ref - logp_policy)
    kl = torch.exp(delta) - delta - 1.0
    kl = kl * mask
    denom = mask.sum().clamp(min=1)
    return kl.sum() / denom


def grpo_loss(logp_policy, advantages, mask, *,
              logp_ref=None, kl_coef=0.0, logp_old=None, clip_eps=None):
    """
    统一的 GRPO 损失。默认行为 = 组标准化优势的策略梯度；
    传 logp_ref + kl_coef 开启 KL 约束；传 logp_old + clip_eps 开启 PPO 截断。

    logp_policy: [B, T] 当前策略对每个 target token 的 log-prob
    advantages : [B]    每条序列一个标量优势（组标准化后）
    mask       : [B, T] 生成 token=1，其余=0
    """
    adv = advantages.unsqueeze(-1)                        # [B,1] 广播到 [B,T]

    if logp_old is not None and clip_eps is not None:
        # PPO-style ratio clipping
        ratio = torch.exp(logp_policy - logp_old)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        per_tok = torch.minimum(unclipped, clipped)
    else:
        per_tok = logp_policy * adv

    per_tok = per_tok * mask
    denom = mask.sum().clamp(min=1)
    pg = per_tok.sum() / denom
    loss = -pg

    if logp_ref is not None and kl_coef > 0.0:
        loss = loss + kl_coef * compute_kl_penalty(logp_policy, logp_ref, mask)

    return loss

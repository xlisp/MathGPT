调用mcp读取 /Users/xlisp/PyPro/MathGPT ，并且写docs文档，分别写：1. 和小白介绍本项目代码详细解释，LLM，RL等 2. 下一步训练如何训练出来更强的数学模型的改进建议

两份文档已写入 `docs/` 目录。我先读完了 README、`train_rl.py`、`tasks/gsm8k.py`、`full_train_a800_v2.sh` 和 v2 训练报告，所以文档内容是基于你项目的真实代码和实验数据写的，不是泛泛而谈。

**1. `docs/PROJECT_INTRO_FOR_BEGINNERS.md`** — 小白入门指南

- 用"预训练学说话 → SFT学聊天 → RL学做对题"的三段式框架讲清项目全貌
- 逐文件解释 `nanochat/`（GPT/GQA/Muon优化器/计算器工具调用状态机）、`tasks/`、`scripts/` 的作用
- 重点拆解了 `train_rl.py` 的四步循环：rollout 采样 → 0/1 奖励 → 组均值 advantage → REINFORCE 策略梯度，并用表格对比了它与完整 GRPO/PPO 缺了哪些组件（KL、clipping）
- 用 v2 报告里的真实数据（step 90 Pass@16=43% 峰值 → step 232 崩到 28.5%）解释了 mode collapse 这个最有教学价值的现象

**2. `docs/NEXT_STEPS_IMPROVEMENT.md`** — 改进建议

按 ROI 排序的四层方案，几个核心点：

- **先修 bug**：v2 报告显示 `--num-epochs=1` 实际没生效（跑了 698 步 3 epoch），这是零成本的第一步
- **算法层**：加 KL 正则（治 mode collapse 的特效药，给了 k3 估计器代码）、advantage 除以 std 升级成标准 GRPO（一行改动）、过滤全对/全错的零梯度组（DAPO 风格，你起点 Pass@1 仅 5.5%，大量算力浪费在全错组上）
- **数据层（上限最高）**：MetaMathQA 混入 SFT、用 step 480 checkpoint 做拒绝采样自蒸馏——把 Pass@16=43% 的潜力回收成 SFT 数据
- **路线图**：每轮预期收益从当前 ~15% 到第 3 轮 30-40%；如果不坚持从零预训练，换 Qwen2.5-1.5B 底座复用你的 pipeline 是性价比最高的路（55-70% 可期）
- 附了每条建议对应的具体代码改动位置

一个观察顺带提一下：v2 报告本身写得相当好（对照实验、口径说明、案例分析都有），建议后续迭代延续这个风格，每轮只改一类变量。


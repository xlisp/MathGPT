# MathGPT

自包含的数学推理助手，从零预训练 → SFT 对话微调 → REINFORCE 强化学习（简化版 GRPO）在 GSM8K 数学题上训练。
nanochat 核心代码已内嵌到本项目，无需外部依赖。

## 运行环境

| 组件 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.3.1+cu118 |
| GPU | NVIDIA GeForce GTX 1080（8GB VRAM，SM 6.1） |
| 精度 | float32（GTX 1080 不支持 bfloat16） |

> **兼容性说明**：原始 nanochat 需要 PyTorch ≥ 2.5，本项目已直接在源码中修复：
> - `F.rms_norm`（PyTorch 2.4+ 才有）→ 手动实现方差归一化
> - `torch.compile`（Python 3.12 + PyTorch 2.3 不支持 Dynamo）→ 运行时检测，替换为恒等操作
> - `F.scaled_dot_product_attention(enable_gqa=...)`（PyTorch 2.5+ 才有）→ 用 `repeat_interleave` 手动展开 KV 头
> - 优化器中 0-D CPU 张量无法 `lerp_` 到 CUDA 张量 → 添加 `.to(dev)`

## 项目结构

```
MathGPT/
├── nanochat/                    ← 内嵌的核心框架（已适配本机环境）
│   ├── gpt.py                   ← GPT 模型（RMSNorm / SDPA / GQA）
│   ├── optim.py                 ← MuonAdamW 优化器
│   ├── flash_attention.py       ← 注意力（PyTorch SDPA 回退）
│   ├── engine.py                ← KV Cache 推理引擎 + 工具调用
│   ├── tokenizer.py             ← BPE 分词器
│   ├── checkpoint_manager.py    ← 检查点保存/加载
│   ├── common.py                ← 设备/精度配置
│   ├── dataloader.py            ← 预训练数据加载器
│   └── dataset.py               ← ClimbMix 数据集下载
│
├── tasks/                       ← 评估与 SFT 数据集
│   ├── gsm8k.py                 ← GSM8K 数学题（RL 奖励 + SFT 数据）
│   ├── smoltalk.py              ← SmolTalk 通用对话（SFT）
│   ├── mmlu.py                  ← MMLU 学科知识（SFT）
│   ├── arc.py                   ← ARC 推理题（评估）
│   ├── humaneval.py             ← HumanEval 代码题（评估）
│   ├── spellingbee.py           ← 拼写任务（SFT + 评估）
│   ├── customjson.py            ← 自定义 JSONL 数据集
│   └── common.py                ← 任务混合工具
│
├── scripts/                     ← 训练与推理脚本
│   ├── run.py                   ← 启动器（设置 PYTHONPATH 后运行模块）
│   ├── full_train.sh            ← 一键完整训练脚本（5 个阶段）
│   ├── base_train.py            ← Base 模型预训练
│   ├── tok_train.py             ← BPE Tokenizer 训练
│   ├── chat_sft.py              ← 监督微调（SFT）
│   ├── train_rl.py              ← MathGPT RL 强化学习（REINFORCE / GRPO）
│   ├── chat_cli.py              ← 命令行聊天
│   ├── chat_web.py              ← Web 聊天服务（FastAPI + SSE 流式）
│   ├── chat_eval.py             ← 评估工具（categorical / generative）
│   ├── base_eval.py             ← Base 模型评估
│   └── tok_eval.py              ← Tokenizer 评估
│
├── math_gpt/                    ← Web UI 资源
│   ├── ui.html                  ← 聊天界面（KaTeX 公式 + 流式输出）
│   └── compat.py                ← 历史兼容模块（已不再使用）
│
├── docs/                        ← 技术文档
│   └── RL_SFT_GRPO_INTRO.md    ← SFT / GRPO / RL 历史介绍
│
├── TRAINING_LOG.md              ← 训练日志与实验结果
├── nanochat_readme.md           ← 原始 nanochat 文档（参考）
└── runs/                        ← 检查点与数据（已 gitignore）
    ├── base_data_climbmix/      ← 预训练数据分片
    ├── tokenizer/               ← 训练好的 BPE 分词器
    ├── base_checkpoints/        ← Base 模型检查点
    ├── chatsft_checkpoints/     ← SFT 模型检查点
    └── chatrl_checkpoints/      ← RL 模型检查点（MathGPT）
```

## 训练流程

所有脚本从项目根目录运行，`scripts/run.py` 启动器会自动设置 `PYTHONPATH` 和 `NANOCHAT_BASE_DIR`。

### 一键训练

```bash
cd /home/xlisp/PyPro/MathGPT
bash scripts/full_train.sh
```

### 分步训练

```bash
# 1. 下载训练数据集（约 800MB，8 个分片）
python3 -m scripts.run nanochat.dataset -n 8

# 2. 训练 BPE 分词器（vocab=32768，约 70 秒）
python3 -m scripts.run scripts.tok_train --max-chars=2000000000

# 3. Base 模型预训练（GTX 1080 配置，约 25 分钟）
python3 -m scripts.run scripts.base_train \
    --depth=6 --head-dim=64 --window-pattern=L \
    --max-seq-len=256 --device-batch-size=16 --total-batch-size=8192 \
    --num-iterations=3000 --eval-every=200 --eval-tokens=131072 \
    --core-metric-every=-1 --sample-every=500 --run=dummy

# 4. SFT 微调（对话格式 + 数学工具调用，约 4~8 分钟）
python3 -m scripts.run scripts.chat_sft \
    --max-seq-len=256 --device-batch-size=16 --total-batch-size=8192 \
    --num-iterations=1000 --eval-every=200 --eval-tokens=131072 \
    --chatcore-every=-1 --run=dummy

# 5. MathGPT RL 强化学习（GSM8K 数学题，1 epoch ≈ 934 步）
python3 -m scripts.train_rl \
    --num-epochs=1 --device-batch-size=4 \
    --examples-per-step=8 --num-samples=8 \
    --max-new-tokens=256 --run=dummy
```

## 与模型对话

**命令行模式：**
```bash
python3 -m scripts.chat_cli                        # 使用 RL 训练的数学模型
python3 -m scripts.chat_cli --source sft           # 使用 SFT 模型
python3 -m scripts.chat_cli --prompt "15% of 80 is?"
```

**Web UI 模式（支持 LaTeX 公式渲染）：**
```bash
python3 -m scripts.chat_web               # http://localhost:8000
python3 -m scripts.chat_web --port 8080
python3 -m scripts.chat_web --source sft  # 使用 SFT 模型
```

打开 http://localhost:8000，界面特性：
- **KaTeX 数学公式渲染**：`$...$`（行内）和 `$$...$$`（块级）
- **流式输出**：逐 token 显示答案
- **斜杠命令**：`/temperature 0.8`、`/topk 30`、`/clear`、`/help`
- **编辑 / 重新生成**：点击任意消息即可编辑或重新生成
- **示例题目**：点击预设例题快速提问

## RL 训练原理

对每个训练步骤：

1. **采样 Rollouts**：对每道 GSM8K 数学题生成 N 条候选答案
2. **计算奖励**：将最终答案与标准答案对比
   - 答对 → reward = 1.0
   - 答错 → reward = 0.0
3. **计算优势**：`advantage = reward − mean(reward)`
4. **策略梯度更新**：最大化高优势答案的生成概率
   ```
   loss = −(logp × advantage)
   ```

这是一个干净的 REINFORCE 实现，无 KL 正则化，无 PPO clipping，采用 DAPO 风格的 token 级归一化。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | `rl` | 加载的检查点类型：`sft` 或 `rl` |
| `--model-tag` | 自动 | 模型标签（如 `math_d6`） |
| `--temperature` | 0.6 | 采样温度 |
| `--top-k` | 50 | Top-k 采样 |
| `--max-tokens` | 512 | 最大生成长度 |

检查点目录由 `NANOCHAT_BASE_DIR` 环境变量控制（MathGPT 脚本默认设为 `./runs/`）。

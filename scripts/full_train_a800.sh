#!/bin/bash
# MathGPT 完整训练流程（A800 高性能版）
# 适配 A800-SXM4-80GB (SM 8.0, bfloat16, 80GB VRAM)
# Python 3.11 + PyTorch 2.7.1+cu128
#
# 对比 GTX 1080 版的关键提升：
#   数据量:  8 shards (800MB)   → 100 shards (~10GB)
#   模型:    depth=6 (73.5M)    → depth=20 (~700M)
#   上下文:  256 tokens         → 2048 tokens
#   预训练:  3000步 × 8K batch  → 5000步 × 524K batch
#   SFT:     500步              → 3000步
#   RL:      1 epoch            → 3 epochs
#
# 用法（从 MathGPT 根目录执行）:
#   bash scripts/full_train_a800.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export NANOCHAT_BASE_DIR="$PROJECT_ROOT/runs"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

RUN="python3 -m scripts.run"

GREEN='\033[0;32m'; NC='\033[0m'
step() { echo -e "\n${GREEN}==============================\n$1\n==============================${NC}\n"; }

cd "$PROJECT_ROOT"

# ──────────────────────────────────────────────────────────────
step "1/5  下载训练数据集（100 分片 ≈ 10GB）"
# 上次仅 8 shards (~800MB, ~24M tokens)，远远不够
# 100 shards 约提供 ~5B tokens 的文本数据
# $RUN nanochat.dataset -n 100 ## --- 已经下载好：/mnt/openclaw/MathGPT/runs/base_data_climbmix/

# ──────────────────────────────────────────────────────────────
step "2/5  训练 Tokenizer (BPE, vocab=32768)"

if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "下载 identity_conversations.jsonl ..."
    curl -fL -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

$RUN scripts.tok_train --max-chars=2000000000

# ──────────────────────────────────────────────────────────────
step "3/5  Base 模型预训练（A800 配置，depth=20，~700M 参数，bf16）"
# 上次: depth=6, seq=256, 3000步 × 8K = 24M tokens → BPB 1.356
# 本次: depth=20, seq=2048, 5000步 × 524K = 2.6B tokens（提升 ~108x）
$RUN scripts.base_train \
    --depth=20 \
    --head-dim=128 \
    --window-pattern=SSSL \
    --max-seq-len=2048 \
    --device-batch-size=32 \
    --total-batch-size=524288 \
    --num-iterations=5000 \
    --eval-every=500 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=1000 \
    --save-every=1000 \
    --run=dummy

# ──────────────────────────────────────────────────────────────
step "4/5  SFT 微调（对话格式 + 数学工具调用，3000 步）"
# 上次: 500步，对话格式学习不充分
# 本次: 3000步，GSM8K 8 epochs 充分学习数学推理模式
$RUN scripts.chat_sft \
    --max-seq-len=2048 \
    --device-batch-size=32 \
    --total-batch-size=524288 \
    --num-iterations=3000 \
    --eval-every=500 \
    --eval-tokens=524288 \
    --gsm8k-epochs=8 \
    --mmlu-epochs=5 \
    --run=dummy

# ──────────────────────────────────────────────────────────────
step "5/5  MathGPT RL 强化学习（GSM8K 数学题，3 epochs）"
# 上次: 1 epoch, 8 samples, max_new_tokens=256 → Pass@1 最高 2.75%
# 本次: 3 epochs, 16 samples/题, max_new_tokens=512
#   更多 epoch → 更多优化步数
#   更多 samples → 方差更低的梯度估计
#   更长生成 → 完整推理链不被截断
python3 -m scripts.train_rl \
    --num-epochs=3 \
    --device-batch-size=16 \
    --examples-per-step=32 \
    --num-samples=16 \
    --max-new-tokens=512 \
    --eval-every=60 \
    --eval-examples=400 \
    --save-every=60 \
    --run=dummy

echo -e "\n${GREEN}====== 训练完成！ ======${NC}"
echo "  命令行对话: python3 -m scripts.chat_cli"
echo "  Web UI 对话: python3 -m scripts.chat_web  →  http://localhost:8000"

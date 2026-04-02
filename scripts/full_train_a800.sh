#!/bin/bash
# MathGPT 完整训练流程
# 适配 A800-SXM4-80GB (SM 8.0, bfloat16, 80GB VRAM)
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
step "1/5  下载训练数据集（8 分片 ≈ 800MB）"
$RUN nanochat.dataset -n 8

# ──────────────────────────────────────────────────────────────
step "2/5  训练 Tokenizer (BPE, vocab=32768)"

if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "下载 identity_conversations.jsonl ..."
    curl -fL -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

$RUN scripts.tok_train --max-chars=2000000000

# ──────────────────────────────────────────────────────────────
step "3/5  Base 模型预训练（A800 配置，depth=20，bf16）"
$RUN scripts.base_train \
    --depth=20 \
    --head-dim=128 \
    --window-pattern=SSSL \
    --max-seq-len=2048 \
    --device-batch-size=32 \
    --total-batch-size=524288 \
    --num-iterations=3000 \
    --eval-every=200 \
    --eval-tokens=131072 \
    --core-metric-every=-1 \
    --sample-every=500 \
    --run=dummy

# ──────────────────────────────────────────────────────────────
step "4/5  SFT 微调（对话格式 + 数学工具调用）"
$RUN scripts.chat_sft \
    --max-seq-len=2048 \
    --device-batch-size=32 \
    --total-batch-size=524288 \
    --num-iterations=1000 \
    --eval-every=200 \
    --eval-tokens=131072 \
    --run=dummy

# ──────────────────────────────────────────────────────────────
step "5/5  MathGPT RL 强化学习（GSM8K 数学题）"
python3 -m scripts.train_rl \
    --num-epochs=1 \
    --device-batch-size=16 \
    --examples-per-step=32 \
    --num-samples=16 \
    --max-new-tokens=512 \
    --run=dummy

echo -e "\n${GREEN}====== 训练完成！ ======${NC}"
echo "  命令行对话: python3 -m scripts.chat_cli"
echo "  Web UI 对话: python3 -m scripts.chat_web  →  http://localhost:8000"

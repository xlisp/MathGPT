#!/bin/bash
# MathGPT 完整训练流程
# 适配 GTX 1080 (8GB VRAM, SM 6.1, float32, 无 bfloat16)
#
# 用法:
#   cd /home/xlisp/PyPro/MathGPT
#   bash scripts/full_train.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NANOCHAT_DIR="$(cd "$PROJECT_ROOT/../nanochat" && pwd)"

export NANOCHAT_BASE_DIR="$PROJECT_ROOT/runs"
export PYTHONPATH="$NANOCHAT_DIR:$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$NANOCHAT_BASE_DIR"

# 统一用 run.py 启动（自动打补丁）
RUN="python3 -m scripts.run"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
step() { echo -e "\n${GREEN}==============================\n$1\n==============================${NC}\n"; }

cd "$PROJECT_ROOT"

# ──────────────────────────────────────────────────────────────
step "1/5  下载训练数据集（8 分片 ≈ 800MB）"
$RUN nanochat.dataset -n 8

# ──────────────────────────────────────────────────────────────
step "2/5  训练 Tokenizer (BPE, vocab=32768)"

# 下载 identity 对话数据（SFT 用）
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "下载 identity_conversations.jsonl ..."
    curl -fL -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

$RUN scripts.tok_train --max-chars=2000000000

# ──────────────────────────────────────────────────────────────
step "3/5  Base 模型预训练（GTX 1080 小配置，depth=6）"
# 参数说明：
#   depth=6       : 6 层 Transformer (~50M 参数)
#   max-seq-len=256   : 序列长度，适配 8GB VRAM
#   device-batch-size=16
#   num-iterations=3000 : 约 30~60 分钟
$RUN scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=256 \
    --device-batch-size=16 \
    --total-batch-size=8192 \
    --num-iterations=3000 \
    --eval-every=200 \
    --eval-tokens=131072 \
    --core-metric-every=-1 \
    --sample-every=500 \
    --run=dummy

# ──────────────────────────────────────────────────────────────
step "4/5  SFT 微调（对话格式 + 数学工具调用）"
$RUN scripts.chat_sft \
    --max-seq-len=256 \
    --device-batch-size=16 \
    --total-batch-size=8192 \
    --num-iterations=1000 \
    --eval-every=200 \
    --eval-tokens=131072 \
    --run=dummy

# ──────────────────────────────────────────────────────────────
step "5/5  MathGPT RL 强化学习（GSM8K 数学题）"
python3 -m scripts.train_rl \
    --num-epochs=1 \
    --device-batch-size=4 \
    --examples-per-step=8 \
    --num-samples=8 \
    --max-new-tokens=256 \
    --run=dummy

echo -e "\n${GREEN}====== 训练完成！ ======${NC}"
echo ""
echo "  对话（命令行）: python3 -m scripts.chat_cli"
echo "  对话（Web UI）: python3 -m scripts.chat_web  → http://localhost:8000"

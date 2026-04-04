#!/bin/bash
# MathGPT A800 训练 v2 — 基于第一轮经验的优化版
#
# 核心改动（对比 full_train_a800.sh v1）:
#   1. 跳过 Base 预训练 — 直接复用 v1 的 base checkpoint (d20, step 5000, BPB 0.732)
#   2. SFT: total-batch-size 从 524K 降到 65K，确保 num-iterations=3000 生效
#          （v1 的 bug：数据 1 epoch 跑完就停了，只跑了 375 步）
#   3. SFT: gsm8k-epochs 从 8 提升到 16，数学数据占比翻倍
#   4. RL:  num-epochs 从 3 降到 1，减少过拟合
#   5. RL:  num-samples 从 16 增到 32，更多采样 → 更稳定的梯度
#   6. RL:  max-new-tokens 从 512 增到 768，避免回答被截断
#   7. RL:  eval-every 从 60 降到 30，更早发现最优点
#   8. RL:  init-lr-frac 从 0.05 降到 0.02，减小初期震荡
#
# v1 问题诊断:
#   - SFT 仅 375 步（数据用完自动停止），模型没学会完整对话
#   - RL 最优点在 step 120 (17%)，之后 83% 训练量全部浪费在过拟合
#   - 最终模型简单算术题都回答不了（截断、计算错误）
#
# 预期效果:
#   - SFT 跑满 3000 步 → GSM8K Pass@1 从 12.5% 提升到 20-25%
#   - RL 更保守训练 + 更频繁 eval → Pass@1 目标 30-40%
#
# 前置条件:
#   - Base checkpoint 已存在: runs/base_checkpoints/d20/model_005000.pt
#   - 已修复 chat_sft.py 中 num_iterations 被数据耗尽覆盖的 bug
#   - 离线数据集已准备: data/hf_datasets/
#
# 用法:
#   cd /mnt/openclaw/MathGPT && bash scripts/full_train_a800_v2.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export NANOCHAT_BASE_DIR="$PROJECT_ROOT/runs"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

RUN="python3 -m scripts.run"

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'
step()  { echo -e "\n${GREEN}==============================\n$1\n==============================${NC}\n"; }
warn()  { echo -e "${YELLOW}[WARN] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; exit 1; }

cd "$PROJECT_ROOT"

# ──────────────────────────────────────────────────────────────
# 前置检查
# ──────────────────────────────────────────────────────────────
step "0/3  前置检查"

# 确认 base checkpoint 存在
BASE_CKPT="$NANOCHAT_BASE_DIR/base_checkpoints/d20/model_005000.pt"
if [ ! -f "$BASE_CKPT" ]; then
    error "Base checkpoint 不存在: $BASE_CKPT\n  请先运行 full_train_a800.sh 的 Base 预训练步骤"
fi
echo "Base checkpoint: $BASE_CKPT"

# 确认离线数据集存在
OFFLINE_DIR="$PROJECT_ROOT/data/hf_datasets"
if [ ! -d "$OFFLINE_DIR" ]; then
    warn "离线数据集目录不存在: $OFFLINE_DIR，将尝试在线下载"
    OFFLINE_FLAG=""
else
    echo "离线数据集: $OFFLINE_DIR"
    OFFLINE_FLAG="--offline=$OFFLINE_DIR"
fi

# 确认 tokenizer 存在 (RustBPE 用 tokenizer.pkl, HuggingFace 用 tokenizer.json)
TOK_DIR="$NANOCHAT_BASE_DIR/tokenizer"
if [ ! -d "$TOK_DIR" ]; then
    error "Tokenizer 目录不存在: $TOK_DIR\n  请先运行 full_train_a800.sh 的 Tokenizer 训练步骤"
fi
echo "Tokenizer: $TOK_DIR"

echo -e "\n${GREEN}前置检查通过${NC}"

# ──────────────────────────────────────────────────────────────
step "1/3  SFT 微调 v2（3000 步，降低 batch size，增加数学数据）"
# v1: total-batch-size=524288, gsm8k-epochs=8 → 只跑了 375 步
# v2: total-batch-size=65536,  gsm8k-epochs=16 → 跑满 3000 步
#
# 关键改动:
#   - total-batch-size: 524288 → 65536 (8x 降低)
#     → grad_accum = 65536 / (32 * 2048) = 1 步（无梯度累积，训练更快）
#   - gsm8k-epochs: 8 → 16（数学数据占比从 ~5% 提升到 ~9%）
#   - num-iterations=3000 现在真正生效（已修复 chat_sft.py 中的 bug）
#   - eval-every=200, chatcore-every=500（更频繁监控）
#
# 数据混合: SmolTalk 460K + MMLU 500K + GSM8K 120K + Spelling 280K + Identity 2K ≈ 1.36M rows
# 3000 步 × 65K batch ÷ ~2048 tok/row ≈ 每行约被看 ~96 次（充分迭代）

$RUN scripts.chat_sft \
    --max-seq-len=2048 \
    --device-batch-size=32 \
    --total-batch-size=65536 \
    --num-iterations=3000 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --chatcore-every=500 \
    --gsm8k-epochs=16 \
    --mmlu-epochs=5 \
    $OFFLINE_FLAG \
    --run=dummy

# ──────────────────────────────────────────────────────────────
step "1.5/3  SFT 效果验证"
# 在进入 RL 之前先验证 SFT 模型的对话质量
# 如果 SFT 模型连完整回答都给不出，RL 也无法挽救

echo "运行 SFT 模型评估 (GSM8K quick)..."
NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" python3 -m scripts.eval_report \
    --source sft \
    --quick \
    $OFFLINE_FLAG \
    --output=EVAL_REPORT_SFT_V2.md

echo ""
echo "SFT 评估报告已保存到: EVAL_REPORT_SFT_V2.md"
echo "请检查 SFT 模型质量后继续 RL 训练"
echo ""

# ──────────────────────────────────────────────────────────────
step "2/3  RL 强化学习 v2（更保守，更频繁 eval）"
# v1: 3 epochs, 16 samples, eval-every=60, init-lr-frac=0.05
#     → 699 步，最优在 step 120，后 83% 过拟合浪费
# v2: 1 epoch, 32 samples, eval-every=30, init-lr-frac=0.02
#     → 更短训练 + 更多采样 + 更频繁 eval + 更小学习率
#
# 关键改动:
#   - num-epochs: 3 → 1（减少过拟合风险，v1 证明 1 epoch 内就能达到峰值）
#   - num-samples: 16 → 32（更多采样 → 更多有效梯度信号）
#   - max-new-tokens: 512 → 768（避免回答截断导致的质量问题）
#   - eval-every: 60 → 30（更早发现最优点，v1 可能错过了 step 90-120 间的真正最优）
#   - save-every: 60 → 30（配合更频繁的 eval）
#   - init-lr-frac: 0.05 → 0.02（更小初始学习率，减少初期震荡）
#   - device-batch-size: 16 → 32（配合 num-samples=32）
#
# 预计步数: ~7500 train examples / 32 per step ≈ 233 步
# 训练时间预估: ~2-3 小时

python3 -m scripts.train_rl \
    --source=sft \
    --num-epochs=1 \
    --device-batch-size=32 \
    --examples-per-step=32 \
    --num-samples=32 \
    --max-new-tokens=768 \
    --temperature=1.0 \
    --top-k=50 \
    --init-lr-frac=0.02 \
    --eval-every=30 \
    --eval-examples=400 \
    --save-every=30 \
    $OFFLINE_FLAG \
    --run=dummy

# ──────────────────────────────────────────────────────────────
step "3/3  最终评估与报告"

echo "运行 RL 模型完整评估..."
NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" python3 -m scripts.eval_report \
    --source rl \
    --all-steps \
    $OFFLINE_FLAG \
    --output=EVAL_REPORT_RL_V2.md

echo ""
echo -e "${GREEN}====== v2 训练完成！ ======${NC}"
echo ""
echo "评估报告:"
echo "  SFT: EVAL_REPORT_SFT_V2.md"
echo "  RL:  EVAL_REPORT_RL_V2.md"
echo ""
echo "使用最优 checkpoint 对话:"
echo "  命令行: NANOCHAT_BASE_DIR=./runs python3 -m scripts.chat_cli --step <best_step>"
echo "  Web UI: NANOCHAT_BASE_DIR=./runs python3 -m scripts.chat_web --step <best_step>"
echo ""
echo "查看 EVAL_REPORT_RL_V2.md 中的 'Best Pass@1' 找到最优 step"

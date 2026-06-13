"""
RFT / 拒绝采样自举 (Rejection-sampling Fine-Tuning, STaR 思路)
================================================================
这是"敏捷改进"里【风险最低、收益最大】的一步，应该最先做：
不动 RL 框架，只用现有 SFT 模型 + 现有数据，就能拿到一大块提升。

流程：
  1. 用当前最好的 checkpoint（SFT 或上一轮 RL）对每道训练题采样 K 条 CoT；
  2. 只保留【最终答案正确】的链路（拒绝采样）；
  3. 每题去重、最多保留 N 条，写成 SFT 训练用的 JSONL；
  4. 把这份"自蒸馏"数据混回 chat_sft 再训一轮 -> 模型学会自己生成的正确解法。
  迭代 2~3 轮，pass@1 通常稳定上升，且完全可控、易回滚。

依赖项目内现有组件：Engine / load_model / GSM8K（与 train_rl.py 同一套）。
答案判定复用改进版 gsm8k_reward.extract_answer（鲁棒抽取）。

用法:
  python -m scripts.rft_bootstrap --source sft --k 8 --keep 2 \
      --out runs/rft_data/gsm8k_round1.jsonl
"""

import os
import sys
import json
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.join(_PROJECT_ROOT, "runs"))

import torch

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# 复用改进版鲁棒抽取；若未接入则回退到任务自带 evaluate
try:
    from improved.gsm8k_reward import extract_answer as robust_extract
except Exception:
    robust_extract = None


def parse_args():
    p = argparse.ArgumentParser(description="RFT / rejection-sampling bootstrap for GSM8K")
    p.add_argument("--source", type=str, default="sft", choices=["sft", "rl"])
    p.add_argument("--model-tag", type=str, default=None)
    p.add_argument("--model-step", type=int, default=None)
    p.add_argument("--k", type=int, default=8, help="每题采样条数")
    p.add_argument("--keep", type=int, default=2, help="每题最多保留的正确解条数")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-examples", type=int, default=None, help="调试用，限制题数")
    p.add_argument("--device-batch-size", type=int, default=8)
    p.add_argument("--out", type=str, required=True, help="输出 JSONL 路径")
    p.add_argument("--offline", type=str, default=None)
    return p.parse_args()


def is_correct(task, conversation, response):
    """优先用鲁棒抽取判定，回退到任务自带 evaluate。"""
    if robust_extract is not None:
        ref_text = conversation["messages"][-1]["content"][-1]["text"]
        ref, pred = robust_extract(ref_text), robust_extract(response)
        return ref is not None and pred is not None and ref == pred
    return bool(task.evaluate(conversation, response))


def main():
    args = parse_args()
    device_type = autodetect_device_type()
    ddp, rank, local_rank, world_size, device = compute_init(device_type)
    master = rank == 0

    model, tokenizer, meta = load_model(
        args.source, device, phase="eval",
        model_tag=args.model_tag, step=args.model_step,
    )
    engine = Engine(model, tokenizer)
    model.eval()

    task = GSM8K(subset="main", split="train", offline_dir=args.offline)
    n = len(task) if args.max_examples is None else min(args.max_examples, len(task))
    print0(f"[RFT] bootstrap over {n} problems | k={args.k} keep={args.keep}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept_total, seen_correct = 0, 0

    # 仅 master 写文件；多卡可各写分片后合并，这里保持简单单文件
    fout = open(args.out, "w", encoding="utf-8") if master else None

    for idx in range(rank, n, world_size):
        conversation = task[idx]
        prompt_tokens = tokenizer.render_for_completion(conversation)
        prefix_len = len(prompt_tokens)

        correct_texts, n_done = [], 0
        while n_done < args.k:
            bs = min(args.device_batch_size, args.k - n_done)
            seqs, _ = engine.generate_batch(
                prompt_tokens, num_samples=bs,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
                seed=(idx * 100003 + n_done) & 0x7FFFFFFF,
            )
            for s in seqs:
                text = tokenizer.decode(s[prefix_len:])
                if is_correct(task, conversation, text):
                    seen_correct += 1
                    if text not in correct_texts:        # 去重
                        correct_texts.append(text)
            n_done += bs

        # 每题最多保留 keep 条，构造 SFT 样本
        for text in correct_texts[: args.keep]:
            if fout is not None:
                record = {
                    "messages": [
                        {"role": "user", "content": conversation["messages"][0]["content"]},
                        {"role": "assistant", "content": text},
                    ],
                    "source": "rft", "problem_idx": idx,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept_total += 1

        if master and idx % 50 == 0:
            print0(f"[RFT] idx={idx} kept_total={kept_total} correct_seen={seen_correct}")

    if fout is not None:
        fout.close()
        print0(f"[RFT] done. wrote {kept_total} examples -> {args.out}")
        print0("[RFT] 下一步：把该 JSONL 经 tasks/customjson 混入 chat_sft 再训一轮。")

    compute_cleanup()


if __name__ == "__main__":
    main()

"""
MathGPT: Comprehensive evaluation and report generation.

Evaluates RL checkpoints across multiple steps and generates a markdown report
with pass@k metrics, benchmark scores, and sample outputs.

Usage:
  # Evaluate latest checkpoint only
  NANOCHAT_BASE_DIR=./runs python3 -m scripts.eval_report

  # Evaluate specific steps
  NANOCHAT_BASE_DIR=./runs python3 -m scripts.eval_report --steps 0 60 120 360 698

  # Evaluate all saved checkpoints
  NANOCHAT_BASE_DIR=./runs python3 -m scripts.eval_report --all-steps

  # Quick mode: fewer examples, fewer samples
  NANOCHAT_BASE_DIR=./runs python3 -m scripts.eval_report --quick

  # Also run categorical benchmarks (ARC, MMLU)
  NANOCHAT_BASE_DIR=./runs python3 -m scripts.eval_report --benchmarks
"""

import os
import sys
import glob
import json
import argparse
import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.join(_PROJECT_ROOT, "runs"))

import torch

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_model_from_dir, find_last_step
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# ---------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser(description="MathGPT evaluation & report generation")
parser.add_argument("--source", type=str, default="rl", choices=["sft", "rl"], help="checkpoint source")
parser.add_argument("--model-tag", type=str, default=None, help="model tag (e.g. math_d20)")
parser.add_argument("--steps", type=int, nargs="+", default=None, help="specific steps to evaluate")
parser.add_argument("--all-steps", action="store_true", help="evaluate every saved checkpoint")
parser.add_argument("--num-samples", type=int, default=16, help="samples per problem for pass@k")
parser.add_argument("--eval-examples", type=int, default=400, help="number of GSM8K test problems")
parser.add_argument("--max-new-tokens", type=int, default=512, help="max tokens to generate")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling")
parser.add_argument("--sample-problems", type=int, default=10, help="number of problems to show detailed outputs for")
parser.add_argument("--quick", action="store_true", help="quick mode: 100 examples, 4 samples")
parser.add_argument("--benchmarks", action="store_true", help="also run ARC/MMLU/HumanEval/SpellingBee benchmarks")
parser.add_argument("--offline", type=str, default=None, help="path to local HF datasets dir")
parser.add_argument("--output", type=str, default="EVAL_REPORT.md", help="output report file")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
args = parser.parse_args()

if args.quick:
    args.eval_examples = 100
    args.num_samples = 4
    args.sample_problems = 5

# ---------------------------------------------------------------------------
# Init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0


def print_master(msg):
    if master_process:
        print(msg)


# ---------------------------------------------------------------------------
# Discover checkpoints
def discover_steps(checkpoints_dir, model_tag):
    """Find all available checkpoint steps for a model tag."""
    ckpt_dir = os.path.join(checkpoints_dir, model_tag)
    files = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
    steps = sorted(int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in files)
    return steps


model_dir_name = {"sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}[args.source]
base_dir = get_base_dir()
checkpoints_dir = os.path.join(base_dir, model_dir_name)

# Figure out model_tag
if args.model_tag:
    model_tag = args.model_tag
else:
    from nanochat.checkpoint_manager import find_largest_model
    model_tag = find_largest_model(checkpoints_dir)
    print_master(f"Auto-detected model tag: {model_tag}")

available_steps = discover_steps(checkpoints_dir, model_tag)
print_master(f"Available checkpoint steps: {available_steps}")

if args.all_steps:
    eval_steps = available_steps
elif args.steps:
    eval_steps = [s for s in args.steps if s in available_steps]
    missing = [s for s in args.steps if s not in available_steps]
    if missing:
        print_master(f"Warning: steps {missing} not found in checkpoints, skipping")
else:
    # Default: just the latest
    eval_steps = [available_steps[-1]]

print_master(f"Will evaluate steps: {eval_steps}")

# ---------------------------------------------------------------------------
# Load dataset
val_task = GSM8K(subset="main", split="test", offline_dir=args.offline)
print_master(f"GSM8K test set: {len(val_task)} problems, evaluating {args.eval_examples}")

# ---------------------------------------------------------------------------
# Evaluation functions

@torch.no_grad()
def evaluate_gsm8k_passk(model, tokenizer, engine, task, num_samples, max_examples, max_tokens, temperature, top_k):
    """Evaluate pass@k on GSM8K, returning per-problem results."""
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    num_examples = min(max_examples, len(task))
    results = []

    for idx in range(ddp_rank, num_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        generated_seqs, _ = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        completions = [tokenizer.decode(s[prefix_length:]) for s in generated_seqs]
        outcomes = [task.evaluate(conversation, c) for c in completions]

        # Extract question text
        question = conversation["messages"][0]["content"]
        # Extract ground truth
        assistant_msg = conversation["messages"][-1]
        gt_parts = assistant_msg["content"]
        gt_text = "".join(p["text"] for p in gt_parts if p["type"] == "text")

        results.append({
            "idx": idx,
            "question": question,
            "ground_truth": gt_text.strip(),
            "completions": completions,
            "outcomes": outcomes,  # list of 0/1
            "num_correct": sum(outcomes),
        })

        done = len(results)
        correct_any = sum(1 for r in results if any(r["outcomes"]))
        print(f"\r\033[KRank {ddp_rank} | {done}/{num_examples // ddp_world_size} | "
              f"pass@1_approx={correct_any}/{done}", end="", flush=True)

    print()
    return results


def compute_passk(results, max_k):
    """From per-problem results, compute pass@k for k=1..max_k."""
    passk = {}
    for k in range(1, max_k + 1):
        passed = sum(1 for r in results if any(r["outcomes"][:k]) )
        passk[k] = passed / len(results) if results else 0.0
    return passk


def aggregate_results_ddp(results, device):
    """Gather results from all DDP ranks."""
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    if not ddp:
        return results

    # Serialize results to JSON for gathering
    import torch.distributed as dist
    local_data = json.dumps(results).encode("utf-8")
    local_tensor = torch.tensor(list(local_data), dtype=torch.uint8, device=device)
    local_size = torch.tensor([len(local_data)], dtype=torch.long, device=device)

    # Gather sizes
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(ddp_world_size)]
    dist.all_gather(all_sizes, local_size)

    # Gather data
    max_size = max(s.item() for s in all_sizes)
    padded = torch.zeros(max_size, dtype=torch.uint8, device=device)
    padded[:len(local_data)] = local_tensor
    all_padded = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(ddp_world_size)]
    dist.all_gather(all_padded, padded)

    # Decode on all ranks
    all_results = []
    for i, (tensor, size) in enumerate(zip(all_padded, all_sizes)):
        data = bytes(tensor[:size.item()].cpu().tolist())
        all_results.extend(json.loads(data.decode("utf-8")))

    # Sort by original index
    all_results.sort(key=lambda r: r["idx"])
    return all_results


# ---------------------------------------------------------------------------
# Run benchmarks if requested
def run_benchmarks(model, tokenizer, engine, offline_dir):
    """Run categorical + generative benchmarks, return dict of {task: accuracy}."""
    from scripts.chat_eval import run_chat_eval
    tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
    results = {}
    for task_name in tasks:
        try:
            acc = run_chat_eval(
                task_name, model, tokenizer, engine,
                batch_size=8, num_samples=1, max_new_tokens=512,
                temperature=0.0, top_k=50, offline_dir=offline_dir,
            )
            results[task_name] = acc
            print_master(f"  {task_name}: {100*acc:.2f}%")
        except Exception as e:
            print_master(f"  {task_name}: FAILED ({e})")
            results[task_name] = None
    return results


# ---------------------------------------------------------------------------
# Main evaluation loop

all_step_results = {}

for step in eval_steps:
    print_master(f"\n{'='*60}")
    print_master(f"Evaluating step {step}")
    print_master(f"{'='*60}")

    # Load model for this step
    ckpt_dir = os.path.join(checkpoints_dir, model_tag)
    model, tokenizer, meta = load_model_from_dir(
        checkpoints_dir, device, phase="eval", model_tag=model_tag, step=step
    )
    engine = Engine(model, tokenizer)

    # 1. GSM8K pass@k evaluation
    print_master(f"\n--- GSM8K Pass@k (n={args.num_samples}, {args.eval_examples} problems) ---")
    raw_results = evaluate_gsm8k_passk(
        model, tokenizer, engine, val_task,
        num_samples=args.num_samples,
        max_examples=args.eval_examples,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # Aggregate across DDP ranks
    all_results = aggregate_results_ddp(raw_results, device)
    passk = compute_passk(all_results, args.num_samples)

    if master_process:
        passk_str = ", ".join(f"Pass@{k}: {v:.4f}" for k, v in passk.items())
        print(f"[Step {step}] {passk_str}")

    step_data = {
        "step": step,
        "passk": passk,
        "num_examples": len(all_results),
        "num_samples": args.num_samples,
    }

    # 2. Collect sample outputs (correct + incorrect examples)
    if master_process and all_results:
        correct_examples = [r for r in all_results if r["num_correct"] > 0]
        wrong_examples = [r for r in all_results if r["num_correct"] == 0]

        n_show = args.sample_problems
        samples = []
        # Show some correct, some wrong
        n_correct_show = min(n_show // 2, len(correct_examples))
        n_wrong_show = min(n_show - n_correct_show, len(wrong_examples))

        for r in correct_examples[:n_correct_show]:
            best_completion = r["completions"][r["outcomes"].index(1)] if 1 in r["outcomes"] else r["completions"][0]
            samples.append({
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "model_output": best_completion[:500],
                "correct": True,
                "pass_rate": f"{r['num_correct']}/{len(r['outcomes'])}",
            })
        for r in wrong_examples[:n_wrong_show]:
            samples.append({
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "model_output": r["completions"][0][:500],
                "correct": False,
                "pass_rate": f"0/{len(r['outcomes'])}",
            })

        step_data["samples"] = samples

        # Summary stats
        total_correct_any = sum(1 for r in all_results if r["num_correct"] > 0)
        avg_correct_rate = sum(r["num_correct"] / len(r["outcomes"]) for r in all_results) / len(all_results)
        step_data["summary"] = {
            "problems_solved_any": total_correct_any,
            "problems_total": len(all_results),
            "avg_correct_rate": avg_correct_rate,
        }

    # 3. Optional: run benchmarks
    if args.benchmarks:
        print_master(f"\n--- Benchmarks ---")
        bench_results = run_benchmarks(model, tokenizer, engine, args.offline)
        step_data["benchmarks"] = bench_results

    all_step_results[step] = step_data

    # Free memory before loading next checkpoint
    del model, engine
    torch.cuda.empty_cache() if device.type == "cuda" else None

# ---------------------------------------------------------------------------
# Generate report

if master_process:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# MathGPT Evaluation Report")
    lines.append(f"")
    lines.append(f"> Generated: {now}")
    lines.append(f"> Source: {args.source} | Model: {model_tag} | Device: {device_type}")
    lines.append(f"> GSM8K test problems: {args.eval_examples} | Samples/problem: {args.num_samples} | Temperature: {args.temperature}")
    lines.append(f"")

    # --- Pass@k summary table ---
    lines.append(f"## 1. GSM8K Pass@k Summary")
    lines.append(f"")

    # Header
    k_values = [1, 2, 4, 8, 16]
    k_values = [k for k in k_values if k <= args.num_samples]
    header = "| Step | " + " | ".join(f"Pass@{k}" for k in k_values) + " | Solved (any) | Avg Rate |"
    sep = "|------|" + "|".join("--------" for _ in k_values) + "|-------------|----------|"
    lines.append(header)
    lines.append(sep)

    best_pass1_step = None
    best_pass1_val = -1

    for step in sorted(all_step_results.keys()):
        data = all_step_results[step]
        passk = data["passk"]
        row_vals = []
        for k in k_values:
            val = passk.get(k, passk.get(str(k), 0))
            row_vals.append(f"{100*val:.1f}%")

        summary = data.get("summary", {})
        solved = summary.get("problems_solved_any", "—")
        total = summary.get("problems_total", "—")
        avg_rate = summary.get("avg_correct_rate", 0)

        p1 = passk.get(1, passk.get("1", 0))
        if p1 > best_pass1_val:
            best_pass1_val = p1
            best_pass1_step = step

        bold = "**" if p1 == best_pass1_val else ""
        lines.append(f"| {bold}{step}{bold} | " + " | ".join(row_vals) + f" | {solved}/{total} | {100*avg_rate:.1f}% |")

    lines.append(f"")
    if best_pass1_step is not None:
        lines.append(f"**Best Pass@1**: {100*best_pass1_val:.1f}% at step {best_pass1_step}")
    lines.append(f"")

    # --- Benchmarks table ---
    has_benchmarks = any("benchmarks" in d for d in all_step_results.values())
    if has_benchmarks:
        lines.append(f"## 2. Benchmark Scores")
        lines.append(f"")
        bench_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
        header = "| Step | " + " | ".join(bench_tasks) + " |"
        sep = "|------|" + "|".join("------" for _ in bench_tasks) + "|"
        lines.append(header)
        lines.append(sep)
        for step in sorted(all_step_results.keys()):
            data = all_step_results[step]
            bench = data.get("benchmarks", {})
            vals = []
            for t in bench_tasks:
                v = bench.get(t)
                vals.append(f"{100*v:.1f}%" if v is not None else "—")
            lines.append(f"| {step} | " + " | ".join(vals) + " |")
        lines.append(f"")

    # --- Sample outputs ---
    section_num = 3 if has_benchmarks else 2
    lines.append(f"## {section_num}. Sample Outputs")
    lines.append(f"")

    for step in sorted(all_step_results.keys()):
        data = all_step_results[step]
        samples = data.get("samples", [])
        if not samples:
            continue

        lines.append(f"### Step {step}")
        lines.append(f"")

        for i, s in enumerate(samples):
            status = "CORRECT" if s["correct"] else "WRONG"
            lines.append(f"**Example {i+1}** [{status}] (pass rate: {s['pass_rate']})")
            lines.append(f"")
            lines.append(f"**Q:** {s['question'][:300]}")
            lines.append(f"")
            lines.append(f"**Ground truth:** ...{s['ground_truth'][-200:]}")
            lines.append(f"")
            lines.append(f"**Model output:**")
            lines.append(f"```")
            lines.append(s["model_output"][:400])
            lines.append(f"```")
            lines.append(f"")

    # --- Analysis ---
    section_num += 1
    lines.append(f"## {section_num}. Analysis")
    lines.append(f"")

    if len(all_step_results) > 1:
        steps_sorted = sorted(all_step_results.keys())
        first_step = steps_sorted[0]
        last_step = steps_sorted[-1]
        first_p1 = all_step_results[first_step]["passk"].get(1, all_step_results[first_step]["passk"].get("1", 0))
        last_p1 = all_step_results[last_step]["passk"].get(1, all_step_results[last_step]["passk"].get("1", 0))

        lines.append(f"- **Pass@1 trajectory**: {100*first_p1:.1f}% (step {first_step}) → {100*last_p1:.1f}% (step {last_step})")
        if best_pass1_step != last_step:
            lines.append(f"- **Best checkpoint is NOT the final one** — peak at step {best_pass1_step} ({100*best_pass1_val:.1f}%), suggesting overfitting after that point")
            lines.append(f"- **Recommendation**: Use step {best_pass1_step} checkpoint for deployment")
        lines.append(f"")

    # Analyze correct vs wrong
    final_step = sorted(all_step_results.keys())[-1]
    final_data = all_step_results[final_step]
    summary = final_data.get("summary", {})
    if summary:
        solved = summary["problems_solved_any"]
        total = summary["problems_total"]
        unsolved = total - solved
        lines.append(f"- **Solved** (at least 1/{args.num_samples} correct): {solved}/{total} ({100*solved/total:.1f}%)")
        lines.append(f"- **Unsolved** (0/{args.num_samples} correct): {unsolved}/{total} ({100*unsolved/total:.1f}%)")
        lines.append(f"- Large unsolved fraction with {args.num_samples} samples indicates many problems are beyond the model's current capability")
        lines.append(f"")

    lines.append(f"---")
    lines.append(f"*Report generated by `python3 -m scripts.eval_report`*")

    report_text = "\n".join(lines)

    output_path = os.path.join(_PROJECT_ROOT, args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved to: {output_path}")

    # Also dump raw JSON for further analysis
    json_path = output_path.replace(".md", ".json")
    # Convert int keys to string for JSON
    json_data = {}
    for step, data in all_step_results.items():
        d = dict(data)
        if "passk" in d:
            d["passk"] = {str(k): v for k, v in d["passk"].items()}
        json_data[str(step)] = d
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Raw data saved to: {json_path}")

compute_cleanup()

"""
MathGPT: Reinforcement learning training on GSM8K math problems.

This script trains a math reasoning model using REINFORCE-style policy gradient
(simplified GRPO) on the GSM8K grade-school math dataset.

Prerequisites:
  1. Train nanochat base model:  cd ../nanochat && python -m scripts.base_train
  2. Run nanochat SFT:           cd ../nanochat && python -m scripts.chat_sft

Then run MathGPT RL training:
  Single GPU:
    python -m scripts.train_rl

  Multi-GPU (8x):
    torchrun --standalone --nproc_per_node=8 -m scripts.train_rl

The trained checkpoints are saved to:
  ./runs/chatrl_checkpoints/<model_tag>/
"""

import os
import sys

# 项目根目录（MathGPT/）加入 sys.path，使 nanochat/ tasks/ 包可直接导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.join(_PROJECT_ROOT, "runs"))

import argparse
import itertools
import wandb
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="MathGPT RL training on GSM8K")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading - loads from nanochat SFT checkpoints by default
parser.add_argument("--source", type=str, default="sft", choices=["sft", "rl"], help="source checkpoint type: sft (start fresh RL) or rl (continue RL)")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from (e.g. 'd12')")
parser.add_argument("--model-step", type=int, default=None, help="model step to load (default: latest)")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K training set")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of rollout samples per math problem")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature for rollouts")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
parser.add_argument("--offline", type=str, default=None, help="path to local HF datasets dir for offline training")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="mathgpt-rl", name=args.run, config=user_config)

# TensorBoard logging init
tb_writer = None
if master_process:
    tb_log_dir = os.path.join(get_base_dir(), "tb_logs", "train_rl", args.run if args.run != "dummy" else "default")
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print0(f"TensorBoard logging to: {tb_log_dir}")

print0(f"MathGPT RL Training")
print0(f"Checkpoints will be saved to: {get_base_dir()}/chatrl_checkpoints/")
print0(f"Loading {args.source} checkpoint...")

# Init model and tokenizer — load from SFT or prior RL checkpoint
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Rollout generator: yields batches of (sequences, inputs, targets, rewards, advantages)

train_task = GSM8K(subset="main", split="train", offline_dir=args.offline)
val_task   = GSM8K(subset="main", split="test", offline_dir=args.offline)
num_steps  = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"GSM8K train size: {len(train_task)}  |  steps: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices  = range(ddp_rank, len(train_task), ddp_world_size)
    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]

        # Tokenize up to (but not past) the assistant start token
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate `num_samples` rollouts, chunked to avoid OOM
        model.eval()
        generated_seqs, masks = [], []
        for sampling_step in range(args.num_samples // args.device_batch_size):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            seqs_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed,
            )
            generated_seqs.extend(seqs_batch)
            masks.extend(masks_batch)

        # Score each rollout with the GSM8K reward function (1.0 correct / 0.0 wrong)
        rewards = []
        for sample_tokens in generated_seqs:
            generated_text = tokenizer.decode(sample_tokens[prefix_length:])
            rewards.append(train_task.reward(conversation, generated_text))

        # Pad all sequences to the same length
        max_len = max(len(s) for s in generated_seqs)
        padded_seqs  = [s + [assistant_end] * (max_len - len(s)) for s in generated_seqs]
        padded_masks = [m + [0]            * (max_len - len(m)) for m in masks]

        ids      = torch.tensor(padded_seqs,  dtype=torch.long,  device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long,  device=device)

        inputs  = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1  # ignore prompt tokens in the loss

        rewards    = torch.tensor(rewards, dtype=torch.float, device=device)
        advantages = rewards - rewards.mean()  # REINFORCE with baseline

        yield generated_seqs, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# Evaluation: GSM8K pass@k

def run_gsm8k_eval(task, tokenizer, engine, max_examples=None, num_samples=1,
                   max_completion_tokens=256, temperature=0.0, top_k=50):
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        assert num_samples <= args.device_batch_size
        generated_seqs, _ = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        outcomes = [
            {"is_correct": task.evaluate(conversation, tokenizer.decode(s[prefix_length:]))}
            for s in generated_seqs
        ]
        yield {"idx": idx, "outcomes": outcomes}

# -----------------------------------------------------------------------------
# Optimizer setup

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Start from a fraction of the base LR and decay to zero over training
for group in optimizer.param_groups:
    group["lr"]         = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

def get_lr_multiplier(it):
    return 1.0 - it / num_steps

print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}")
assert args.examples_per_step % ddp_world_size == 0
examples_per_rank = args.examples_per_step // ddp_world_size
print0(f"Examples per rank per step: {examples_per_rank}")

# -----------------------------------------------------------------------------
# Training loop

batch_iterator = get_batch()

for step in range(num_steps):

    # Periodic evaluation
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device)
        records = list(run_gsm8k_eval(
            val_task, tokenizer, engine,
            num_samples=args.device_batch_size,
            max_examples=args.eval_examples,
            temperature=1.0,
        ))
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk /= num_records.item()
        print_passk = [f"Pass@{k}: {passk[k-1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"[Eval] Step {step} | {', '.join(print_passk)}")
        wandb_run.log({"step": step, **{f"pass@{k}": passk[k-1].item() for k in range(1, args.device_batch_size + 1)}})
        if tb_writer is not None:
            for k in range(1, args.device_batch_size + 1):
                tb_writer.add_scalar(f"eval/pass@{k}", passk[k-1].item(), step)

    # Accumulate gradients over `examples_per_rank` problems
    rewards_list, sequence_lengths = [], []
    for example_step in range(examples_per_rank):
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        model.train()
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs     = inputs_all[b0:b1]
            targets    = targets_all[b0:b1]
            advantages = advantages_all[b0:b1]

            # Policy gradient objective: maximize E[logp * advantage]
            logp      = -model(inputs, targets, loss_reduction='none').view_as(inputs)
            pg_obj    = (logp * advantages.unsqueeze(-1)).sum()
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj    = pg_obj / (num_valid * num_passes * examples_per_rank)
            loss      = -pg_obj
            loss.backward()

            rewards = rewards_all[b0:b1]
            print0(f"Step {step}/{num_steps} | Ex {example_step} | Pass {pass_idx} "
                   f"| loss: {loss.item():.6f} | reward: {rewards.mean().item():.3f}")

        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(s) for s in sequences_all)

    # Log step statistics
    mean_reward  = sum(rewards_list) / len(rewards_list)
    mean_seq_len = sum(sequence_lengths) / len(sequence_lengths)
    if ddp:
        r_t = torch.tensor(mean_reward,  dtype=torch.float, device=device)
        l_t = torch.tensor(mean_seq_len, dtype=torch.float, device=device)
        dist.all_reduce(r_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(l_t, op=dist.ReduceOp.AVG)
        mean_reward, mean_seq_len = r_t.item(), l_t.item()
    print0(f"Step {step}/{num_steps} | reward: {mean_reward:.4f} | seq_len: {mean_seq_len:.1f}")
    wandb_run.log({"step": step, "reward": mean_reward, "sequence_length": mean_seq_len})
    if tb_writer is not None:
        tb_writer.add_scalar("train/reward", mean_reward, step)
        tb_writer.add_scalar("train/sequence_length", mean_seq_len, step)

    # Update parameters
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({"step": step, "lr_multiplier": lrm})
    if tb_writer is not None:
        tb_writer.add_scalar("train/lr_multiplier", lrm, step)

    # Save checkpoint
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        depth          = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"math_d{depth}"
        checkpoint_dir = os.path.join(get_base_dir(), "chatrl_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,
            {"model_config": model.config.__dict__},
        )
        print0(f"Saved checkpoint to {checkpoint_dir}/model_{step:06d}.pt")

if tb_writer is not None:
    tb_writer.close()
wandb_run.finish()
compute_cleanup()
batch_iterator.close()
print0("MathGPT RL training complete.")

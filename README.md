# MathGPT

A math reasoning assistant built on the [nanochat](../nanochat) framework.
Uses REINFORCE-style RL (simplified GRPO) to train a GPT model to solve
grade-school math problems (GSM8K) step by step.

## Architecture

```
nanochat/          ← base framework (GPT model, tokenizer, training utils)
MathGPT/
├── scripts/
│   ├── train_rl.py   ← RL training on GSM8K math problems
│   ├── chat_cli.py   ← interactive CLI chat
│   └── chat_web.py   ← web server with LaTeX math rendering
├── math_gpt/
│   └── ui.html       ← math-focused web UI with KaTeX rendering
└── runs/             ← checkpoints (gitignored)
```

## Training pipeline

### 1. Prerequisites — train the base model with nanochat

```bash
cd ../nanochat

# (a) Train the base language model
python -m scripts.base_train --depth 12

# (b) SFT fine-tuning (teaches the model chat format + basic math)
python -m scripts.chat_sft
```

### 2. MathGPT RL training

```bash
cd ../MathGPT

# Install dependencies
uv sync          # or: pip install -e ../nanochat && pip install -e .

# Single GPU
python -m scripts.train_rl

# Multi-GPU (4x)
torchrun --standalone --nproc_per_node=4 -m scripts.train_rl

# Resume from previous RL checkpoint
python -m scripts.train_rl --source rl --model-tag math_d12

# With wandb logging
python -m scripts.train_rl --run mathgpt-v1
```

Checkpoints are saved to `runs/chatrl_checkpoints/math_d<N>/`.

### 3. Chat with MathGPT

**CLI:**
```bash
python -m scripts.chat_cli                        # uses RL checkpoint
python -m scripts.chat_cli --source sft           # uses SFT checkpoint
python -m scripts.chat_cli --prompt "What is 15% of 80?"
```

**Web UI** (with LaTeX math rendering):
```bash
python -m scripts.chat_web               # http://localhost:8000
python -m scripts.chat_web --port 8080
python -m scripts.chat_web --source sft  # use SFT model
```

Open http://localhost:8000 — ask any math problem and see step-by-step solutions.

## How RL training works

For each training step:

1. **Sample rollouts** — for each GSM8K problem, generate `N` candidate solutions
2. **Score rewards** — compare each solution's final answer against the ground truth
   - Correct answer → reward = 1.0
   - Wrong answer   → reward = 0.0
3. **Compute advantages** — `advantage = reward − mean(reward)`
4. **Policy gradient** — update the model to increase probability of high-advantage responses:
   ```
   loss = −(logp × advantage)
   ```

This is a clean REINFORCE implementation without KL regularization or PPO clipping,
following the DAPO style with token-level normalization.

## Web UI features

- **KaTeX rendering** — LaTeX math in responses is rendered automatically
  - Inline: `$E = mc^2$`
  - Display: `$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$`
- **Streaming** — responses appear token by token
- **Slash commands** — `/temperature 0.8`, `/topk 30`, `/clear`, `/help`
- **Edit & regenerate** — click any message to edit/regenerate
- **Example prompts** — click chips to try sample problems

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `rl` | Checkpoint to load: `sft` or `rl` |
| `--model-tag` | auto | Model tag (e.g. `math_d12`) |
| `--temperature` | 0.6 | Sampling temperature |
| `--top-k` | 50 | Top-k sampling |
| `--max-tokens` | 512 | Max response length |

Checkpoint directory is controlled by the `NANOCHAT_BASE_DIR` environment variable
(defaults to `./runs/` when running MathGPT scripts).

"""
MathGPT interactive CLI chat.

Usage:
  python -m scripts.chat_cli                           # chat with RL-trained math model
  python -m scripts.chat_cli --source sft              # chat with SFT model instead
  python -m scripts.chat_cli --prompt "What is 2+2?"  # single-shot mode
  python -m scripts.chat_cli --source rl --model-tag math_d12

Commands inside the chat:
  clear   - start a new conversation
  quit    - exit
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.join(_PROJECT_ROOT, "runs"))

import argparse
import torch

from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description="MathGPT CLI Chat")
parser.add_argument("--source", type=str, default="rl", choices=["sft", "rl"],
                    help="Checkpoint type to load: 'rl' (math RL) or 'sft' (base SFT)")
parser.add_argument("--model-tag", type=str, default=None, help="Model tag, e.g. 'math_d12'")
parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
parser.add_argument("--prompt", type=str, default="", help="Single-shot mode: run one prompt and exit")
parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps", ""],
                    help="Device type (empty = autodetect)")
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

print(f"Loading MathGPT ({args.source}) checkpoint...")
model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                    model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)

bos           = tokenizer.get_bos_token_id()
user_start    = tokenizer.encode_special("<|user_start|>")
user_end      = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end   = tokenizer.encode_special("<|assistant_end|>")

print()
print("=" * 60)
print("  MathGPT - Math Reasoning Assistant")
print("=" * 60)
print("Ask me any math problem! I can solve arithmetic, algebra,")
print("word problems, and more step by step.")
print()
print("Commands: 'clear' to reset, 'quit'/'exit' to leave")
print("=" * 60)

conversation_tokens = [bos]

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break

    if user_input.lower() == "clear":
        conversation_tokens = [bos]
        print("Conversation cleared. Ask a new math problem!")
        continue

    if not user_input:
        continue

    # Append user turn
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)

    # Stream the assistant response
    response_tokens = []
    print("\nMathGPT: ", end="", flush=True)
    for token_column, token_masks in engine.generate(
        conversation_tokens,
        num_samples=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ):
        token = token_column[0]
        response_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
    print()

    if response_tokens and response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    if args.prompt:
        break

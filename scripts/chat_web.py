"""
MathGPT web chat server.

Serves a chat UI with LaTeX/KaTeX math rendering.
The model streams responses token-by-token.

Usage:
  python -m scripts.chat_web                         # default: RL model, port 8000
  python -m scripts.chat_web --source sft            # use SFT model
  python -m scripts.chat_web --port 8080             # custom port
  python -m scripts.chat_web --model-tag math_d12    # specific model tag

Then open http://localhost:8000 in your browser.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.join(_PROJECT_ROOT, "runs"))

import argparse
import json
import random
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Optional, AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Abuse-prevention limits (same as nanochat)
MAX_MESSAGES = 500
MAX_MSG_LEN  = 8000
MAX_TOTAL_LEN = 32000

parser = argparse.ArgumentParser(description="MathGPT Web Server")
parser.add_argument("--source", type=str, default="rl", choices=["sft", "rl"],
                    help="Checkpoint type: 'rl' (math RL) or 'sft' (base SFT)")
parser.add_argument("--model-tag", type=str, default=None, help="Model tag, e.g. 'math_d12'")
parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for parallel workers")
parser.add_argument("--temperature", type=float, default=0.6, help="Default sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="Default top-k sampling")
parser.add_argument("--max-tokens", type=int, default=512, help="Default max tokens per response")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps", ""],
                    help="Device type (empty = autodetect)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)


@dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object


class WorkerPool:
    import asyncio as _asyncio

    def __init__(self, num_gpus: int = 1):
        import asyncio
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available: "asyncio.Queue[Worker]" = None  # initialised in async context

    async def initialize(self, source, model_tag=None, step=None):
        import asyncio
        self.available = asyncio.Queue()
        for gpu_id in range(self.num_gpus):
            if device_type == "cuda":
                dev = torch.device(f"cuda:{gpu_id}")
            else:
                dev = torch.device(device_type)
            logger.info(f"Loading MathGPT ({source}) on {dev}...")
            m, tok, _ = load_model(source, dev, phase="eval", model_tag=model_tag, step=step)
            worker = Worker(gpu_id=gpu_id, device=dev, engine=Engine(m, tok), tokenizer=tok)
            self.workers.append(worker)
            await self.available.put(worker)
        logger.info(f"All {self.num_gpus} worker(s) ready.")

    async def acquire(self) -> Worker:
        return await self.available.get()

    async def release(self, worker: Worker):
        await self.available.put(worker)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None


def validate(req: ChatRequest):
    if not req.messages:
        raise HTTPException(400, "No messages provided")
    if len(req.messages) > MAX_MESSAGES:
        raise HTTPException(400, f"Too many messages (max {MAX_MESSAGES})")
    total = 0
    for i, m in enumerate(req.messages):
        if not m.content:
            raise HTTPException(400, f"Message {i} is empty")
        if len(m.content) > MAX_MSG_LEN:
            raise HTTPException(400, f"Message {i} too long (max {MAX_MSG_LEN} chars)")
        total += len(m.content)
        if m.role not in ("user", "assistant"):
            raise HTTPException(400, f"Invalid role '{m.role}'")
    if total > MAX_TOTAL_LEN:
        raise HTTPException(400, f"Total conversation too long (max {MAX_TOTAL_LEN} chars)")
    if req.temperature is not None and not (0.0 <= req.temperature <= 2.0):
        raise HTTPException(400, "temperature must be in [0.0, 2.0]")
    if req.top_k is not None and not (0 <= req.top_k <= 200):
        raise HTTPException(400, "top_k must be in [0, 200]")
    if req.max_tokens is not None and not (1 <= req.max_tokens <= 4096):
        raise HTTPException(400, "max_tokens must be in [1, 4096]")


async def stream_response(worker: Worker, tokens, temperature, max_tokens, top_k) -> AsyncGenerator[str, None]:
    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos_id        = worker.tokenizer.get_bos_token_id()
    accumulated   = []
    last_clean    = ""

    for token_column, _ in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1),
    ):
        token = token_column[0]
        if token in (assistant_end, bos_id):
            break
        accumulated.append(token)
        current_text = worker.tokenizer.decode(accumulated)
        if not current_text.endswith(""):
            new_text = current_text[len(last_clean):]
            if new_text:
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = WorkerPool(num_gpus=args.num_gpus)
    await pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    app.state.pool = pool
    logger.info(f"MathGPT server ready at http://localhost:{args.port}")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    ui_path = os.path.join(_PROJECT_ROOT, "math_gpt", "ui.html")
    with open(ui_path, encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


@app.get("/health")
async def health():
    pool = getattr(app.state, "pool", None)
    return {
        "status": "ok",
        "ready": pool is not None and len(pool.workers) > 0,
        "num_workers": pool.num_gpus if pool else 0,
    }


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    validate(request)

    pool   = app.state.pool
    worker = await pool.acquire()

    try:
        tok = worker.tokenizer
        bos             = tok.get_bos_token_id()
        user_start      = tok.encode_special("<|user_start|>")
        user_end        = tok.encode_special("<|user_end|>")
        assistant_start = tok.encode_special("<|assistant_start|>")
        assistant_end   = tok.encode_special("<|assistant_end|>")

        conv_tokens = [bos]
        for msg in request.messages:
            if msg.role == "user":
                conv_tokens += [user_start] + tok.encode(msg.content) + [user_end]
            elif msg.role == "assistant":
                conv_tokens += [assistant_start] + tok.encode(msg.content) + [assistant_end]
        conv_tokens.append(assistant_start)

        temperature = request.temperature if request.temperature is not None else args.temperature
        max_tokens  = request.max_tokens  if request.max_tokens  is not None else args.max_tokens
        top_k       = request.top_k       if request.top_k       is not None else args.top_k

        response_chunks = []

        async def stream_and_release():
            try:
                async for chunk in stream_response(worker, conv_tokens, temperature, max_tokens, top_k):
                    data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in data:
                        response_chunks.append(data["token"])
                    yield chunk
            finally:
                logger.info(f"[ASSISTANT]: {''.join(response_chunks)[:120]}")
                await pool.release(worker)

        return StreamingResponse(stream_and_release(), media_type="text/event-stream")

    except Exception as e:
        await pool.release(worker)
        raise e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

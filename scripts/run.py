"""
MathGPT 兼容启动器 —— 在 nanochat 目录以子进程方式运行目标模块。

用法:
    python3 -m scripts.run <module> [args...]

例:
    python3 -m scripts.run nanochat.dataset -n 8
    python3 -m scripts.run scripts.tok_train
    python3 -m scripts.run scripts.base_train --depth=6 ...
    python3 -m scripts.run scripts.chat_sft ...

nanochat 源码已直接修复了 Python 3.12 / PyTorch 2.3 的兼容问题，
此启动器仅负责切换工作目录和传递 NANOCHAT_BASE_DIR 环境变量。
"""
import os
import sys
import subprocess

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NANOCHAT_DIR = os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "nanochat"))
_RUNS_DIR     = os.path.join(_PROJECT_ROOT, "runs")

if len(sys.argv) < 2:
    print("用法: python3 -m scripts.run <module> [args...]")
    sys.exit(1)

module_name = sys.argv[1]
extra_args  = sys.argv[2:]

env = os.environ.copy()
env["NANOCHAT_BASE_DIR"] = _RUNS_DIR

# 在 nanochat 项目目录下运行，这样 scripts.* 都能找到
cmd = [sys.executable, "-m", module_name] + extra_args
print(f"[run] cd {_NANOCHAT_DIR}  &&  {' '.join(cmd)}")
result = subprocess.run(cmd, cwd=_NANOCHAT_DIR, env=env)
sys.exit(result.returncode)

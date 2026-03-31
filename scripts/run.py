"""
MathGPT 兼容启动器。

用法:
    python3 -m scripts.run <module_name> [args...]

例:
    python3 -m scripts.run nanochat.dataset -n 8
    python3 -m scripts.run scripts.tok_train
    python3 -m scripts.run scripts.base_train --depth=6 ...
    python3 -m scripts.run scripts.chat_sft ...

先应用 PyTorch/Python 3.12 兼容补丁，再以 runpy 方式运行目标模块。
"""
import os
import sys
import runpy

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NANOCHAT_DIR = os.path.join(_PROJECT_ROOT, "..", "nanochat")

os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.join(_PROJECT_ROOT, "runs"))
if _NANOCHAT_DIR not in sys.path:
    sys.path.insert(0, _NANOCHAT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 先打补丁，再导入任何 nanochat 模块 ──
from math_gpt import compat
compat.apply()

if len(sys.argv) < 2:
    print("用法: python3 -m scripts.run <module_name> [args...]")
    sys.exit(1)

module_name = sys.argv[1]
# 剩余参数传给目标模块
sys.argv = sys.argv[1:]

runpy.run_module(module_name, run_name="__main__", alter_sys=True)

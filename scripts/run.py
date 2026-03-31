"""
MathGPT 启动器 —— 设置环境后以 runpy 方式运行任意脚本模块。

用法:
    python3 -m scripts.run <module> [args...]

例:
    python3 -m scripts.run nanochat.dataset -n 8
    python3 -m scripts.run scripts.tok_train
    python3 -m scripts.run scripts.base_train --depth=6 ...
    python3 -m scripts.run scripts.chat_sft ...
"""
import os
import sys
import runpy

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.join(_PROJECT_ROOT, "runs"))

if len(sys.argv) < 2:
    print("用法: python3 -m scripts.run <module> [args...]")
    sys.exit(1)

module_name = sys.argv[1]
sys.argv = sys.argv[1:]   # 把剩余参数交给目标模块

runpy.run_module(module_name, run_name="__main__", alter_sys=True)

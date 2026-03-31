"""
MathGPT - A math reasoning model built on the nanochat framework.

Training pipeline:
1. Train base model:     cd ../nanochat && python -m scripts.base_train
2. SFT fine-tuning:      cd ../nanochat && python -m scripts.chat_sft
3. Math RL training:     python -m scripts.train_rl
4. Chat (CLI):           python -m scripts.chat_cli
5. Chat (Web):           python -m scripts.chat_web
"""

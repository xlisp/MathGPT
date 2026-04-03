  # 1. 本地下载（有网络的机器上）
  python scripts/download_hf_datasets.py --output_dir data/hf_datasets

  # 2. 上传到 A800 服务器
  scp -r data/hf_datasets/ root@<A800_IP>:/mnt/openclaw/MathGPT/data/hf_datasets/

  # 3. 在服务器上重新运行训练
  bash scripts/full_train_a800.sh

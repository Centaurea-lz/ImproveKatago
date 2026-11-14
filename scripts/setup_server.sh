#!/usr/bin/env bash
# AutoDL image: Ubuntu 22.04 + CUDA 13.2 + PyTorch (preinstalled) + Python 3.12.
# Run on the server:  bash setup_server.sh
set -euo pipefail

# 1) Verify GPU + torch (torch is already installed by the image — do NOT reinstall it)
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# 2) Install only the light deps
pip install numpy sgfmill flask

echo "OK — environment ready."

#!/bin/bash
set -e

echo "[*] Installing CUDA Toolkit 12.8 on Ubuntu 22.04..."

# 1. 清理旧源
sudo rm -f /etc/apt/sources.list.d/cuda*.list
sudo rm -f /etc/apt/sources.list.d/nvidia*.list

# 2. 添加 keyring
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb

# 3. 添加 CUDA 12.8 源
sudo tee /etc/apt/sources.list.d/cuda-12-8.list > /dev/null <<EOF
deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /
EOF

# 4. 安装 toolkit
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8

# 5. 设置默认路径
sudo ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# 6. 验证
source ~/.bashrc
/usr/local/cuda/bin/nvcc --version

echo "[✔] CUDA 12.8 installation complete."

# Face-Parsing Lower-Face Occlusion Detection

基于 zllrunning/face-parsing.PyTorch BiSeNet 19 类语义分割网络的下半脸/唇部遮挡自动化过滤模块。


### 1. 安装依赖
pip install torch torchvision opencv-python pillow numpy

### 2. 克隆子模块与下载预训练权重
cd occlusion_detection

# 克隆官方 BiSeNet 仓库
git clone https://github.com/zllrunning/face-parsing.PyTorch.git face_parsing_repo

# 创建权重目录并下载权重文件 (79999_iter.pth)
mkdir -p face_parsing_repo/res/cp
curl -L -o face_parsing_repo/res/cp/79999_iter.pth "https://huggingface.co/vivym/face-parsing-bisenet/resolve/main/79999_iter.pth"

### 3. 运行遮挡评估
python occlusion_filter.py
# Install PyTorch and PyTorch-Geometric
TORCH=1.4.0

# Linux with Cuda 10.0
CUDA=cu100
pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster==1.5.4 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==1.6.3

# Linux with Cuda 10.1
#CUDA=cu101
#pip install torch==${TORCH} -f https://download.pytorch.org/whl/${CUDA}/torch_stable.html
#pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#pip install torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#pip install torch-cluster==1.5.4 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#pip install torch-geometric==1.6.3

# MacOS
#CUDA=cpu
#pip install torch==${TORCH}
#MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-cluster==1.5.4 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-geometric==1.6.3

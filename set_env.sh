
conda activate InfiTensor_py310
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

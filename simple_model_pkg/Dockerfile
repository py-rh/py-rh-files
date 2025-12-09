# Base image: NVIDIA PyTorch container with CUDA 12.9 and PyTorch 2.8
# See: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:25.06-py3

# Install torchvision and additional dependencies for FashionMNIST training
RUN pip install --no-cache-dir \
    torchvision \
    pillow

# Set working directory
WORKDIR /workspace

# Copy model file into the container
COPY . /workspace/.

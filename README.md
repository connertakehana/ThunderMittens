# ThunderMittens

## Overview

ThunderMittens is an Apple Metal Shading Language (MSL) port for ThunderKittens, providing a robust framework for GPU-accelerated computing on Apple Silicon.

## Project Structure

The repository supports two primary use cases:

### 1. MSL Kernel Development

For writing Metal Shading Language (MSL) kernels:
- Clone the ThunderKittens repository
- Open the project in Xcode
- Xcode will handle all build processes

### 2. MLX Kernel Integration with Python

For using ThunderKittens kernels within MLX in Python:

#### Prerequisites
- Python 3.8+
- CMake
- Xcode Command Line Tools

#### Installation Steps

1. Navigate to ThunderMittens/mlx directory
2. Install MLX with parallel build:
   ```bash
   CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e ".[dev]"
   ```

3. Navigate to ThunderMittens/kernels directory
4. Install kernel requirements:
   ```bash
   pip install -r requirements.txt
   ```

5. Build kernels and bindings:
   ```bash
   python setup.py build_ext -j8 --inplace
   ```

## Dependencies

- ThunderKittens
- MLX
- Apple Metal Framework
- Python development tools

## Support

For issues and questions, please open a GitHub issue in the repository.

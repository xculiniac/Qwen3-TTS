Based on the successful install, this is the configuration I’d reuse for future **Windows + RTX 5090 + CUDA 13.2 + FlashAttention 2** installs.

Your original failure came from FlashAttention building its C++ extension as C++17 while PyTorch 2.13 headers contained C++20-only constructs.  The log also showed the separate Windows `M_LOG2E` problem during CUDA compilation. 

### Known-working stack

```text
Windows 11
Python 3.12
NVIDIA RTX 5090
Compute Capability 12.0 / SM120

CUDA Toolkit 13.2
nvcc 13.2

Visual Studio 2026
MSVC 19.51

PyTorch     2.12.1+cu132
TorchVision 0.27.1+cu132
TorchAudio  2.11.0+cu132

FlashAttention 2.8.4 source build
```

The main fix was **using PyTorch 2.12.1 instead of PyTorch 2.13.0**. Do not force FlashAttention to C++20, and you do not need to install the older VS2022/MSVC 19.39 compiler.

For a new environment, I would use:

```bat
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132

pip install torchaudio==2.11.0+cu132 --index-url https://download.pytorch.org/whl/test/cu132 --no-deps

pip install ninja packaging psutil einops setuptools wheel
```

Then verify the basic CUDA stack:

```bat
python -c "import torch; print('Torch:',torch.__version__); print('CUDA:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0)); print('CC:',torch.cuda.get_device_capability(0))"
```

Expected:

```text
Torch: 2.12.1+cu132
CUDA: 13.2
GPU: NVIDIA GeForce RTX 5090
CC: (12, 0)
```

For compiling FlashAttention, use these environment variables:

```bat
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2

set DISTUTILS_USE_SDK=1
set FLASH_ATTENTION_FORCE_BUILD=TRUE

set FLASH_ATTN_CUDA_ARCHS=120

set MAX_JOBS=2
set NVCC_THREADS=1

set CL=/Zc:preprocessor /Zc:__cplusplus
```

`FLASH_ATTN_CUDA_ARCHS=120` is important for the RTX 5090. Your successful compilation was correctly generating `sm_120` code. 

Clone FlashAttention rather than letting pip compile from a long temporary directory:

```bat
cd /d C:\Users\alexa

git clone https://github.com/Dao-AILab/flash-attention.git fa
cd fa
```

For the Windows `M_LOG2E` issue, edit:

```text
C:\Users\alexa\fa\csrc\flash_attn\src\softmax.h
```

and add near the top, after `<cmath>`:

```cpp
#ifndef M_LOG2E
#define M_LOG2E 1.44269504088896340736
#endif
```

Then compile:

```bat
cd /d C:\Users\alexa\fa

python -m pip install . --no-build-isolation --no-cache-dir -v
```

Do **not** patch PyTorch's `StringUtil.h`, `AutogradState.h`, etc. when using Torch 2.12.1. Those patches were attempts to work around the PyTorch 2.13/C++20 incompatibility and are unnecessary in the working configuration.

## Validation test

First do a quick import test:

```bat
python -c "import torch, flash_attn; print('Torch:',torch.__version__); print('CUDA:',torch.version.cuda); print('FlashAttention:',flash_attn.__version__); print('GPU:',torch.cuda.get_device_name(0)); from flash_attn import flash_attn_func; print('FlashAttention import OK')"
```

Then run an actual FlashAttention GPU calculation:

```python
import torch
import flash_attn
from flash_attn import flash_attn_func

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

# [batch, sequence, heads, head_dim]
q = torch.randn(
    2, 512, 16, 64,
    device="cuda",
    dtype=torch.float16
)

k = torch.randn_like(q)
v = torch.randn_like(q)

output = flash_attn_func(
    q,
    k,
    v,
    causal=True
)

print("Input shape :", q.shape)
print("Output shape:", output.shape)
print("Output dtype:", output.dtype)
print("Output CUDA :", output.is_cuda)
print("Contains NaN:", torch.isnan(output).any().item())

assert output.shape == q.shape
assert output.is_cuda
assert not torch.isnan(output).any()

print("\nFLASHATTENTION 2 TEST PASSED")
```

Save that as:

```text
test_flash_attn.py
```

and run:

```bat
python test_flash_attn.py
```

A successful result should end with:

```text
Torch: 2.12.1+cu132
CUDA: 13.2
FlashAttention: 2.8.4
GPU: NVIDIA GeForce RTX 5090
Compute capability: (12, 0)

Input shape : torch.Size([2, 512, 16, 64])
Output shape: torch.Size([2, 512, 16, 64])
Output dtype: torch.float16
Output CUDA : True
Contains NaN: False

FLASHATTENTION 2 TEST PASSED
```

That test is better than just `import flash_attn`: it confirms the compiled CUDA extension actually loads and executes a FlashAttention kernel on the **RTX 5090**.

# Activate MSVC

"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

# Install torch, torch audio with CU132
python -m pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132

or

pip install torchaudio==2.11.0+cu132 --index-url https://download.pytorch.org/whl/test/cu132 --no-deps
pip install torch --index-url https://download.pytorch.org/whl/cu132


NEW
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132
pip install torchaudio==2.11.0+cu132 --index-url https://download.pytorch.org/whl/test/cu132 --no-deps
python -c "import torch,torchaudio; print(torch.__version__); print(torch.version.cuda); print(torchaudio.__version__); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"

# Install 
set FLASH_ATTENTION_FORCE_BUILD=TRUE
pip install flash-attn==2.8.3.post1 --no-build-isolation -v

# dlrm

## Env
- cuda 12.4

## Useful Commands

```shell
# env setup
python -m venv .venv
source ./.venv/bin/activate
pip install --upgrade pip

pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torchrec --index-url https://download.pytorch.org/whl/cu124
```
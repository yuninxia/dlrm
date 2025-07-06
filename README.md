# dlrm

## Env
- cuda 12.4

## Useful Commands

```shell

# git clone
git clone --recursive https://github.com/yuninxia/dlrm

# env setup
python -m venv .venv
source ./.venv/bin/activate # use deactivate to exit
pip install --upgrade pip

pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torchrec --index-url https://download.pytorch.org/whl/cu124

# profiling
source /home/ynxia/playground/hpctoolkit/setup_hpctoolkit.sh
hpcstruct --gpucf yes -j 64 --psize 20000000 ./hpctoolkit-python3.11-measurements/
```
Experiment Workload Introduction: Deep Learning Recommendation Model (DLRM)
1. Experiment Goal
The objective of this experiment is to conduct an in-depth performance analysis of a typical Deep Learning Recommendation Model (DLRM) workload. We are using HPCToolkit as our performance analysis tool to run the model on an NVIDIA GPU, collecting both CPU Time (CPUTIME) and detailed GPU performance metrics (gpu=nvidia,pc) during its execution.
The core goal is to identify performance bottlenecks within the model, thereby providing data-driven support for subsequent performance optimization efforts.
2. Core Workload Configuration
- Project: https://github.com/facebookresearch/dlrm
- Model Framework: PyTorch
- Execution Device: NVIDIA H100 80GB HBM3
- Dataset: Synthetic Data
  - We are using randomly generated data (--data-generation=random). This allows us to focus on the computational characteristics of the model itself, eliminating the interference of disk I/O from real-world scenarios on our performance analysis.
3. DLRM Model Architecture Parameters
Our DLRM model consists of three main components: a Bottom Multi-Layer Perceptron (MLP) for processing dense features, an Embedding Layer for sparse features, and a Top MLP.
- Bottom MLP:
  - --arch-mlp-bot="512-512-64"
  - A three-layer fully connected network designed to process dense features.
- Embedding Layer:
  - --arch-sparse-feature-size=64
  - Maps high-dimensional sparse features into 64-dimensional dense embedding vectors.
- Feature Interaction:
  - --arch-interaction-op="dot"
  - Uses the Dot Product to perform feature crosses between the embedding vectors and the output of the bottom MLP. This is a core operation in DLRM.
- Top MLP:
  - --arch-mlp-top="1024-1024-1024-1"
  - A four-layer fully connected network that makes the final prediction based on the interacted features, outputting a single scalar value (e.g., click-through rate).
4. Runtime Configuration
- Batch Size:
  - --mini-batch-size=2048
  - A large batch size is set with the goal of fully utilizing the parallel computing capabilities of the GPU.
- Number of Iterations:
  - --num-batches=100
  - The experiment runs for 100 batches to quickly capture steady-state performance characteristics.
- Data Loading:
  - Data loading is performed synchronously within the main process.
#!/usr/bin/env python
"""
Tiny hybrid‑DLRM demo
CPU‑side: 2 embedding tables (32 k rows × 16 d)   ≈ 1 MB each
GPU‑side: 3‑layer MLP (32 → 64 → 64 → 1)
One forward + backward pass to exercise H2D/D2H.
Runtime: < 2 s on a single modern GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrec
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

torch.manual_seed(0)
device_gpu = torch.device("cuda")

# 1. EmbeddingBagCollection on CPU (slightly larger tables)
ebc = torchrec.EmbeddingBagCollection(
    device="cpu",
    tables=[
        torchrec.EmbeddingBagConfig(
            name="product_table",
            embedding_dim=16,
            num_embeddings=32_768,      # was 4 096
            feature_names=["product"],
            pooling=torchrec.PoolingType.SUM,
        ),
        torchrec.EmbeddingBagConfig(
            name="user_table",
            embedding_dim=16,
            num_embeddings=32_768,
            feature_names=["user"],
            pooling=torchrec.PoolingType.SUM,
        ),
    ]
)

# 2. 3‑layer MLP on GPU
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)

mlp = MLP(in_dim=32, hidden=64).to(device_gpu)

# 3. Build a tiny random batch (multi‑hot length 2)
product_jt = torchrec.sparse.jagged_tensor.JaggedTensor(
    values=torch.tensor([1, 2, 3, 4], dtype=torch.long),
    lengths=torch.tensor([2, 2], dtype=torch.long),
)
user_jt = torchrec.sparse.jagged_tensor.JaggedTensor(
    values=torch.tensor([5, 6, 7, 8], dtype=torch.long),
    lengths=torch.tensor([2, 2], dtype=torch.long),
)
kjt = KeyedJaggedTensor.from_jt_dict({"product": product_jt, "user": user_jt})

# 4. Forward + backward (so gradients move D2H)
optim = torch.optim.Adagrad(list(ebc.parameters()) + list(mlp.parameters()), lr=0.1)

# Forward
emb = ebc(kjt)                                        # CPU lookup
dense_in = torch.cat([emb["product"], emb["user"]], 1).to(device_gpu)
pred = mlp(dense_in)

# Fake label & loss
label = torch.ones_like(pred)
loss = F.binary_cross_entropy_with_logits(pred, label)

# Backward & update
optim.zero_grad(set_to_none=True)
loss.backward()
optim.step()

print("Output:", pred.detach().cpu())

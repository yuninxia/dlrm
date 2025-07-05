import torch
import torch.nn as nn
import torchrec
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# 1. EmbeddingBagCollection 放在 CPU
ebc = torchrec.EmbeddingBagCollection(
    device="cpu",
    tables=[
        torchrec.EmbeddingBagConfig(
            name="product_table",
            embedding_dim=16,
            num_embeddings=4096,
            feature_names=["product"],
            pooling=torchrec.PoolingType.SUM,
        ),
        torchrec.EmbeddingBagConfig(
            name="user_table",
            embedding_dim=16,
            num_embeddings=4096,
            feature_names=["user"],
            pooling=torchrec.PoolingType.SUM,
        ),
    ]
)

# 2. Dense 部分（MLP）放在 GPU
class DenseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

dense_model = DenseModel(input_dim=32, hidden_dim=64, output_dim=1).cuda()

# 3. 构造输入
product_jt = torchrec.sparse.jagged_tensor.JaggedTensor(
    values=torch.tensor([1, 2, 1, 5], dtype=torch.long),
    lengths=torch.tensor([3, 1], dtype=torch.long)
)
user_jt = torchrec.sparse.jagged_tensor.JaggedTensor(
    values=torch.tensor([2, 3, 4, 1], dtype=torch.long),
    lengths=torch.tensor([2, 2], dtype=torch.long)
)
kjt = KeyedJaggedTensor.from_jt_dict({"product": product_jt, "user": user_jt})

# 4. 前向推理：Embedding 查表在 CPU，Dense 计算在 GPU
with torch.no_grad():
    # Embedding 查表（CPU）
    emb = ebc(kjt)
    # 拼接所有 embedding，转到 GPU
    emb_cat = torch.cat([emb["product"], emb["user"]], dim=1).cuda()
    # Dense 计算（GPU）
    output = dense_model(emb_cat)
    print("模型输出：", output)

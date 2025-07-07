#!/usr/bin/env python
"""
Production-Scale DLRM for CPU+GPU Performance Characterization
CPU-side: 8 large embedding tables (>1M rows each)   ≈ 2GB+ total
GPU-side: Deep MLPs with feature interactions
Multiple iterations with large batch sizes
Runtime: 30+ seconds for meaningful profiling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrec
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
import time
import numpy as np

# Configuration for meaningful CPU+GPU workload
BATCH_SIZE = 4096           # Maximum batch for >1GB transfer
NUM_ITERATIONS = 50         # Balanced for memory and time
EMBEDDING_DIM = 128         # Larger embedding dimension
MLP_HIDDEN_DIMS = [512, 512, 256, 256, 128]  # Deep MLP
INTERACTION_ARCH = "cat"    # Simple concatenation (more stable)

torch.manual_seed(42)
device_gpu = torch.device("cuda")
device_cpu = torch.device("cpu")

# GPU Optimization Settings for Better H2D Bandwidth
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

print("🚀 Initializing Production-Scale DLRM")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Iterations: {NUM_ITERATIONS}")
print(f"   Embedding Dim: {EMBEDDING_DIM}")
print(f"   MLP Layers: {len(MLP_HIDDEN_DIMS)}")

# 1. Large EmbeddingBagCollection on CPU (realistic industry scale)
print("\n📊 Creating Large Embedding Tables on CPU...")
ebc = torchrec.EmbeddingBagCollection(
    device="cpu",
    tables=[
        # User features (very large)
        torchrec.EmbeddingBagConfig(
            name="user_id_table",
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=2_000_000,    # 2M users
            feature_names=["user_id"],
            pooling=torchrec.PoolingType.SUM,
        ),
        # Item features (large)
        torchrec.EmbeddingBagConfig(
            name="item_id_table", 
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=1_500_000,    # 1.5M items
            feature_names=["item_id"],
            pooling=torchrec.PoolingType.SUM,
        ),
        # Category features (medium)
        torchrec.EmbeddingBagConfig(
            name="category_table",
            embedding_dim=EMBEDDING_DIM//2,
            num_embeddings=500_000,      # 500K categories
            feature_names=["category"],
            pooling=torchrec.PoolingType.SUM,
        ),
        # Brand features
        torchrec.EmbeddingBagConfig(
            name="brand_table",
            embedding_dim=EMBEDDING_DIM//2,
            num_embeddings=200_000,      # 200K brands
            feature_names=["brand"],
            pooling=torchrec.PoolingType.SUM,
        ),
        # Shop features
        torchrec.EmbeddingBagConfig(
            name="shop_table",
            embedding_dim=EMBEDDING_DIM//2,
            num_embeddings=100_000,      # 100K shops
            feature_names=["shop"],
            pooling=torchrec.PoolingType.SUM,
        ),
        # Geography features
        torchrec.EmbeddingBagConfig(
            name="geo_table",
            embedding_dim=EMBEDDING_DIM//4,
            num_embeddings=50_000,       # 50K geo locations
            feature_names=["geo"],
            pooling=torchrec.PoolingType.SUM,
        ),
        # Device features
        torchrec.EmbeddingBagConfig(
            name="device_table",
            embedding_dim=EMBEDDING_DIM//4,
            num_embeddings=10_000,       # 10K device types
            feature_names=["device"],
            pooling=torchrec.PoolingType.SUM,
        ),
        # Time features
        torchrec.EmbeddingBagConfig(
            name="time_table",
            embedding_dim=EMBEDDING_DIM//4,
            num_embeddings=100_000,      # 100K time buckets
            feature_names=["time_bucket"],
            pooling=torchrec.PoolingType.SUM,
        ),
    ]
)

# Calculate total embedding parameters
total_embedding_params = sum([
    2_000_000 * EMBEDDING_DIM,          # user_id
    1_500_000 * EMBEDDING_DIM,          # item_id  
    500_000 * (EMBEDDING_DIM//2),       # category
    200_000 * (EMBEDDING_DIM//2),       # brand
    100_000 * (EMBEDDING_DIM//2),       # shop
    50_000 * (EMBEDDING_DIM//4),        # geo
    10_000 * (EMBEDDING_DIM//4),        # device
    100_000 * (EMBEDDING_DIM//4),       # time
])
print(f"   Total Embedding Parameters: {total_embedding_params:,} ({total_embedding_params*4/1e9:.2f} GB)")

# 2. Complex MLP architectures on GPU
print("\n🧠 Creating Deep MLPs on GPU...")

class BottomMLP(nn.Module):
    """Bottom MLP for dense features"""
    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class FeatureInteraction(nn.Module):
    """Complex feature interaction layer"""
    def __init__(self, sparse_feature_dims: list, interaction_arch: str = "dot"):
        super().__init__()
        self.interaction_arch = interaction_arch
        self.sparse_feature_dims = sparse_feature_dims
        self.num_sparse_features = len(sparse_feature_dims)
        
        if interaction_arch == "dot":
            # Dot product interactions between feature embeddings
            self.num_interactions = self.num_sparse_features * (self.num_sparse_features - 1) // 2
        elif interaction_arch == "cat":
            # Simple concatenation
            self.num_interactions = 0
            
    def forward(self, sparse_features, dense_features):
        if self.interaction_arch == "dot":
            # Split concatenated sparse features back into individual embeddings
            batch_size = sparse_features.size(0)
            feature_embeddings = []
            start_idx = 0
            
            for dim in self.sparse_feature_dims:
                end_idx = start_idx + dim
                feature_embeddings.append(sparse_features[:, start_idx:end_idx])
                start_idx = end_idx
            
            # Compute pairwise interactions
            interactions = []
            for i in range(self.num_sparse_features):
                for j in range(i + 1, self.num_sparse_features):
                    # Element-wise multiplication then sum
                    interaction = (feature_embeddings[i] * feature_embeddings[j]).sum(dim=1, keepdim=True)
                    interactions.append(interaction)
            
            if interactions:
                interaction_tensor = torch.cat(interactions, dim=1)
                # Concatenate: original sparse + interactions + dense
                return torch.cat([
                    sparse_features,
                    interaction_tensor,
                    dense_features
                ], dim=1)
            else:
                return torch.cat([sparse_features, dense_features], dim=1)
        else:
            # Simple concatenation
            return torch.cat([sparse_features, dense_features], dim=1)

class TopMLP(nn.Module):
    """Top MLP for final prediction"""
    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            if i < len(hidden_dims) - 1:  # Not the last layer
                layers.extend([
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2)
                ])
            else:  # Last layer
                layers.append(nn.Sigmoid())
            
            current_dim = hidden_dim
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# Dense feature dimensions (simulating real features)
DENSE_FEATURE_DIM = 256

# Create MLPs
bottom_mlp = BottomMLP(DENSE_FEATURE_DIM, MLP_HIDDEN_DIMS).to(device_gpu)

# Calculate interaction layer input dimension
sparse_feature_dims = [EMBEDDING_DIM if 'user_id' in config.feature_names[0] or 'item_id' in config.feature_names[0] 
                       else EMBEDDING_DIM//2 if 'category' in config.feature_names[0] or 'brand' in config.feature_names[0] or 'shop' in config.feature_names[0]
                       else EMBEDDING_DIM//4 
                       for config in ebc.embedding_bag_configs()]
total_sparse_dim = sum(sparse_feature_dims)

feature_interaction = FeatureInteraction(sparse_feature_dims, INTERACTION_ARCH).to(device_gpu)

# Calculate top MLP input dimension
num_sparse_features = len(sparse_feature_dims)
if INTERACTION_ARCH == "dot":
    interaction_output_dim = total_sparse_dim + (num_sparse_features * (num_sparse_features - 1) // 2) + MLP_HIDDEN_DIMS[-1]
else:
    interaction_output_dim = total_sparse_dim + MLP_HIDDEN_DIMS[-1]

top_mlp = TopMLP(interaction_output_dim, [512, 256, 128, 64, 1]).to(device_gpu)

# Count total GPU parameters
gpu_params = sum(p.numel() for p in bottom_mlp.parameters()) + \
            sum(p.numel() for p in top_mlp.parameters())
print(f"   Total GPU Parameters: {gpu_params:,} ({gpu_params*4/1e6:.2f} MB)")

# 3. Create realistic large batch data
print(f"\n📦 Generating Large Batch Data (size={BATCH_SIZE})...")

def create_large_batch():
    """Create a realistic large batch of data"""
    sparse_data = {}
    feature_configs = [
        ("user_id", 2_000_000, 1, 3),       # 1-3 user features per sample
        ("item_id", 1_500_000, 1, 5),       # 1-5 item features per sample  
        ("category", 500_000, 1, 2),        # 1-2 category features
        ("brand", 200_000, 1, 1),           # exactly 1 brand
        ("shop", 100_000, 1, 1),            # exactly 1 shop
        ("geo", 50_000, 1, 1),              # exactly 1 geo
        ("device", 10_000, 1, 1),           # exactly 1 device
        ("time_bucket", 100_000, 1, 1),     # exactly 1 time bucket
    ]
    
    for feature_name, vocab_size, min_len, max_len in feature_configs:
        # Random sequence lengths for each sample in batch
        lengths = torch.randint(min_len, max_len + 1, (BATCH_SIZE,))
        total_length = lengths.sum().item()
        
        # Random feature values
        values = torch.randint(0, vocab_size, (total_length,), dtype=torch.long)
        
        jt = torchrec.sparse.jagged_tensor.JaggedTensor(
            values=values,
            lengths=lengths
        )
        sparse_data[feature_name] = jt
    
    kjt = KeyedJaggedTensor.from_jt_dict(sparse_data)
    
    # Dense features (user behavior, item stats, etc.)
    dense_features = torch.randn(BATCH_SIZE, DENSE_FEATURE_DIM, device=device_gpu)
    
    # Labels for CTR prediction
    labels = torch.randint(0, 2, (BATCH_SIZE, 1), dtype=torch.float32, device=device_gpu)
    
    return kjt, dense_features, labels

# 4. Setup optimizer
print("\n⚙️  Setting up Optimizer...")
optimizer = torch.optim.AdamW(
    list(ebc.parameters()) + list(bottom_mlp.parameters()) + list(top_mlp.parameters()),
    lr=0.001,
    weight_decay=0.01
)

# Learning rate scheduler for realism
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 5. Training loop for performance characterization
print(f"\n🔥 Starting Training Loop ({NUM_ITERATIONS} iterations)...")
print("=" * 60)

total_samples = 0
start_time = time.time()

for iteration in range(NUM_ITERATIONS):
    iter_start = time.time()
    
    # Create batch data
    kjt, dense_features, labels = create_large_batch()
    total_samples += BATCH_SIZE
    
    # Forward pass
    # 1. CPU: Embedding lookup
    sparse_embeddings = ebc(kjt)  # This stays on CPU
    
    # 2. Move sparse embeddings to GPU and concatenate (different dims)
    sparse_list = []
    for feature_name in ["user_id", "item_id", "category", "brand", "shop", "geo", "device", "time_bucket"]:
        sparse_list.append(sparse_embeddings[feature_name].to(device_gpu))
    # Concatenate all sparse features (different embedding dimensions)
    sparse_features = torch.cat(sparse_list, dim=1)  # [batch_size, total_sparse_dim]
    
    # 3. GPU: Bottom MLP for dense features
    dense_output = bottom_mlp(dense_features)
    
    # 4. GPU: Feature interactions
    interaction_output = feature_interaction(sparse_features, dense_output)
    
    # 5. GPU: Top MLP for final prediction
    predictions = top_mlp(interaction_output)
    
    # 6. Compute loss
    loss = F.binary_cross_entropy(predictions, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()  # This will create D2H traffic for gradients
    optimizer.step()
    
    # Update learning rate
    if iteration % 20 == 0:
        scheduler.step()
    
    iter_time = time.time() - iter_start
    
    # Progress reporting
    if (iteration + 1) % 10 == 0:
        throughput = BATCH_SIZE / iter_time
        print(f"Iter {iteration+1:2d}/{NUM_ITERATIONS}: Loss={loss.item():.4f}, "
              f"Time={iter_time:.3f}s, Throughput={throughput:.1f} samples/s")

total_time = time.time() - start_time
avg_throughput = total_samples / total_time

print("=" * 60)
print(f"🎯 Training Completed!")
print(f"   Total Time: {total_time:.2f} seconds")
print(f"   Total Samples: {total_samples:,}")
print(f"   Average Throughput: {avg_throughput:.1f} samples/s")
print(f"   Final Loss: {loss.item():.4f}")
print("✅ Production-Scale DLRM Workload Completed!")
print(f"💾 Memory Usage - GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("🔬 Ready for HPCToolkit Analysis!")

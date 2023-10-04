# import argparse
# import os.path as osp

# import torch
# import torch.nn.functional as F
# from ogb.graphproppred import Evaluator
# from ogb.graphproppred import PygGraphPropPredDataset as OGBG
# from ogb.graphproppred.mol_encoder import AtomEncoder
# from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# import torch_geometric.transforms as T
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import EGConv, global_mean_pool
# from torch_geometric.typing import WITH_TORCH_SPARSE
# from torch_sparse import SparseTensor


# path = './'
# dataset = OGBG('ogbg-molhiv', path, pre_transform=T.ToSparseTensor())
# evaluator = Evaluator('ogbg-molhiv')

# split_idx = dataset.get_idx_split()
# train_dataset = dataset[split_idx['train']]
# val_dataset = dataset[split_idx['valid']]
# test_dataset = dataset[split_idx['test']]

# train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4,
#                           shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=256)
# test_loader = DataLoader(test_dataset, batch_size=256)

# for data in train_loader:
#     print("data : ", data.x.size())
#     print("adj_t : ",data.adj_t)
#     print("batch : ", data.batch)    

from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
from torch_sparse import SparseTensor

# 예제 입력 데이터 (64, 64, 80)
X = np.random.rand(64, 64, 80)

# Nearest Neighbors 모델 초기화
k_neighbors = 6
knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')

# 데이터를 2D로 변환 (64*64, 80)
X_reshaped = X.reshape(-1, 80)

# 모델 피팅
knn_model.fit(X_reshaped)

# 이웃 인덱스 및 거리 계산
distances, neighbor_indices = knn_model.kneighbors(X_reshaped)
print(neighbor_indices)
# 이웃 인덱스를 이용하여 인접 행렬 생성
adj_matrix = np.zeros((64*64, 64*64))
print(X.shape[0])
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        node_idx = i * X.shape[1] + j
        neighbors = neighbor_indices[node_idx]
        adj_matrix[node_idx, neighbors] = 1

# 최종 인접 행렬

adj_matrix = torch.Tensor(adj_matrix)
edges = adj_matrix.nonzero(as_tuple=True)
adj_matrix = SparseTensor(row = edges[0], col = edges[1],sparse_sizes=(4096, 4096))
print(adj_matrix)


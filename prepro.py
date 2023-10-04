import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import csv
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torch_sparse import SparseTensor

# def adjacency_matrix(X, k_neighbors):
#     knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    
#     knn_model.fit(X)  # EEG 데이터를 모델에 피팅합니다.

#     # KNN 그래프를 얻기 위해 kneighbors 메서드 사용
#     distances, indices = knn_model.kneighbors(X)
        
#     all_indices = torch.tensor(indices)

#     adjacency_matrix = torch.zeros((64, 64), dtype=torch.float32)
    
#     for i, neighbors in enumerate(all_indices):
#         adjacency_matrix[i, neighbors] = 1.0

#     return adjacency_matrix
# def adjacency_matrix(X, k_neighbors):
#     batch_size, num_channels, num_inputs = X.shape

#     adjacency_matrix = torch.zeros((batch_size, num_channels, num_channels), dtype=torch.float32)
#     for b in range(batch_size):  
#         knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
#         # 데이터를 2D로 reshape
#         X_reshaped = X[b].view(num_channels, num_inputs)
#         knn_model.fit(X_reshaped)

#         # KNN 그래프를 얻기 위해 kneighbors 메서드 사용
#         distances, indices = knn_model.kneighbors(X_reshaped)
#         # 인접 행렬을 초기화
       
#         for c in range(num_channels):
#             neighbors = indices[c]  # 각 채널에 대한 이웃 인덱스 가져오기
#             adjacency_matrix[b, c, neighbors] = 1.0  # (i, j) 위치에 1.0 설정
    
#     D_inv_sqrt = torch.diag_embed(torch.pow(adjacency_matrix.sum(dim=-1), -0.5))

#     D_adj_matrix = D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt
#     return D_adj_matrix
def adjacency_matrix(X, k_neighbors):
    batch_size, num_channels, num_inputs = X.shape
    print("num : ", num_inputs)
    knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    X_reshaped = X.reshape(-1, num_inputs)
    knn_model.fit(X_reshaped)
    distances, neighbor_indices = knn_model.kneighbors(X_reshaped)

    adj_matrix = np.zeros((batch_size*num_channels, batch_size*num_channels))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            node_idx = i * X.shape[1] + j
            neighbors = neighbor_indices[node_idx]
            adj_matrix[node_idx, neighbors] = 1
    adj_matrix = torch.Tensor(adj_matrix)
    edges = adj_matrix.nonzero(as_tuple=True)
    adj_matrix = SparseTensor(row = edges[0], col = edges[1],sparse_sizes=(batch_size*num_channels, batch_size*num_channels))

    return adj_matrix

def split_time(input_data):

    min_value = input_data.min()
    max_value = input_data.max()
    input_data = (input_data - min_value) / (max_value - min_value)
    # 잘라낼 데이터 길이 (0.5초 데이터)
    segment_length = 80

    # 데이터를 잘라내고 리스트에 저장
    segments = []
    for i in range(0, input_data.shape[2], segment_length):
        segment = input_data[:,:, i:i+segment_length]
        segments.append(segment)

    return torch.stack(segments, dim=0)



class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        return img, label

if __name__ == "__main__":
    X = torch.rand(64, 64, 80)
    # 인접 행렬 생성
    k_neighbors = 6  # KNN에서 이웃의 개수
    adj_matrix = adjacency_matrix(X, k_neighbors)

    print(adj_matrix.size())


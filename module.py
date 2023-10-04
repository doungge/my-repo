import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import snntorch as snn
from sklearn.neighbors import NearestNeighbors
from snntorch import surrogate
from torch_geometric.nn import EGConv
from torch_sparse import SparseTensor


np.random.seed(31415)
torch.manual_seed(4)

class spike_encodig(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding = nn.Parameter(torch.Tensor(80,80))
        self.encoding.data = self.encoding.data.type(torch.float64)
        spike_grad_lstm = snn.surrogate.fast_sigmoid(slope=25)
        self.lif1 = snn.Leaky(beta=0.6, threshold=0.3, spike_grad = spike_grad_lstm)
        
        nn.init.xavier_uniform_(self.encoding.data, gain=1.414)
    def forward(self, x, mem):
        # mem1 = self.lif1.init_leaky()

        cur1 =  x @ self.encoding
        spk1, mem1 = self.lif1(cur1, mem)
        
        return spk1, mem1

# class MGLayer(nn.Module):

#     def __init__(self, c_in, c_out, num_heads=8, num_bases=4, concat_heads = True):
#         super().__init__()
#         self.num_heads = num_heads
#         self.num_bases = num_bases
#         self.concat_heads = concat_heads
#         if self.concat_heads:
#             assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
#             c_out = c_out // num_heads

#         # Sub-modules and parameters needed in the layer
        
#         # self.a = nn.Parameter(torch.Tensor(c_in, c_out / num_heads)) # One per head
#         self.w = nn.Parameter(torch.Tensor(self.num_heads, self.num_bases, 1, 64)) 
#         self.Wb = nn.Parameter(torch.Tensor(self.num_bases, c_in, 8))

#         nn.init.xavier_uniform_(self.w.data, gain=1.414)
#         nn.init.xavier_uniform_(self.Wb.data, gain=1.414)
#     def forward(self, x, adj_matrix, print_attn_probs=False):
#         print(self.Wb.size())
#         batch_size, num_nodes = x.size(0), x.size(1)
#         x = x.view(batch_size*num_nodes, -1)
#         all_node_feature = []
#         # Apply linear layer and sort nodes by head
        
#         D_inv_sqrt = torch.diag_embed(torch.pow(adj_matrix.sum(dim=-1), -0.5))
#         D_adj_matrix = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
#         for nh in range(self.num_heads):
#             sum_XW = torch.zeros(1, 8).cuda("cuda:0")
#             for nb in range(self.num_bases):
#                 Dx = D_adj_matrix @ x
                
#                 xWb = Dx @ self.Wb[nb,:,:]
#                 test = self.w[nh, nb, :, :] @ xWb
                
#                 sum_XW += test

#             all_node_feature.append(sum_XW)             
#         return torch.stack(all_node_feature).reshape(batch_size, num_nodes)



class MGLayer(nn.Module):

    def __init__(self, c_in, c_out, num_heads=8, num_bases=4, concat_heads = True):
        super().__init__()
        self.num_heads = num_heads
        self.num_bases = num_bases
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        
        # self.a = nn.Parameter(torch.Tensor(c_in, c_out / num_heads)) # One per head
        self.w = nn.Parameter(torch.Tensor(self.num_heads, self.num_bases, 64)) 
        self.Wb = nn.Parameter(torch.Tensor(self.num_bases, c_in, 8))

        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wb.data, gain=1.414)
    def forward(self, x, adj_matrix, print_attn_probs=False):
        batch_size, num_nodes = x.size(0), x.size(1)
        # x = x.view(batch_size*num_nodes, -1)
        all_node_feature = []
        # Apply linear layer and sort nodes by head

        for nh in range(self.num_heads):
            sum_XW = torch.zeros(batch_size, 8).cuda("cuda:1")
            for nb in range(self.num_bases):
                Dx = torch.einsum('bii,bij->bij',adj_matrix, x)
                xWb = torch.einsum('bij,jk->bik', Dx, self.Wb[nb,:,:])

                test = torch.einsum('i,bik->bk',self.w[nh, nb, :], xWb)              
                sum_XW += test

            all_node_feature.append(sum_XW)             
        return torch.stack(all_node_feature).reshape(batch_size, num_nodes)

class GraphModule(nn.Module):
    def __init__(self):
        super().__init__()

        num_inputs = 64
        num_hidden1 = 128
        spike_grad_lstm = snn.surrogate.fast_sigmoid(slope=25)

        # initialize layers
        # self.mglayer = MGLayer(c_in=80, c_out=64, num_heads=8, num_bases=4)
        self.egclayer = EGConv(80, 64, aggregators=['symnorm'],
                       num_heads=8, num_bases=4)
        self.lif1 = snn.Leaky(beta=0.6, threshold=0.3,spike_grad=spike_grad_lstm)
        

    def forward(self, x, mem, adj_matrix):
        # Initialize hidden states and outputs at t=0
        adj_matrix = 
        # mem1 = self.lif1.init_leaky()
        print("x1 : ",x.size())
        # cur1 = self.mglayer(x, adj_matrix)
        cur1 = self.egclayer(x, adj_matrix)
        print("cur1 : ",cur1.size())
        spk1, mem1 = self.lif1(cur1, mem)

        return spk1, mem1


class LSTMCell1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        hx, cx = state
        threshold = 0.3
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.erf(ingate)
        forgetgate = torch.erf(forgetgate)
        cellgate = torch.floor(torch.clamp((cellgate + threshold) / (1 - threshold), min=0, max=1))
        outgate = torch.floor(torch.clamp((outgate + threshold) / (1 - threshold), min=0, max=1))
        cy = torch.floor(torch.clamp(((forgetgate * cx) + (ingate * cellgate)+threshold)/(1- threshold), min=0, max=1))
        hy = outgate * cy
        return hy, cy

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        num_inputs = 64
        num_hidden1 = 128

        spike_grad_lstm = snn.surrogate.fast_sigmoid(slope=25)

        # initialize layers
        self.slstm1 = snn.SLSTM(num_inputs, num_hidden1,threshold=0.3,
        spike_grad=spike_grad_lstm)
        self.slstm1.lstm_cell = LSTMCell1(num_inputs, num_hidden1)

    def forward(self, spk, syn, mem):

        # Initialize hidden states and outputs at t=0
        # _, mem1 = self.slstm1.init_slstm()
        spk1, syn1, mem1 = self.slstm1(spk, syn, mem)

        return spk1, syn1, mem1

class FC(nn.Module):
    def __init__(self):
        super().__init__()

        num_inputs = 128
        num_hidden = 256
        num_output = 4
        spike_grad = snn.surrogate.fast_sigmoid(slope=25)


        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=0.6, threshold=0.3,spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.lif2 = snn.Leaky(beta=0.6, threshold=0.3,spike_grad=spike_grad, output=True)


    def forward(self, x, mem1, mem2):
        # mem1 = self.lif1.init_leaky()
        # mem2 = self.lif2.init_leaky()
        
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        # print("spk value",spk2)
        
        return spk2, mem1, mem2


class SGLnet(nn.Module):
    def __init__(self, time_step):
        super().__init__()
        self.se = spike_encodig()
        self.mhg = GraphModule()
        self.ls = LSTM()
        self.fc = FC()
        self.ts = time_step
        # self.spk_output = []
        # self.mem_output = []

        
    def forward(self, x, adj_matrix):
        mem = self.se.lif1.init_leaky()
        gc_mem = self.mhg.lif1.init_leaky()
        fc1_mem = self.fc.lif1.init_leaky()
        fc2_mem = self.fc.lif2.init_leaky()
        ls_syn, ls_mem = self.ls.slstm1.init_slstm()
        spk_output=[]
        mem_output=[]
        # mem_rec = []
        # spk_out = []

        for i in range(self.ts):
            spk, mem = self.se(x[i], mem)
            gc_spk, gc_mem = self.mhg(spk.view(-1,64,80), gc_mem, adj_matrix)
            ls_spk, ls_syn, ls_mem = self.ls(gc_spk,ls_syn, ls_mem)
            fc_spk, fc1_mem, fc2_mem = self.fc(ls_spk, fc1_mem, fc2_mem)
            # print("final_size",fc_spk.size())
            spk_output.append(fc_spk)
            
            mem_output.append(fc2_mem)

        spk_rec = torch.stack(spk_output)
        mem_rec = torch.stack(mem_output)

        # total_sum = torch.stack(self.output).sum(dim=0)
        # total_mean = total_sum / 8

        
        return spk_rec, mem_rec

    def update(self, data):    
        inputs, targets = data





if __name__=="__main__":
    def adjacency_matrix(X, k_neighbors):
        knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        
        knn_model.fit(X)  # EEG 데이터를 모델에 피팅합니다.

        # KNN 그래프를 얻기 위해 kneighbors 메서드 사용
        distances, indices = knn_model.kneighbors(X)
            
        all_indices = torch.tensor(indices)

        adjacency_matrix = torch.zeros((64, 64), dtype=torch.float32)
        
        for i, neighbors in enumerate(all_indices):
            adjacency_matrix[i, neighbors] = 1.0

        return adjacency_matrix
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

    np.random.seed(31415)
    torch.manual_seed(4)

    input_data = torch.randn(1, 64, 640)

    # min_value = input_data.min()
    # max_value = input_data.max()
    # input_data = (input_data - min_value) / (max_value - min_value)
    # # 잘라낼 데이터 길이 (0.5초 데이터)
    # segment_length = 80

    # # 데이터를 잘라내고 리스트에 저장
    # segments = []
    # for i in range(0, input_data.shape[2], segment_length):
    #     segment = input_data[:,:, i:i+segment_length]
    #     segments.append(segment)
    # segments_tensor = torch.stack(segments, dim=0)
    segments_tensor = split_time(input_data)
    adj_matrix = adjacency_matrix(input_data[0],6)
    # 리스트를 텐서로 변환
    

    model = SGLnet(8)
    print(model.modules)
    model(segments_tensor, adj_matrix)
    
    
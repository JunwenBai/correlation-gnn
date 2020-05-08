import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #print("input:", input.shape)
        #print("adj:", adj.shape)
        #print("adj[0,0]:", adj)
        #print("weight:", self.weight.shape)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        #print("support:", support.shape)
        #print("output:", output.shape)
        #exit()
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    '''
    simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    '''
    def __init__(self, in_dim, out_dim, alpha, dropout, nonlinear=False, use_cuda=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.nonlinear = nonlinear
        self.use_cuda = use_cuda

        self.W = nn.Parameter(torch.zeros(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W, gain=1.414)
        self.a = nn.Parameter(torch.zeros(2*out_dim, 1))
        nn.init.xavier_uniform_(self.a, gain=1.414)

    @staticmethod
    def getMask(global_feature, nodes, neighbors_list):
        neighbors_list = [(neighbors | set([nodes[i]])) for i, neighbors in enumerate(neighbors_list)]
        unique_nodes_list = list(set.union(*neighbors_list))
        unique_nodes_dict = {node:n for n, node in enumerate(unique_nodes_list)}

        mask = torch.zeros(len(nodes), len(unique_nodes_list))
        row_indices = [i for i, neighbors in enumerate(neighbors_list) for node in neighbors]
        col_indices = [unique_nodes_dict[node] for neighbors in neighbors_list for node in neighbors]
        mask[row_indices, col_indices] = 1

        return mask, unique_nodes_list

    def meanAggregate(self, global_feature, nodes, neighbors_list):
        mask, unique_nodes_list = self.getMask(global_feature, nodes, neighbors_list)
        if self.use_cuda:
            mask = mask.cuda()
        neighbor_num = mask.sum(1, keepdim=True)
        mask = mask.div(neighbor_num)

        neighbors_feature = global_feature[unique_nodes_list]
        return torch.matmul(mask, neighbors_feature)

    def forward(self, global_feature, nodes, neighbors_list):
        mask, unique_nodes_list = self.getMask(global_feature, nodes, neighbors_list)
        if self.use_cuda:
            mask = mask.cuda()

        nodes_feature = torch.matmul(global_feature[nodes], self.W) ## B x out_dim
        neighbors_feature = torch.matmul(global_feature[unique_nodes_list], self.W) ## N x out_dim
        B = nodes_feature.size(0)
        N = neighbors_feature.size(0)

        concate_feature = torch.cat((nodes_feature.repeat(1, N).view(B*N, -1), neighbors_feature.repeat(B, 1)), dim = 1) ## BN x 2out_dim
        e = torch.matmul(concate_feature, self.a).squeeze(1).view(B, N)
        # residual_feature = nodes_feature.repeat(1, N).view(B*N, -1) - neighbors_feature.repeat(B, 1) ## BN x out_dim
        # e = torch.matmul(residual_feature, self.a).squeeze(1).view(B, N)
        e = self.leakyrelu(e)
        neg_inf = -9e15 * torch.ones_like(e)
        e = torch.where(mask>0, e, neg_inf)
        attention = F.softmax(e, dim=1) ## B x N

        attention = F.dropout(attention, self.dropout, training=self.training)
        out_feature = torch.matmul(attention, neighbors_feature)

        out_feature = F.normalize(out_feature, p=2, dim=1)

        if self.nonlinear:
            out_feature = F.elu(out_feature)

        return out_feature

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'

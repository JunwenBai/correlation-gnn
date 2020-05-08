import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttentionLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)
        return x

class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, class_num, alpha, dropout, nheads, use_cuda):
        super(GAT, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.class_num = class_num
        self.alpha = alpha
        self.dropout = dropout
        self.nheads = nheads
        self.use_cuda = use_cuda

        self.attentions = [GraphAttentionLayer(self.in_dim, self.hid_dim, self.alpha, self.dropout, nonlinear=True, use_cuda=self.use_cuda) for _ in range(self.nheads)]

        for k in range(self.nheads):
            self.add_module('attention_' + str(k), self.attentions[k])

        ## we change the second-layer attention to fc layers.
        self.classifier = nn.Sequential(
                        nn.Linear(self.hid_dim, self.class_num),
        )

    def forward(self, global_feature, nodes, neighbors_list):
        # global_feature = F.dropout(global_feature, self.dropout, training=self.training)
        # new_feature = torch.cat([atten(global_feature, nodes, neighbors_list) for atten in self.attentions], dim=1)
        new_feature = torch.mean(torch.cat([atten(global_feature, nodes, neighbors_list).view(1, -1) for atten in self.attentions], dim=0), dim=0).view(len(nodes), -1)
        # new_feature = F.dropout(new_feature, self.dropout, training=self.training)
        logit = self.classifier(new_feature)
        return new_feature, logit

import torch as th
from torch import nn
import torch.nn.functional as F
import dgl.function as fn

class SGConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 n_hid=32,
                 cached=False,
                 bias=True,
                 norm=None):
        super(SGConv, self).__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self.n_hid = n_hid
        #self.fc1 = nn.Linear(in_feats, n_hid, bias=bias)
        self.fc1 = nn.Linear(in_feats, out_feats, bias=bias)
        self.fc2 = nn.Linear(n_hid, out_feats, bias=bias)
        
        #self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)


    def forward(self, graph, feat):
        graph = graph.local_var()
        if self._cached_h is not None:
            feat = self._cached_h
        else:
            # compute normalization
            degs = graph.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(feat.device).unsqueeze(1)
            # compute (D^-1 A^k D)^k X
            for _ in range(self._k):
                feat = feat * norm
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                feat = feat * norm

            if self.norm is not None:
                feat = self.norm(feat)

            # cache feature
            if self._cached:
                self._cached_h = feat
        x = self.fc1(feat)
        #x = self.fc1(F.elu(feat))
        #x = self.fc2(F.elu(x))
        return x

import dgl
import sys
sys.path.append(".")
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
import tqdm
import traceback
import scipy.sparse as sp
from sklearn.metrics import r2_score
from utils import lp_refine, R2, sparse_mx_to_torch_sparse_tensor, normalize
from gpytorch import inv_matmul, logdet
from gpytorch.utils import linear_cg
from torch import matmul

#### Neighbor sampler

class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y
        return y

def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.
    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])

def compute_r2(pred, labels):
    """
    Compute the R2 of prediction given the labels.
    """
    #return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)
    return r2_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())

def evaluate(model, g, inputs, labels, val_mask, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the R2 for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_r2(pred[val_mask], labels[val_mask])

def evaluate_test(model, g, inputs, labels, test_mask, batch_size, device, lp_dict, coeffs, meta):
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device).view(-1)

    output = pred.cuda()
    labels = labels.cuda()
    idx_test = lp_dict['idx_test']
    idx_train = lp_dict['idx_train']
    adj = lp_dict['adj']

    labels, output, adj = labels.cpu(), output.cpu(), adj.cpu()
    loss = F.mse_loss(output[idx_test].squeeze(), labels[idx_test].squeeze())
    r2_test = compute_r2(output[test_mask], labels[test_mask])
    lp_output = lp_refine(idx_test, idx_train, labels, output, adj, torch.tanh(coeffs[0]).item(), torch.exp(coeffs[1]).item())
    lp_r2_test = compute_r2(lp_output, labels[idx_test])
    lp_output_raw_conv = lp_refine(idx_test, idx_train, labels, output, adj)
    lp_r2_test_raw_conv = R2(lp_output_raw_conv, labels[idx_test])

    print("------------")
    print("election year {}".format(meta))
    print("loss:", loss.item())
    print("raw_r2:", r2_test)
    print("refined_r2:", lp_r2_test)
    print("refined_r2_raw_conv:", lp_r2_test_raw_conv)
    print("------------")

    model.train()

    return lp_r2_test

def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def setdiff(n, idx):
    idx = idx.cpu().detach().numpy()
    cp_idx = np.setdiff1d(np.arange(n), idx)
    return cp_idx

def loss_fcn(output, labels, idx, S, coeffs, device, add_logdet):
    rL = labels - output
    S = S.to_dense()
    Gamma = (torch.eye(S.size(0)).to(device) - torch.tanh(coeffs[0]) * S.to(device)) * torch.exp(coeffs[1])
    cp_idx = setdiff(len(S), idx)

    loss1 = rL.dot(matmul(Gamma[idx, :][:, idx], rL) - matmul(Gamma[idx, :][:, cp_idx], inv_matmul(Gamma[cp_idx, :][:, cp_idx], matmul(Gamma[cp_idx, :][:, idx], rL))))
    loss2 = 0.
    if add_logdet: loss2 = logdet(Gamma) - logdet(Gamma[cp_idx, :][:, cp_idx])
    l = loss1 - loss2
    return l/len(idx)

#### Entry point
def run(args, device, data):
    # Unpack data
    train_mask, val_mask, test_mask, in_feats, labels, ind_labels, n_classes, g, ind_g, lp_dict = data

    train_nid = th.LongTensor(np.nonzero(train_mask)[0])
    val_nid = th.LongTensor(np.nonzero(val_mask)[0])
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    coeffs = Variable(torch.FloatTensor([1., 3.0]).to(device) , requires_grad=True)
    coeffs_optimizer = optim.SGD([coeffs], lr=1e-1, momentum=0.0)

    # Training loop
    avg = 0
    iter_tput = []
    steps_per_epoch = len(dataloader)
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, blocks in enumerate(dataloader):
            tic_step = time.time()

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device)
            # Compute loss and prediction
            model.train()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred.squeeze(), batch_labels.squeeze(), seeds, lp_dict['adj'], coeffs, device, False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % (steps_per_epoch//2) == 0:
                model.train()
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred.squeeze(), batch_labels.squeeze(), seeds, lp_dict['adj'], coeffs, device, True)
                coeffs_optimizer.zero_grad()
                loss.backward()
                coeffs_optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                r2 = compute_r2(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                #print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train R2 {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(epoch, step, loss.item(), r2.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train R2 {:.4f} | alpha: {:.4f} | beta: {:.4f}'.format(epoch, step, loss.item(), r2.item(), th.tanh(coeffs[0]).item(), th.exp(coeffs[1]).item()))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_r2 = evaluate(model, g, g.ndata['features'], labels, val_mask, args.batch_size, device)
            print('Eval R2: {:.4f}'.format(eval_r2))
            
    evaluate_test(model, g, g.ndata['features'], labels, test_mask, args.batch_size, device, lp_dict, coeffs, "2012")
    evaluate_test(model, ind_g, ind_g.ndata['features'], ind_labels, test_mask, args.batch_size, device, lp_dict, coeffs, "2016")

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=500)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--seed', type=int, default=19940423)
    argparser.add_argument('--fan-out', type=str, default='25,25')
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    args = argparser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
        torch.cuda.manual_seed(args.seed)
    else:
        device = th.device('cpu')

    path = './data/county/election/2012'
    adj = np.load(path+"/A.npy")
    labels = np.load(path+"/labels.npy")
    features = np.load(path+"/feats.npy")
    idx_train = np.load(path+"/train_idx.npy")-1
    idx_val = np.load(path+"/val_idx.npy")-1
    idx_test = np.load(path+"/test_idx.npy")-1
    n = len(adj)
    train_mask = np.zeros(n).astype(bool)
    train_mask[idx_train] = True
    val_mask = np.zeros(n).astype(bool)
    val_mask[idx_val] = True
    test_mask = np.zeros(n).astype(bool)
    test_mask[idx_test] = True
    in_feats = features.shape[1]
    labels = th.FloatTensor(labels)
    n_classes = 1

    sp_adj = sp.coo_matrix(adj)
    g = dgl.graph((th.LongTensor(sp_adj.row), th.LongTensor(sp_adj.col)))
    g.ndata['features'] = th.FloatTensor(features)
    prepare_mp(g)
    lp_dict = {'idx_test': th.LongTensor(idx_test), 'idx_train': th.LongTensor(idx_train), 'adj': sparse_mx_to_torch_sparse_tensor(normalize(sp_adj.astype(float)))}

    ind_path = './data/county/election/2016'
    ind_features = np.load(ind_path+"/feats.npy")
    ind_labels = np.load(ind_path+"/labels.npy")
    ind_labels = th.FloatTensor(ind_labels)
    ind_g = dgl.graph((th.LongTensor(sp_adj.row), th.LongTensor(sp_adj.col)))
    ind_g.ndata['features'] = th.FloatTensor(ind_features)
    prepare_mp(ind_g)

    # Pack data
    data = train_mask, val_mask, test_mask, in_feats, labels, ind_labels, n_classes, g, ind_g, lp_dict

    run(args, device, data)

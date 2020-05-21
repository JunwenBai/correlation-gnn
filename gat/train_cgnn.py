"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from gpytorch import inv_matmul, logdet
from gpytorch.utils import linear_cg
from torch import matmul

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from gat import GAT
from utils import EarlyStopping
import scipy.sparse as sp
from sklearn.metrics import r2_score
from utils import sparse_mx_to_torch_sparse_tensor, normalize, lp_refine

def compute_r2(pred, labels):
    return r2_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        pred = model(features)
        pred = pred[mask]
        labels = labels[mask]
        return compute_r2(pred, labels)

def evaluate_test(model, features, labels, test_mask, lp_dict, coeffs, meta="2012"):
    model.eval()
    with torch.no_grad():
        output = model(features).squeeze()

    output = output.cuda()
    labels = labels.cuda()
    idx_test = lp_dict['idx_test']
    idx_train = lp_dict['idx_train']
    adj = sparse_mx_to_torch_sparse_tensor(normalize(lp_dict['sp_adj']))

    labels, output, adj = labels.cpu(), output.cpu(), adj.cpu()
    loss = F.mse_loss(output[idx_test].squeeze(), labels[idx_test].squeeze())
    r2_test = compute_r2(output[idx_test], labels[idx_test])
    lp_output = lp_refine(idx_test, idx_train, labels, output, adj, torch.tanh(coeffs[0]).item(), torch.exp(coeffs[1]).item())
    lp_r2_test = compute_r2(lp_output, labels[idx_test])
    lp_output_raw_conv = lp_refine(idx_test, idx_train, labels, output, adj)
    lp_r2_test_raw_conv = compute_r2(lp_output_raw_conv, labels[idx_test])

    print("------------")
    print("election year {}".format(meta))
    print("loss:", loss.item())
    print("raw_r2:", r2_test)
    print("refined_r2:", lp_r2_test)
    print("refined_r2_raw_conv:", lp_r2_test_raw_conv)
    print("------------")


def load_cls_data(args):
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    n_classes = data.num_labels

    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)

    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    row = g.edges()[0]
    col = g.edges()[1]
    g = dgl.graph((row, col))

    return g, features, labels, n_classes, train_mask, val_mask, test_mask

def load_reg_data(args):
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
    n_classes = 1
    sp_adj = sp.coo_matrix(adj)
    g = dgl.graph((torch.LongTensor(sp_adj.row), torch.LongTensor(sp_adj.col)))
    lp_dict = {'idx_test': torch.LongTensor(idx_test), 'idx_train': torch.LongTensor(idx_train), 'sp_adj': sp_adj.astype(float), 'adj':sparse_mx_to_torch_sparse_tensor(normalize(sp_adj.astype(float)))}

    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    path = './data/county/election/2016'
    ind_features = torch.FloatTensor(np.load(path+"/feats.npy"))
    ind_labels = torch.FloatTensor(np.load(path+"/labels.npy"))

    return g, features, labels, n_classes, train_mask, val_mask, test_mask, lp_dict, ind_features, ind_labels

def loss_fcn(output, labels, idx, S, coeffs, add_logdet):
    output, labels = output.squeeze(), labels.squeeze()
    rL = labels - output
    S = S.to_dense()
    Gamma = (torch.eye(S.size(0)).cuda() - torch.tanh(coeffs[0]) * S.cuda()) * torch.exp(coeffs[1])
    cp_idx = setdiff(len(S), idx)

    loss1 = rL.dot(matmul(Gamma[idx, :][:, idx], rL) - matmul(Gamma[idx, :][:, cp_idx], inv_matmul(Gamma[cp_idx, :][:, cp_idx], matmul(Gamma[cp_idx, :][:, idx], rL))))

    loss2 = torch.Tensor([0.]).cuda()
    if add_logdet: loss2 = logdet(Gamma) - logdet(Gamma[cp_idx, :][:, cp_idx])
    l = loss1 - loss2
    return l/len(idx)

def setdiff(n, idx):
    idx = idx.cpu().detach().numpy()
    cp_idx = np.setdiff1d(np.arange(n), idx)
    return cp_idx

def main(args):
    # load and preprocess dataset
    g, features, labels, n_classes, train_mask, val_mask, test_mask, lp_dict, ind_features, ind_labels = load_reg_data(args)
    num_feats = features.shape[1]
    n_edges = g.number_of_edges()

    print("""----Data statistics------'
      #use cuda: %d
      #Edges %d
      #Classes %d 
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (args.gpu, n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))
    
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        ind_features = ind_features.cuda()
        labels = labels.cuda()
        ind_labels = ind_labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual,
                args.bias)
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    coeffs = Variable(torch.FloatTensor([1., 3.0]).cuda() if cuda else torch.FloatTensor([1., 3.0]) , requires_grad=True)
    coeffs_optimizer = torch.optim.SGD([coeffs], lr=1e-1, momentum=0.0)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        pred = model(features)
        loss = loss_fcn(pred[train_mask], labels[train_mask], lp_dict['idx_train'], lp_dict['adj'], coeffs, False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if epoch % 10 == 0:
            model.train()
            pred = model(features)
            loss = loss_fcn(pred[train_mask], labels[train_mask], lp_dict['idx_train'], lp_dict['adj'], coeffs, True)
            train_r2 = compute_r2(pred[train_mask], labels[train_mask])
            coeffs_optimizer.zero_grad()
            loss.backward()
            coeffs_optimizer.step()

        if args.fastmode:
            val_r2 = compute_r2(pred[val_mask], labels[val_mask])
        else:
            val_r2 = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_r2, model):
                    break

        if epoch > 3:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainR2 {:.4f} |"
              " Val R2 {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_r2,
                     val_r2, n_edges / np.mean(dur) / 1000))

    print()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    evaluate_test(model, features, labels, test_mask, lp_dict, coeffs, meta="2012")
    evaluate_test(model, ind_features, ind_labels, test_mask, lp_dict, coeffs, meta="2016")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=32,
                        help="number of hidden units")
    parser.add_argument("--seed", type=int, default=19940423,
                        help="random seed")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--bias', action='store_true', default=False,
                        help="whether to add Dense layer bias")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)

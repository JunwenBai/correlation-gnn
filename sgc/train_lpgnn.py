"""
This code was modified from the GCN implementation in DGL examples.
Simplifying Graph Convolutional Networks
Paper: https://arxiv.org/abs/1902.07153
Code: https://github.com/Tiiiger/SGC
SGC implementation in DGL.
"""
import argparse, time, math
import numpy as np
from sklearn.metrics import r2_score
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
#from dgl.nn.pytorch.conv import SGConv
from sgconv import SGConv
from utils import normalize, sparse_mx_to_torch_sparse_tensor, lp_refine

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        pred = model(g, features)[mask] # only compute the evaluation set
        labels = labels[mask]
        return compute_r2(pred, labels)

def evaluate_test(model, g, inputs, labels, test_mask, lp_dict, meta):
    model.eval()
    with torch.no_grad():
        pred = model(g, inputs).squeeze()

    output = pred.cuda()
    labels = labels.cuda()
    idx_test = lp_dict['idx_test']
    idx_train = lp_dict['idx_train']
    adj = sparse_mx_to_torch_sparse_tensor(normalize(lp_dict['sp_adj']))
    #print(adj.to_dense()[np.arange(100), np.arange(100)+1])

    labels, output, adj = labels.cpu(), output.cpu(), adj.cpu()
    loss = F.mse_loss(output[idx_test].squeeze(), labels[idx_test].squeeze())
    r2_test = compute_r2(output[idx_test], labels[idx_test])
    lp_output = lp_refine(idx_test, idx_train, labels, output, adj)
    lp_r2_test = compute_r2(lp_output, labels[idx_test])

    print("------------")
    print("election year {}".format(meta))
    print("loss:", loss.item())
    print("raw_r2:", r2_test)
    print("refined_r2:", lp_r2_test)
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

    g = DGLGraph(data.graph)
    g.add_edges(g.nodes(), g.nodes())

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
    lp_dict = {'idx_test': torch.LongTensor(idx_test), 'idx_train': torch.LongTensor(idx_train), 'sp_adj': sp_adj.astype(float)}

    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    path = './data/county/election/2016'
    ind_features = torch.FloatTensor(np.load(path+"/feats.npy"))
    ind_labels = torch.FloatTensor(np.load(path+"/labels.npy"))

    return g, features, labels, n_classes, train_mask, val_mask, test_mask, lp_dict, ind_features, ind_labels

def loss_fcn(pred, labels):
    pred, labels = pred.squeeze(), labels.squeeze()
    return F.mse_loss(pred, labels)

def compute_r2(pred, labels):
    pred, labels = pred.squeeze(), labels.squeeze()
    return r2_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())

def main(args):
    # load and preprocess dataset
    g, features, labels, n_classes, train_mask, val_mask, test_mask, lp_dict, ind_features, ind_labels = load_reg_data(args)
    n_edges = g.number_of_edges()
    in_feats = features.shape[1]
    
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
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

    # create SGC model
    model = SGConv(in_feats,
                   n_classes,
                   k=2,
                   n_hid=32,
                   cached=True,
                   bias=args.bias)

    if cuda: model.cuda()
    
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        pred = model(g, features) # only compute the train set
        loss = loss_fcn(pred[train_mask], labels[train_mask])
        train_r2 = compute_r2(pred[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        r2 = evaluate(model, g, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | R2 {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             r2, n_edges / np.mean(dur) / 1000))

    print()
    evaluate_test(model, g, features, labels, test_mask, lp_dict, "2012")
    evaluate_test(model, g, ind_features, ind_labels, test_mask, lp_dict, "2016")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGC')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.2,
            help="learning rate")
    parser.add_argument("--bias", action='store_true', default=False,
            help="flag to use bias")
    parser.add_argument("--n-epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=5e-6,
            help="Weight for L2 loss")
    parser.add_argument('--seed', type=int, default=19940423, help='Random seed.')
    args = parser.parse_args()

    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)

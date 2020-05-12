from __future__ import division
from __future__ import print_function

import sys
sys.path.append(".")

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#from pygcn.utils import load_data, R2
#from pygcn.models import GCN
from utils import load_data, R2, load_jj_data, lp_refine
from models import GCN, GAT
from torch import matmul
from gpytorch import inv_matmul, logdet
from gpytorch.utils import linear_cg

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--use_gcn', action='store_true', default=False,
                    help='use chebynet')
parser.add_argument('--seed', type=int, default=19940423, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--nb_heads', type=int, default=4, 
                    help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, 
                    help='Alpha for the leaky_relu.')
parser.add_argument('--batch_size', type=int, default=256, 
                    help='batch size.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("use gpu:", args.cuda)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
path = "./data/county/election/2012"
adj, features, labels, idx_train, idx_val, idx_test = load_jj_data(path)
ind_path = "./data/county/election/2016"
_, ind_features, ind_labels, _, _, _ = load_jj_data(ind_path, load_partial=True)

print("adj:", adj.shape)
print("features:", features.shape)
print("labels:", labels.shape, torch.max(labels), torch.min(labels))
print("idx_train:", idx_train.shape, torch.max(idx_train), torch.min(idx_train))
print("idx_val:", idx_val.shape, torch.max(idx_val), torch.min(idx_val))
print("idx_test:", idx_test.shape, torch.max(idx_test), torch.min(idx_test))
print("n_hid:", args.hidden)
print()

idx_train_lst = []
for i in range(args.epochs):
    perm = torch.randperm(len(idx_train))
    sample_idx = perm[:args.batch_size]
    samples = idx_train[sample_idx]
    idx_train_lst.append(samples)
I = torch.eye(adj.size(0))

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

coeffs = Variable(torch.FloatTensor([1., 3.0]).cuda() if args.cuda else torch.FloatTensor([1., 3.0]) , requires_grad=True)
coeffs_optimizer = optim.SGD([coeffs], lr=1e-1, momentum=0.0)

if args.cuda:
    model.cuda()
    features = features.cuda()
    ind_features = ind_features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    ind_labels = ind_labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    I = I.cuda()
print("\nstart training!\n\n")

def setdiff(n, idx):
    idx = idx.cpu().detach().numpy()
    cp_idx = np.setdiff1d(np.arange(n), idx)
    return cp_idx

def loss(output, labels, idx, S, coeffs, add_logdet):
    output = output.view(-1)
    rL = labels[idx] - output[idx]
    S = S.to_dense()
    
    Gamma = (I - torch.tanh(coeffs[0])*S)*torch.exp(coeffs[1])
    cp_idx = setdiff(len(S), idx)
    loss1 = rL.dot(matmul(Gamma[idx, :][:, idx], rL) - matmul(Gamma[idx, :][:, cp_idx], inv_matmul(Gamma[cp_idx, :][:, cp_idx], matmul(Gamma[cp_idx, :][:, idx], rL))))
    loss2 = torch.Tensor([0.]).cuda()
    if add_logdet: loss2 = logdet(Gamma) - logdet(Gamma[cp_idx, :][:, cp_idx])
    l = loss1 - loss2

    return l/len(idx)

def train(epoch):
    t = time.time()

    # without logdet
    model.train()
    optimizer.zero_grad()
    idx = idx_train_lst[epoch]
    output = model(features, adj).view(-1)
    r2_train = R2(output[idx], labels[idx])
    loss_train = loss(output, labels, idx, adj, coeffs, False)
    loss_train.backward()
    optimizer.step()

    # with logdet
    if epoch % 10 == 0:
        model.train()
        coeffs_optimizer.zero_grad()
        output = model(features, adj).view(-1)
        loss_train = loss(output, labels, idx, adj, coeffs, True)
        loss_train.backward()
        coeffs_optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = loss(output, labels, idx_val, adj, coeffs, True)
    r2_val = R2(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'r2_train: {:.4f}'.format(r2_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'r2_val: {:.4f}'.format(r2_val.item()),
          'alpha: {}'.format(torch.tanh(coeffs[0])),
          'beta: {}'.format(torch.exp(coeffs[1])),
          'time: {:.4f}s'.format(time.time() - t))


def test(adj, features, labels, test_meta):
    model.eval()
    output = model(features, adj).view(-1)

    loss_test = loss(output, labels, idx_test, adj, coeffs, True)
    r2_test = R2(output[idx_test], labels[idx_test])
    
    labels, output, adj = labels.cpu(), output.cpu(), adj.cpu()
    lp_output = lp_refine(idx_test, idx_train, labels, output, adj, torch.tanh(coeffs[0]).item(), torch.exp(coeffs[1]).item())
    lp_r2_test = R2(lp_output, labels[idx_test])
    print("Test set ({}) results:".format(test_meta),
          "loss= {:.4f}".format(loss_test.item()),
          "R2= {:.4f}".format(r2_test.item()),
          "LP_R2= {:.4f}\n".format(lp_r2_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test(adj, features, labels, "2012")
test(adj, ind_features, ind_labels, "2016")

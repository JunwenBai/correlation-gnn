import numpy as np
import torch
import scipy.sparse as sp
from gpytorch import inv_matmul, matmul
from gpytorch.utils import linear_cg

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')

def get_Gamma(alpha, beta, S):
    return beta * torch.eye(S.size(0)) - beta * alpha * S

def interpolate(idx_train, idx_test, res_pred_train, Gamma):
    idx_train = idx_train.cpu().detach().numpy()
    idx_test = idx_test.cpu().detach().numpy()
    idx = np.arange(Gamma.shape[0])
    idx_val = np.setdiff1d(idx, np.concatenate((idx_train, idx_test)))
    idx_test_val = np.concatenate((idx_test, idx_val))
    test_val_Gamma = Gamma[idx_test_val, :][:, idx_test_val]

    res_pred_test = inv_matmul(test_val_Gamma, -matmul(Gamma[idx_test_val, :][:, idx_train], res_pred_train))
    return res_pred_test[:len(idx_test)]

def lp_refine(idx_test, idx_train, labels, output, S, alpha=1., beta=1.):
    Gamma = get_Gamma(alpha, beta, S)

    pred_train = output[idx_train]
    pred_test = output[idx_test]
    res_pred_train = labels[idx_train] - output[idx_train]
    refined_test = pred_test + interpolate(idx_train, idx_test, res_pred_train, Gamma)

    return refined_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import r2_score
from gpytorch import inv_matmul, matmul
from gpytorch.utils import linear_cg

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_jj_data(path, load_partial=False):
    features = np.load(path+"/feats.npy")
    labels = np.load(path+"/labels.npy")
    if load_partial:
        return None, torch.FloatTensor(features), torch.FloatTensor(labels), None, None, None
    adj = np.load(path+"/A.npy").astype(float)
    sp_adj = sp.coo_matrix(adj)
    sp_adj = normalize(sp_adj)
    idx_train = np.load(path+"/train_idx.npy")-1
    idx_val = np.load(path+"/val_idx.npy")-1
    idx_test = np.load(path+"/test_idx.npy")-1
    return sparse_mx_to_torch_sparse_tensor(sp_adj), torch.FloatTensor(features), torch.FloatTensor(labels), torch.LongTensor(idx_train), torch.LongTensor(idx_val), torch.LongTensor(idx_test)

def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) ###

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def R2(outputs, labels):
    outputs = outputs.cpu().detach().numpy().reshape(-1)
    labels = labels.cpu().detach().numpy().reshape(-1)
    return r2_score(labels, outputs)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

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

import numpy as np
import torch
from torch.nn import functional


def dist_torch(X):
    v = torch.sum(X * X, 1, keepdim=True)
    Y = v + torch.t(v) - 2 * torch.matmul(X, X.t())
    # Y[Y < 0] = 0
    return Y


def dist_sparse(X):
    v = X.multiply(X)
    v = v.sum(1)
    Y = v + v.transpose() - 2 * X * X.transpose()
    Y[Y < 0] = 0
    return Y


def trace(A=None, B=None):
    if A is None:
        print('please input pytorch tensor')
        val = None
    elif B is None:
        val = torch.sum(A * A)
    else:
        val = torch.sum(A * B)
    return val


def mmread(R, type='float32'):
    row = R.row.astype(int)
    col = R.col.astype(int)
    val = torch.from_numpy(R.data.astype(type))
    index = torch.from_numpy(np.row_stack((row, col)))
    m, n = R.shape
    return torch.sparse.FloatTensor(index, val, torch.Size([m, n]))


def select2binary(s, n):
    A = torch.eye(n)
    return A[:, s]


def dense2sparse(dense):
    d = dense.cpu()
    index = torch.nonzero(d).t()
    val = d[index[0], index[1]]  # modify this based on dimensionality
    if dense.is_cuda:
        return torch.sparse.FloatTensor(index, val, dense.size()).cuda()
    else:
        return torch.sparse.FloatTensor(index, val, dense.size())


def coo2query(test):
    return {str(r): {str(j): float(v) for i, j, v in zip(test.row, test.col, test.data) if i == r} for r in test.row}


def csr2query(run):
    return {str(r): {str(run.indices[ind]): run.data[ind]
                     for ind in range(run.indptr[r], run.indptr[r + 1])} for r in range(run.shape[0])}


def csr2test(test):
    return {str(r): {str(test.indices[ind]): int(1)
                     for ind in range(test.indptr[r], test.indptr[r + 1])}
            for r in range(test.shape[0]) if test.indptr[r] != test.indptr[r + 1]}


def coo2test(test):
    return {str(r): {str(j): int(1) for i, j in zip(test.row, test.col) if i == r} for r in test.row}


def sort2query(run):
    m, n = run.shape
    return {str(i): {str(int(run[i, j])): float(1.0 / (j + 1)) for j in range(n)} for i in range(m)}


def kNN(X, k, binary=False):
    X = functional.normalize(X, dim=1)
    S = torch.matmul(X, X.t())
    _, idx = torch.sort(S, dim=1, descending=True)
    n = S.shape[0]
    col = idx[:, 1:(k + 1)].cpu().numpy().flatten()
    row = np.arange(n)
    row = row.repeat(k)
    H = torch.zeros(n, n)
    H[row, col] = 1
    H = torch.max(H, H.t())
    if binary:
        return H
    else:
        return torch.exp(-S) * H


def ind2hot(X, N):
    m, n = X.shape
    col = X[:, 0:N].flatten()
    row = np.arange(m)
    row = row.repeat(N)
    # ind = np.stack([row, col])
    # val = np.arange(1, N + 1, 1)
    # val = np.tile(val, m)
    H = np.zeros([m, n])
    H[row, col] = 1
    return H


def mask_one(r, i):
    n = len(i)
    R = r.repeat(n, 1)
    R[range(n), i] = 0
    return R


def PN_sample(r, n):
    l = len(r)
    pos = np.arange(l)[r > 0]
    neg = np.arange(l)[r == 0]
    i = pos[np.random.choice(pos.size, 1)]
    j = neg[np.random.choice(neg.size, n, False)]
    return i, j


def point_loss(ri, rj):
    pos = ri - 1
    loss = pos * pos + rj * rj
    return loss


def pair_loss(ri, rj, eps=1e-8):
    r = ri - rj
    loss = -torch.log(torch.sigmoid(r) + eps)
    return loss


def soft_theshold(w, tau):
    sign = w.clone().detach()
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    return sign * torch.clamp(torch.abs(w) - tau, 0)

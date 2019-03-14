import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from torch import nn
from torch.nn.parameter import Parameter

from lib.utils import mmread, trace, dist_torch


class Similarity(nn.Module):
    def __init__(self, R, args):
        super(Similarity, self).__init__()
        self.R = mmread(R).to(args.device)
        self.m, self.n = R.shape
        self.S = Parameter(torch.rand(self.n, self.n))
        self.S.data = self.S.data - torch.diag(torch.diag(self.S.data))

    def recommend(self):
        return torch.matmul(self.R, self.S)

    def save(self, path):
        self.cpu()
        torch.save(self.state_dict(), path)


class Prism(Similarity):
    def __init__(self, R, F, args):
        super(Prism, self).__init__(R, args)
        self.k = args.factor
        self.alpha = args.alpha
        self.lamb = args.lamb
        self.gamma = args.gamma
        self.initer = args.initer
        self.maxiter = args.maxiter
        self.W = Parameter(torch.rand(args.d, self.k))
        RR = R.transpose() * R
        RR = RR.todense() + self.gamma
        self.RR = torch.from_numpy(RR.astype('float32')).to(args.device)
        self.F = torch.from_numpy(F.astype('float32')).to(args.device)
        self.Q = self.alpha / 2 * dist_torch(self.F)

    def train(self):
        print('update S')
        self.update_S()
        for iter in range(self.maxiter):
            print('update W')
            self.update_W()
            print('update S')
            self.update_S()
            print('iteration {}, obj={:.6f}'.format(iter, self.object()))

    def update_W(self):
        S = self.S.data
        S = (S + torch.t(S)) / 2
        L = torch.diag(torch.sum(S, 0)) - S
        FLF = torch.matmul(self.F.t(), L)
        FLF = torch.matmul(FLF, self.F)
        FLF = (FLF + FLF.t()) / 2

        _, v = torch.symeig(FLF, True)
        W = v[:, 0:self.k]
        FW = torch.matmul(self.F, W)
        self.Q = self.alpha / 2 * dist_torch(FW)
        self.W.data = W

    def update_S(self):
        S = self.S.data
        for i in range(self.initer):
            denominator = torch.matmul(self.RR, S) + self.Q + self.lamb * S + 1e-5
            S *= self.RR
            S /= denominator

    def object(self):
        S = self.S.data
        obj = torch.trace(self.RR) / 2
        obj -= trace(self.RR, S)
        obj += trace(torch.matmul(S.t(), self.RR), S) / 2
        obj += trace(self.Q, S)
        obj += trace(S) * self.lamb / 2
        obj += trace(torch.sum(S, 0) - 1) * self.gamma / 2
        return obj

import argparse

import torch
from scipy import io
from sklearn.preprocessing import normalize

from lib.evaluation import Evaluator
from lib.utils import sort2query, csr2test
from recsys.similarity import Prism

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sparse linear method')
    parser.add_argument('-m', '--maxiter', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--fold', help='specify the fold', type=int, default=1)
    parser.add_argument('--dir', help='dataset directory',
                        default='.')
    parser.add_argument('--save', help='save model', action='store_true')
    parser.add_argument('--initer', type=int, default=10)
    parser.add_argument('--data', help='specify dataset', default='music')
    parser.add_argument('-N', help='number of recommended items', type=int, default=20)
    parser.add_argument('-a', '--alpha', help='parameter alpha', type=float, default=0.1)
    parser.add_argument('-l', '--lamb', help='parameter lambda', type=float, default=0.1)
    parser.add_argument('-g', '--gamma', help='parameter gamma', type=float, default=0.1)
    parser.add_argument('-k', '--factor', help='number of factors', type=int, default=1)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    print('dataset directory: ' + args.dir)
    directory = args.dir + '/' + args.data
    path = '{}/loo/train{}.txt'.format(directory, args.fold)
    print('train data path: ' + path)
    R = io.mmread(path)
    # R = normalize(R, axis=0).tocoo()
    args.m, args.n = R.shape
    path = directory + '/feature.txt'
    print('feature data path: ' + path)
    X = io.mmread(path).A
    X = normalize(X, norm='l2', axis=0)
    # X.data[:] = 1
    # X = normalize(X, axis=1).tocoo()
    args.d = X.shape[1]
    model = Prism(R, X, args).to(args.device)

    # Xu = torch.matmul(R, rX)
    model.train()
    score = model.recommend()
    score[R.row, R.col] = 0
    _, idx = torch.sort(score, 1, True)
    path = '{}/loo/test{}.txt'.format(directory, args.fold)
    print('test file path: {}'.format(path))
    T = io.mmread(path)
    run = sort2query(idx[:, 0:args.N])
    test = csr2test(T.tocsr())
    evaluator = Evaluator({'recall', 'recip_rank_cut'})
    evaluator.evaluate(run, test)
    result = evaluator.show(
        ['recall_10', 'recip_rank_cut_10'])
    print(result)

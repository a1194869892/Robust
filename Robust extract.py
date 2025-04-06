import scipy.io as sio
import numpy as np
import cvxpy as cp
import torch
from cvxpy.atoms.elementwise.power import power
from load_datamat import *
from load_datamat import get_datamat
def find_sparse_sol(Y, i, N, D, weight):
    if i == 0:
        Ybari = Y[:, 1:N]
    if i == N - 1:
        Ybari = Y[:, 0:N - 1]
    if i != 0 and i != N - 1:
        Ybari = np.concatenate((Y[:, 0:i], Y[:, i + 1:N]), axis=1)
    yi = Y[:, i].reshape(D, 1)

    ci = cp.Variable(shape=(N - 1, 1))
    constraint = [cp.sum(ci) == 1]
    obj = cp.Minimize(power(cp.norm(yi - Ybari @ ci, 2), 2) + weight * cp.norm(ci, 1))  # lambda = 199101
    prob = cp.Problem(obj, constraint)
    prob.solve(solver='MOSEK')  # Need to activate it yourself
    return ci.value

def extract_coefficient(X, weight):
    # X: (dim, n_sample)
    dim_sample = X.shape[0]
    num_sample = X.shape[1]
    C = np.concatenate((np.zeros((1, 1)), find_sparse_sol(X, 0, num_sample, dim_sample, weight)), axis=0)
    for i in range(1, num_sample):
        ci = find_sparse_sol(X, i, num_sample, dim_sample, weight)
        zero_element = np.zeros((1, 1))
        cif = np.concatenate((ci[0:i, :], zero_element, ci[i:num_sample, :]), axis=0)
        C = np.concatenate((C, cif), axis=1)
        print("iterator %d/%d" % (i, num_sample))
    print("Affinity matrix calculation completed!")
    return C


# load data x: n*d
shape = 32
db = 'human_ESC'
device = torch.device('cuda')
x, y, dropout_matrix = get_datamat(db, shape, device)
x = x.reshape(x.shape[0], -1)
x = x.cpu().numpy()
x = x.T
weights = [100]
for weight in weights:
    C = extract_coefficient(x, weight=0.1)
    sio.savemat('coef/%s/%s_%s.mat' % (db, db, weight), {'coef': C.T})

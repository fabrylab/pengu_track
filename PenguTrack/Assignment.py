from scipy.optimize import linear_sum_assignment
import numpy as np
from copy import copy

def cost_from_logprob(logprob):
    cost_matrix = np.copy(logprob)
    if cost_matrix.std() > 0.:
        # optimize range of values after exponential function
        cost_matrix -= cost_matrix.min()
        cost_matrix *= 745. / (cost_matrix.max() - cost_matrix.min())
        # cost_matrix *= 1454
        cost_matrix -= 745
    cost_matrix = -1 * np.exp(cost_matrix)
    return cost_matrix


def greedy_assignment(cost):
    r_out=[]
    c_out=[]
    rows, cols = np.unravel_index(np.argsort(cost,kind="heapsort", axis=None), cost.shape)
    while len(rows)>0:
        r_out.append(rows[0])
        c_out.append(cols[0])
        mask = (rows!=rows[0])&(cols!=cols[0])
        rows = rows[mask]
        cols = cols[mask]
    return r_out, c_out

# def hungarian(cost):
#     n, m = cost.shape
#     row_uncovered = np.ones(n, dtype=bool)
#     col_uncovered = np.ones(n, dtype=bool)
#     marked = np.zeros((n,m), dtype=int)
#     C = copy(cost)
#     C -= C.min(axis=1)[:, None]

if __name__ == "__main__":
    from PenguTrack.DataFileExtended import DataFileExtended
    from time import time
    db = DataFileExtended("../synth_data.cdb")
    costs = [p.probability_gain for p in db.table_probability_gain.select()]

    def pad(mat):
        n = max(mat.shape)
        out = np.ones((n,n))*np.amin(mat)
        out[:mat.shape[0], mat.shape[1]]=mat
        return mat

    def dominance(mat):
        return (mat/np.sum(mat, axis=0)[None,:])*(mat/np.sum(mat, axis=1)[:,None])

    def test(mat):
        a = time()
        r,c = linear_sum_assignment(mat)
        a=time()-a
        out = [a]
        mat2 = mat[np.argsort(dominance(mat).max(axis=1))]
        b=time()
        r, c = linear_sum_assignment(mat2)
        out.append(time()-b)
        mat3 = mat[np.argsort(dominance(mat).max(axis=1))[::-1]]
        b=time()
        r, c = linear_sum_assignment(mat3)
        out.append(time()-b)
        mat3 = mat[:, np.argsort(dominance(mat).max(axis=0))]
        b=time()
        r, c = linear_sum_assignment(mat3)
        out.append(time()-b)
        mat3 = mat[:, np.argsort(dominance(mat).max(axis=0))[::-1]]
        b=time()
        r, c = linear_sum_assignment(mat3)
        out.append(time()-b)
        print(out)
        return out

    def test2(mat):
        a = time()
        r,c = linear_sum_assignment(mat)
        a=time()-a
        out = [a]
        dom = dominance(mat)
        sort = np.argsort(dom.max(axis=1))[::-1]
        mat2 = mat[sort]
        mat2[~mask(dom>0.9)]=np.amax(mat2)
        b=time()
        r, c = greedy_assignment(mat2)
        r, c = linear_sum_assignment(mat2)
        out.append(time()-b)

    def mask(mat):
        return np.sum(mat, axis=0, dtype=bool)[None,:]|np.sum(mat,axis=1, dtype=bool)[:,None]


    OUT = []
    for i in range(1):
        for cost in costs[1:]:
            OUT.append(test(np.exp(cost)))


    # import matplotlib.pyplot as plt
    #
    # plt.errorbar(range(5), np.mean(OUT, axis=0), yerr=np.std(OUT, axis=0) / len(OUT) ** 0.5)
    # plt.show()
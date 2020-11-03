from scipy.optimize import linear_sum_assignment
import numpy as np
from copy import copy

def reverse_dict(D):
    """
    Auxiliar function to switch dict entries with keys.
    :param D: dictionary
    :return: dictionary
    """
    return dict([[D[d],d] for d in D])

def hungarian_assignment(*args, **kwargs):
    return linear_sum_assignment(*args, **kwargs)

def greedy_assignment(cost):
    r_out=[]
    c_out=[]
    rows, cols = np.unravel_index(np.argsort(cost, kind="quicksort", axis=None), cost.shape)
    while len(rows)>0:
        r_out.append(rows[0])
        c_out.append(cols[0])
        mask = (rows != rows[0]) & (cols != cols[0])
        rows = rows[mask]
        cols = cols[mask]
    return r_out, c_out


def greedy_assignment2(cost):
    r_out = np.zeros(min(cost.shape), dtype=int)
    c_out = np.zeros(min(cost.shape), dtype=int)
    rows, cols = np.unravel_index(np.argsort(cost, kind="quicksort", axis=None), cost.shape)
    for i in range(len(r_out)):
        r_out[i] = rows[0]
        c_out[i] = cols[0]
        mask = (rows != r_out[i]) & (cols != c_out[i])

        rows = rows[mask]
        cols = cols[mask]
    return r_out, c_out

def network_assignment(cost, order=2, threshold=None, method="linear"):
    cost = np.asarray(cost)
    if threshold is None:
        threshold = np.nanmax(cost)

    if cost.shape[1]<cost.shape[0]:
        transposed = True
        cost = cost.T
    else:
        transposed = False

    # dict to store results
    row_col = {}
    # Do for each track
    for i in np.argsort(cost.min(axis=1)):#range(cost.shape[0]):
        # dict to store ids of matching candidates
        all_rows = set([i])
        all_cols = set()
        for n in range(order):
            if n%2:
                id_list = list(all_cols)
                # ids of horizontal candidates
                ids = np.where(cost[:, id_list] < threshold)[0]
                all_rows.update(ids)
            else:
                id_list = list(all_rows)
                # ids of horizontal candidates
                ids = np.where(cost[id_list] < threshold)[1]
                all_cols.update(ids)

        all_cols = all_cols.difference(row_col.values())
        all_rows = all_rows.difference(row_col.keys())

        # dictionary to translate between sliced array and main arrays
        row_dict = dict(enumerate(sorted(all_rows)))
        col_dict = dict(enumerate(sorted(all_cols)))

        # slice array
        inner_array = cost[sorted(all_rows)][:, sorted(all_cols)]
        # classical assignment in local array
        if method == "greedy":
            r, c = greedy_assignment(inner_array)
        else:
            r, c = linear_sum_assignment(inner_array)

        if len(c) > 0 and len(r) > 0:
            col = c[r == reverse_dict(row_dict)[i]]
            if len(col) > 0:
                col = col_dict[int(col[0])]
                # if col not in row_col.values():
                row_col.update({i: col})
    try:
        rows, cols = np.asarray(list(row_col.items())).T
    except ValueError:
        return [],[]
    assert len(set(rows)) == len(rows) & len(set(cols))==len(cols)
    # assert len(rows) == len(cost)
    if transposed:
        return cols, rows
    else:
        return rows, cols



if __name__ == "__main__":
    size = (200, 100)
    cost = -np.random.rand(*size)
    cost[:size[1],:size[0]][np.diag(np.ones(min(size))).astype(bool)] = -1.
    permut = np.arange(size[0])[:,None] == np.random.choice(np.arange(size[0]), size[0], replace=False)[None,:]
    costP = np.dot(permut, cost)
    GTall = np.where(permut)
    GTmask = GTall[1]<min(size)
    GT = GTall[0][GTmask],GTall[1][GTmask]
    g = set(zip(*GT))
    for solver in [hungarian_assignment,
                   greedy_assignment,
                   greedy_assignment2,
                   lambda x: network_assignment(x, order=0),
                   lambda x: network_assignment(x, order=1),
                   lambda x: network_assignment(x, order=2),
                   lambda x: network_assignment(x, order=3)]:
        ST = solver(costP)
        print(np.all(ST[0] == GT[0]) and np.all(ST[1] == GT[1]))
        s = set(zip(*ST))
        print(len(s), len(g), len(s.difference(g)), len(g.difference(s)))

    N = 200
    out = []
    outM = []
    for N in [10,20,50,100,200,500,1000,2000,5000,10000]:
        outS = []
        for i in range(10):
            pos = np.random.random((N,2))
            dist = np.linalg.norm(pos[None,:]-pos[:,None], axis=-1)
            dist[np.diag(np.ones(N)).astype(bool)] = np.nan
            out.append([N, np.nanmin(dist), np.nanmean(dist)])
            outS.append([N, np.nanmin(dist), np.nanmean(dist)])
        outS = np.array(outS)
        outM.append(np.mean(outS, axis=0))
    out = np.array(out)
    outM = np.array(outM)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(out[:,0], out[:,1], color="b")
    # plt.scatter(out[:,0], out[:,2])
    plt.scatter(outM[:,0], outM[:,1], color="r")
    plt.plot(out[:,0], 1./out[:,0])
    plt.plot(out[:,0], 1./out[:,0]**2)
    plt.plot(out[:,0], 1./out[:,0]**1.5)
    plt.show()
    #     print(ST[1][ST[0])
    #
    # ST = linear_sum_assignment(costP)
    # assert np.all(ST[0] == GT[0]) and np.all(ST[1] == GT[1])
    #
    #

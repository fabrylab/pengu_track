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

def cost_from_logprob(logprob, value=None):
    cost_matrix = np.copy(logprob)
    if value is None:
        value = cost_matrix
    # if cost_matrix.std() > 0.:
        # optimize range of values after exponential function
        # value -= cost_matrix.min()
        # value *= 745. / (cost_matrix.max() - cost_matrix.min())
        # value -= 745
    value = -1. * np.exp(value)
    return value


def greedy_assignment(cost):
    r_out=[]
    c_out=[]
    rows, cols = np.unravel_index(np.argsort(cost, kind="quicksort", axis=None), cost.shape)
    # rows, cols = np.unravel_index(np.argsort(cost, kind="heapsort", axis=None), cost.shape)
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
    # rows, cols = np.unravel_index(np.argsort(cost, kind="heapsort", axis=None), cost.shape)
    # while len(rows)>0:
    # mask = np.ones_like(rows, dtype=bool)
    for i in range(len(r_out)):
        r_out[i] = rows[0]
        c_out[i] = cols[0]
        mask = (rows != r_out[i]) & (cols != c_out[i])

        # r_out.append(rows[0])
        # c_out.append(cols[0])
        # mask &= (rows != rows[0]) & (cols != cols[0])
        # mask &= (rows != r_out[i]) & (cols != c_out[i])
        rows = rows[mask]
        cols = cols[mask]
    # assert len(set(c_out)) == len(c_out)
    return r_out, c_out
    # # rows, cols = np.unravel_index(np.argsort(cost, kind="quicksort", axis=None), cost.shape)
    # # rows, cols = np.unravel_index(np.argsort(cost, kind="heapsort", axis=None), cost.shape)
    # # idx = np.zeros_like(cost, dtype=int)
    # # idx[rows, cols] = range(len(rows))
    # idx = np.argsort(cost, kind="heapsort", axis=None).reshape(cost.shape)
    # r,c = np.where(idx == np.amin(idx, axis=1)[:, None])
    # assert len(set(c)) == len(c)
    # return r,c
    # # assert len(set(r))==len(r)
    # # mat = np.array(idx)
    # # mat2 = np.array(idx)
    # # mat[:] = idx.min(axis=1)[:, None]
    # # mat2[:] = idx.min(axis=0)[None, :]
    # # mat3 = np.amin([mat, mat2], axis=0)
    # # id = np.array(idx, dtype=float)
    # # matb = np.array(id)
    # # rows2 =[]
    # # cols2 =[]
    # # # while not np.all(np.isnan(id)):
    # # for i in range(min(id.shape)):
    # #     r, c = np.unravel_index(np.nanargmin(id), id.shape)
    # #     matb[r] = (matb[r] > id[r, c])*id[r, c] + (matb[r] <= id[r, c])*matb[r]
    # #     matb[:, c] = (matb[:, c] > id[r, c])*id[r, c] + (matb[:, c] <= id[r, c])*matb[r]
    # #     id[r] = np.nan
    # #     id[:, c] = np.nan
    # #     rows2.append(r)
    # #     cols2.append(c)
    # # return np.array([np.where(mat3 == i) for i in range(min(idx.shape))]).T[0]


def dummy_assignment(cost):
    x = np.arange(min(cost.shape))
    return x, x


def mask(mat):
    return np.sum(mat, axis=0, dtype=bool)[None,:]|np.sum(mat,axis=1, dtype=bool)[:,None]


def dominance(mat):
    mat = copy(mat)
    mat *=-1.
    # minimum = np.amin(mat)
    # maximum = np.amax(mat)
    # mat = (mat-minimum)/(maximum-minimum)
    # return (mat/np.sum(mat, axis=0)[None,:])*(mat/np.sum(mat, axis=1)[:,None])
    return mat/np.amax(np.meshgrid(np.amax(mat, axis=0), np.amax(mat, axis=1)), axis=0)
    # maximum = np.amax(np.meshgrid(np.amax(mat, axis=0), np.amax(mat, axis=1)), axis=0)
    # second_max = copy(mat)
    # second_max[second_max==maximum]=np.amin(second_max)
    # second_max = np.amax(np.meshgrid(np.amax(second_max, axis=0), np.amax(second_max, axis=1)), axis=0)
    # return 1.-second_max/maximum


def compact_critical(mat, threshold=0.5):
    masked = ~mask(dominance(mat)>threshold)
    new_shape = (masked.sum(axis=0).max(), masked.sum(axis=1).max())
    row_dict = dict(enumerate(np.arange(masked.shape[0])[masked.max(axis=1)>0]))
    col_dict = dict(enumerate(np.arange(masked.shape[1])[masked.max(axis=0)>0]))
    return row_dict, col_dict, mat[masked].reshape(new_shape)


def compact_uncritical(mat, threshold=0.5):
    masked = ~mask(~mask(dominance(mat)>=threshold))
    new_shape = (masked.sum(axis=0).max(), masked.sum(axis=1).max())
    row_dict = dict(enumerate(np.arange(masked.shape[0])[masked.max(axis=1)>0]))
    col_dict = dict(enumerate(np.arange(masked.shape[1])[masked.max(axis=0)>0]))
    return row_dict, col_dict, mat[masked].reshape(new_shape)


def splitted_solver(mat, dom_threshold=0.5):
    c_row_dict, c_col_dict, critical_mat = compact_critical(mat, threshold=dom_threshold)
    uc_row_dict, uc_col_dict, uncritical_mat = compact_uncritical(mat, threshold=dom_threshold)
    r0, c0 = linear_sum_assignment(critical_mat)
    r0 = [c_row_dict[r] for r in r0]
    c0 = [c_col_dict[c] for c in c0]
    r1, c1 = greedy_assignment(uncritical_mat)
    r1 = [uc_row_dict[r] for r in r1]
    c1 = [uc_col_dict[c] for c in c1]
    r0.extend(r1)
    c0.extend(c1)
    r0=np.array(r0)
    c0=np.array(c0)
    args = np.argsort(r0)
    return r0[args], c0[args]

def network_assignment(cost, order=2, threshold=None, method="linear"):
    cost = np.array(cost)
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

        # # append results to match list
        # for R, C in zip(r, c):
        #     # translate to entries in main arrays
        #     R = row_dict[R]
        #     C = col_dict[C]
        #     # if cost[R, C] < threshold:
        #     if R not in row_col:
        #         row_col.update({R: C})
        #         print("Match!")
        #     elif cost[R, row_col[R]] > cost[R, C]:
        #         row_col.update({R: C})
        #         print("Overwriting Match!")
    try:
        rows, cols = np.array(list(row_col.items())).T
    except ValueError:
        return [],[]
    assert len(set(rows)) == len(rows) & len(set(cols))==len(cols)
    # assert len(rows) == len(cost)
    if transposed:
        return cols, rows
    else:
        return rows, cols
# def hungarian(cost):
#     n, m = cost.shape
#     row_uncovered = np.ones(n, dtype=bool)
#     col_uncovered = np.ones(n, dtype=bool)
#     marked = np.zeros((n,m), dtype=int)
#     C = copy(cost)
#     C -= C.min(axis=1)[:, None]


if __name__ == "__main__":
    from PenguTrack.DataFileExtended import DataFileExtended
    import matplotlib.pyplot as plt
    plt.ion()
    from time import time
    import pandas
    db = DataFileExtended("../synth_data_big.cdb")
    costs = [p.probability_gain for p in db.table_probability_gain.select()]

    def get(mat, set):
        return np.sum([mat[k] for k in set])

    def test(mat, data_set_name="default", hungarian_maxsize=250):
        out = []
        # Hungarian algorithm only for small matrices
        if max(mat.shape)<hungarian_maxsize:
            a = time()
            r0, c0 = linear_sum_assignment(mat)
            a = time()-a
            out.append(["hungarian", max(mat.shape), a, data_set_name, 1., get(mat, set(zip(r0,c0)))])
            print(["hungarian", max(mat.shape), a, data_set_name, 1., get(mat, set(zip(r0,c0)))])
        else:
            r0=c0=None

        for t in np.arange(0.1,1.,0.1):
            a = time()
            r,c = splitted_solver(mat, dom_threshold=t)
            a = time()-a
            if r0 is not None:
                p = float(len(set(zip(r0,c0)).intersection(zip(r,c))))/len(set(zip(r0,c0)))
            else:
                p = np.nan
            out.append(["splitted_%s"%t, max(mat.shape), a, data_set_name, p, get(mat, set(zip(r,c)))])
            print(["splitted_%s"%t, max(mat.shape), a, data_set_name, p, get(mat, set(zip(r,c)))])

        a = time()
        r,c = greedy_assignment(mat)
        a = time()-a
        if r0 is not None:
            p = float(len(set(zip(r0,c0)).intersection(zip(r,c))))/len(set(zip(r0,c0)))
        else:
            p = np.nan
        out.append(["greedy", max(mat.shape), a, data_set_name, p, get(mat, set(zip(r,c)))])
        print(["greedy", max(mat.shape), a, data_set_name, p, get(mat, set(zip(r,c)))])
        return out

    if False:
        ALL = []
        for s in 2**np.arange(1, 8):
            s_ = int(s*(1+0.1*np.random.rand()))
            for cost in costs[1:]:
                ALL.extend(test(cost[:s,:s_], data_set_name="Synthetic Tracks", hungarian_maxsize=250))


        data = pandas.DataFrame(ALL, columns=["algorithm","size", "time", "data_set", "performance", "costs"])
        data.to_csv("/home/alex/Promotion/AssignmentPerformance2.csv")



    data = pandas.read_csv("/home/alex/Promotion/AssignmentPerformance2.csv")
    data_m = data.groupby(["algorithm", "data_set", "size"]).mean()
    data_s = data.groupby(["algorithm", "data_set", "size"]).std()/data.groupby(["algorithm", "data_set", "size"]).count()**0.5

    x = data_m.time["greedy", "Synthetic Tracks"].index

    from scipy.optimize import curve_fit
    import seaborn

    cpal = seaborn.color_palette(n_colors=4)
    # cpal1 = seaborn.dark_palette(cpal[0], n_colors=1)
    # cpal2 = seaborn.dark_palette(cpal[1], n_colors=1)
    cpal1 = [cpal[0]]
    cpal2 = [cpal[1]]
    cpal3 = seaborn.light_palette(cpal[3], n_colors=10)

    def polynom_exp(x, A, t):
        x = np.array(x, dtype=float)
        x = np.exp(x)
        return np.log(A * x ** t)

    def polynom(x, A, t):
        x = np.array(x, dtype=float)
        return A * x ** t

    def log(x, A, t):
        x = np.array(x, dtype=float)
        return A*x*np.log(x)

    def log_exp(x, A, t):
        x = np.array(x, dtype=float)
        x = np.exp(x)
        return np.log(A*x*np.log(x))

    f = dict()
    for l in set(data.algorithm.values):
        f[l] = polynom
    # f["greedy"] = log

    f_exp = dict()
    for l in set(data.algorithm.values):
        f_exp[l] = polynom_exp
    # f_exp["greedy"] = log_exp


    i = j = k = 0
    for alg, dat in np.array(np.meshgrid(*data_m.index.levels[:2])).reshape((2, -1)).T:
        if alg.count("hungarian"):
            c = cpal1[i]
            i += 1
        elif alg.count("greedy"):
            c = cpal2[j]
            j += 1
        elif alg.count("splitted"):
            c = cpal3[k]
            k += 1
        plt.errorbar(data_m.time[alg, dat].index, data_m.time[alg, dat].values,
                     fmt='',
                     yerr=data_s.time[alg, dat],
                     label=alg + ", " + dat,
                     color=c)

    params = {}
    for l in set(data.algorithm.values):
        params[l] = curve_fit(f_exp[l],
                           np.log(data_m.time[l, "Synthetic Tracks"].index),
                           np.log(data_m.time[l, "Synthetic Tracks"].values),
                           p0=[10 ** -3.5, 3.])
        # params[l] = curve_fit(f[l],
        #                    data_m.time[l, "Synthetic Tracks"].index,
        #                    data_m.time[l, "Synthetic Tracks"].values,
        #                    p0=[10 ** -3.5, 3.])

    for l in ["hungarian", "greedy"]:
        popt, pcov = params[l]
        plt.plot(x, np.exp(f_exp[l](np.log(x), *popt)), label="%s Fit %.2E, %.2f" % (l, popt[0], popt[1]), color=cpal[2])
        # plt.plot(x, f[l](x, *popt), label="%s Fit %.2E,  %.2f" % (l, popt[0], popt[1]), color=cpal[2])
    # for l in ["greedy"]:
    #     popt, pcov = params[l]
    #     # plt.plot(x, np.exp(f_exp(np.log(x), *popt)), label="%s Powerlaw Fit %.2E x^{%.2f}" % (l, popt[0], popt[1]), color=cpal[2])
    #     plt.plot(x, f(x, *popt), label="%s Powerlaw Fit %.2E x^{%.2f}" % (l, popt[0], popt[1]), color=cpal[2])

    plt.legend()
    plt.loglog()
    plt.grid()
    # plt.axis("equal")
    # plt.savefig("/home/alex/Promotion/assignment_performance.pdf")

    plt.figure()

    x = data_m.performance["greedy", "Synthetic Tracks"].index
    i = j = k = 0
    for alg, dat in np.array(np.meshgrid(*data_m.index.levels[:2])).reshape((2, -1)).T:
        if alg.count("hungarian"):
            continue
            # c = cpal1[i]
            # i += 1
        elif alg.count("greedy"):
            c = cpal2[j]
            j += 1
        elif alg.count("splitted"):
            c = cpal3[k]
            k += 1
        plt.errorbar(data_m.performance[alg, dat].index, data_m.performance[alg, dat].values,
                     fmt='',
                     yerr=data_s.performance[alg, dat],
                     label=alg + ", " + dat,
                     color=c)

    # x = 2 ** np.arange(1, 10)
    # plt.plot(x[:-2],
    #          data[data["algorithm"] == "hungarian"][data["data_set"] == "Synthetic Tracks"].groupby("size")["performance"].mean(),
    #          label="Synthetic Tracks, hu")
    # plt.plot(x[:-2],
    #          data[data["algorithm"] == "greedy"][data["data_set"] == "Synthetic Tracks"].groupby("size")["performance"].mean(),
    #          label="Synthetic Tracks, gy")
    plt.legend()
    # plt.loglog()
    plt.semilogx()
    plt.grid()
    # plt.axis("equal")

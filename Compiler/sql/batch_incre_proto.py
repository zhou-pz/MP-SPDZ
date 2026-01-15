from Compiler.library import *
from Compiler.sorting import radix_sort
from Compiler.types import *
from Compiler.util import *
from Compiler.sql.utils import *

def set_dummies(T_attr, T_valid, update_rows_attr, update_rows_valid):
    """ Set update_rows to dummies in table T based on a unique attribute."""
    n = T_attr.shape[0]
    k = update_rows_attr.shape[0]

    is_targeted = Array(n, sint)
    e = Matrix(k, n, sint)
    for i in range(k):
        e[i] = (T_attr[:] == update_rows_attr[i]) * update_rows_valid[i] * T_valid[:]
        is_targeted[:] += e[i]
    T_valid[:] = T_valid[:] * (1 - is_targeted[:])    
    return e

# Orderby
def Orderby_batch_delete(D, update_rows, limit, outOpt):
    """ Attributes: 0-key, 1-valid, 2-id, ...-payload """
    D_prime = D.same_shape()
    D_prime.assign(D)
    valid, id = 1, 2
    k = update_rows.shape[1]
    e = set_dummies(D_prime[id], D_prime[valid], update_rows[id], update_rows[valid])
    D_prime_trans = D_prime.transpose()
    radix_sort(1 - D_prime[valid], D_prime_trans, n_bits=1)
    if not outOpt:
        e_prime = Array(k, sint)
        for i in range(k):
            e_prime[i] = e[i][0:limit].sum()
        e_s = Array(k, sint)
        e_s.assign(e_prime)
        radix_sort(1 - e_prime, e_s, n_bits=1)
        
        t_outs_ins = D.transpose().get_part(limit, k).transpose()
        t_outs_ins[valid] = t_outs_ins[valid] * e_s
        update_rows[valid] = update_rows[valid] * e_prime
        return D_prime_trans, t_outs_ins, update_rows
    else:
        return D_prime_trans, None, None

# Grouby
def Groupby_batch(G, update_rows, update_manners, outOpt):
    """ Attributes: 0-key, 1-value, 2-valid """
    key, value, valid = 0, 1, 2
    a = G.shape[0]
    n = G.shape[1]
    k = update_rows.shape[1]
    e = Matrix(k, n, sint)
    e_prime = Matrix(k, n, sint)
    for i in range(k):
        e[i] = (G[key][:] == update_rows[key][i]) * update_rows[valid][i]
        if update_manners[i] == 0:  # insert
            e_prime[i] = e[i] * update_rows[value][i]
        else:  # delete
            e_prime[i] = e[i] * ( - update_rows[value][i])

    if outOpt:
        for i in range(n):
            G[value][i] += e_prime.get_column(i).sum()
        return G, None, None
    else:
        e[:] = 1 - e[:]
        f = Matrix(1, n, sint)
        g = Matrix(1, n, sint)
        G_prime = G.same_shape()
        G_prime.assign(G)
        for i in range(n):
            f[0][i] = e_prime.get_column(i).sum()
            G[value][i] += f[0][i]
            g[0][i] = tree_reduce(operator.mul, e.get_column(i))
        g[0][:] = 1 - g[0][:]

        Combined = G_prime.concat(g)
        Combined = Combined.concat(f)
        
        Combined_trans = Combined.transpose()
        radix_sort(1 - g[0], Combined_trans, n_bits=1)
        Combined = Combined_trans.transpose()

        G_prime = Combined.get_part(0, a)
        G_prime[valid] = G_prime[valid] * Combined[a][:]
        f[0] = Combined[a+1]

        t_outs_ins = G_prime.transpose().get_part(0, k).transpose()
        t_outs_del = t_outs_ins.same_shape()
        t_outs_del.assign(t_outs_ins)

        t_outs_ins[value][:] += f[0][0:k]
        return G, t_outs_ins, t_outs_del
    
# Join
def Join_batch_delete(A, B, J, update_rows, target_table, outOpt):
    """ Attributes of A and B: 0-key, 1-valid, ...-payload_A/B
        Attributes of J: 0-key, 1-valid, ...-payload_A, ...-payload_B """
    key, valid = 0, 1
    k = update_rows.shape[1]
    a_A = A.shape[0]
    a_B = B.shape[0]
    a_J, n_J = J.shape[0], J.shape[1]
    e = set_dummies(J[key], J[valid], update_rows[key], update_rows[valid])
    if target_table == 'A':
        set_dummies(A[key], A[valid], update_rows[key], update_rows[valid])
        if not outOpt:
            t_outs = Matrix(a_J, k, sint)
            t_outs.assign(update_rows)
            # compute payloads from B
            e_sum = Array(k, sint)
            e_prime = MultiArray([a_B - 2, k, n_J], sint)
            e_prime_sum = Matrix(a_B - 2, k, sint)
            for i in range(a_B - 2):
                for j in range(k):
                    e_prime[i][j] = e[j] * J[a_A + i]
                    e_prime_sum[i][j] = e_prime[i][j].get_vector().sum()
                t_outs[a_A + i] = e_prime_sum[i]
            for j in range(k):
                e_sum[j] = e[j].get_vector().sum()
            t_outs[valid] = t_outs[valid] * e_sum
            return A, B, J, t_outs
        else:
            return A, B, J, None
    elif target_table == 'B':
        set_dummies(B[key], B[valid], update_rows[key], update_rows[valid])
        if not outOpt:
            t_outs = Matrix(a_J, k, sint)
            t_outs.assign(update_rows)
            for i in range(a_B - 2):
                t_outs[a_A + i] = update_rows[2 + i]
            # compute payloads from A
            e_sum = Array(k, sint)
            e_prime = MultiArray([a_A - 2, k, n_J], sint)
            e_prime_sum = Matrix(a_A - 2, k, sint)
            for i in range(a_A - 2):
                for j in range(k):
                    e_prime[i][j] = e[j] * J[2 + i]
                    e_prime_sum[i][j] = e_prime[i][j].get_vector().sum()
                t_outs[2 + i] = e_prime_sum[i]
            for j in range(k):
                e_sum[j] = e[j].get_vector().sum()
            t_outs[valid] = t_outs[valid] * e_sum
            return A, B, J, t_outs
        else:
            return A, B, J, None
    else:
        raise ValueError("target_table must be either 'A' or 'B'")

# Seletion
def Selection_batch_insert(S, update_rows):
    """ Attributes: 0-valid, 1-id, ...-payload """
    def predicate(columns):
        # Example predicate: payload_1 > 10
        return columns[:] > sint(10)
    update_rows[0][:] = update_rows[0][:] * predicate(update_rows[2])
    S = S.concat_columns(update_rows)
    return S, update_rows

def Selection_batch_delete(S, update_rows):
    """ Attributes: 0-valid, 1-id, ...-payload """
    set_dummies(S[1], S[0], update_rows[1], update_rows[0])
    return S

# Aggregation
def Aggregation_batch_insert(res, values, func):
    def aggregate(values):
        if func == 'sum':
            return values.get_vector().sum()
        elif func == 'max':
            return max(values)
        elif func == 'min':
            return min(values)
    values[values.shape[0] - 1] = res
    return aggregate(values)

def Aggregation_batch_delete(res, values):
    values.assign(-values)
    values[values.shape[0] - 1] = res
    return values.get_vector().sum()

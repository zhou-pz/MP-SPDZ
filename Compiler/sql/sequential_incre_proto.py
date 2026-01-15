from Compiler.library import *
from Compiler.types import *
from Compiler.util import *
from Compiler.program import Program
shift_length = Program.prog.bit_length - 1

# OrderBy
def Optimized_CDIC(update_tuple, D):
    a = D.shape[0]
    h = D.shape[1]
    D_out = Matrix(a, h + 1, sint)

    # b[i] = D[i, -v||k] > update_tuple[-v||k]  # 用-v是因为k升序v降序
    b = Array(h + 1, sint)
    D_vk = D[1].same_shape()
    D_vk[:] = ((1 - D[1][:]) << shift_length) + D[0][:]
    update_vk = ((1 - update_tuple[1]) << shift_length) + update_tuple[0]
    b[0:h] = D_vk[0:h] > update_vk
    b[h] = sint(1)

    c = b.same_shape()
    c[0] = b[0]
    c[1:h+1] = b[1:h+1] - b[0:h]

    D_prime_col = Array(h + 1, sint)
    D_pprime_col = Array(h + 1, sint)
    @for_range_opt(a)
    def _(j):
        # D_prime[0:h] = D[0:h] * (1 - b[0:h])
        D_prime_col[0:h] = D[j][:] * (1 - b[0:h])
        D_prime_col[h] = sint(0)
        # D_pprime[1:h+1] = D_prime[1:h+1] * (b[1:h+1] - c[1:h+1])
        D_pprime_col[1:h+1] = D[j][:] * (b[1:h+1] - c[1:h+1])
        D_pprime_col[0] = sint(0)
        # D_out[0:h+1] = update_tuple * c[0:h+1] + D_prime[0:h+1] + D_pprime[0:h+1]
        D_out[j][:] = update_tuple[j] * c[:] + D_prime_col[:] + D_pprime_col[:]
    return D_out

def CDIC(update_tuple, D):
    """ The 'i,j' in the annotation refer to the Pointers of 'row, attribute' respectively. """
    a = D.shape[0]
    h = D.shape[1]
    D_prime = Matrix(a, h + 1, sint)
    D_pprime = Matrix(a, h + 1, sint)

    # Mask
    # b[i] = D[i, -v||k] > update_tuple[-v||k]  # 用-v是因为k升序v降序
    b = Array(h, sint)
    D_vk = D[1].same_shape()
    D_vk[:] = ((1 - D[1][:]) << shift_length) + D[0][:]
    update_vk = ((1 - update_tuple[1]) << shift_length) + update_tuple[0]
    b[:] = D_vk[0:h] > update_vk
    # D_prime[i] = D_transpose[i] + b[i] * (update_tuple - D_transpose[i])
    @for_range_opt(a)
    def _(j):
        D_prime[j][0:h] = D[j][:] + b[:] * (update_tuple[j] - D[j][:])
    D_prime.set_column(h, update_tuple.get_vector())

    # Rewrite
    D_pprime.set_column(0, D_prime.get_column(0))
    # b[i] = D_transpose[i-1, v||k] > D_prime[i, v||k]
    D_prime_vk = D_prime[1].same_shape()
    D_prime_vk[:] = ((1 - D_prime[1][:]) << shift_length) + D_prime[0][:]
    b[:] = D_vk[0:h] > D_prime_vk[1:h+1]
    # D_pprime[i] = D_prime[i] + b[i] * (D_transpose[i-1] - D_prime[i]
    @for_range_opt(a)
    def _(j):   
        D_pprime[j][1:h+1] = D_prime[j][1:h+1] + b * (D[j][:] - D_prime[j][1:h+1])
    return D_pprime

def OrderBy_insert(update_tuple, D, limit, outOpt):
    """ Attributes: 0-key, 1-valid, 2-id, ...-payload """
    if outOpt == False:
        t_out1 = update_tuple.same_shape()
        t_out2 = update_tuple.same_shape()

        update_tuple_vk = (update_tuple[1] << shift_length) + update_tuple[0]
        D_watchdog = D.transpose()[limit-1]
        D_watchdog_vk = (D_watchdog[1] << shift_length) + D_watchdog[0]
        flag = update_tuple_vk < D_watchdog_vk
        for i in range(len(t_out1)):
            t_out1[i] = D[i][limit-1] * flag
        for i in range(len(t_out2)):
            t_out2[i] = update_tuple[i] * flag
        return Optimized_CDIC(update_tuple, D), t_out1, t_out2
    else:
        return Optimized_CDIC(update_tuple, D)

def LDPC(D, f):
    h = D.shape[1]
    a = D.shape[0]
    D_prime = Matrix.create_from(D)
    f_prime = f.same_shape()
    D_pprime = D.same_shape()
    sone = sintbit(1)
    # Mask
    f_prime = f.get_vector().prefix_sum()
    D_prime[1][:] = D_prime[1][:] * (sone - f_prime)
    # Push down    
    b = sint.Array(h-1)
    D_prime_vk = D_prime[1].same_shape()
    D_prime_vk[:] = (D_prime[1][:] << shift_length) + D_prime[0][:]
    D_vk = D[1].same_shape()
    D_vk[:] = (D[1][:] << shift_length) + D[0][:]
    b[:] = D_prime_vk[0:h-2] >= D_vk[1:h-1]
    @for_range_opt(a)
    def _(j):
        D_pprime[j][:] = D[j][1:h] + b[:] * (D_prime[j][0:h-1] - D[j][1:h])
        D_pprime[j][h-1] = D_prime[j][h-1]
    D_pprime = D_pprime.transpose()
    return D_pprime

def OrderBy_delete(update_tuple, D, limit, outOpt):
    """ Attributes: 0-key, 1-valid, 2-id, ...-payload """
    h = D.shape[1]
    f = sint.Array(h)
    f[:] = (D[2][:] == update_tuple[2]) * D[1][:] * update_tuple[1]
    if outOpt == False:
        f_sum = Array.create_from(f[0:limit]).get_vector().sum()
        t_out1 = Array.create_from(update_tuple[:] * f_sum)
        t_out2 = Array.create_from(D.transpose()[limit][:] * f_sum)
        return LDPC(D, f), t_out1, t_out2
    else:
        return LDPC(D, f)

# GroupBy
def GroupBy_insert(G, update_tuple, outOpt):
    """ Attributes: 0-key, 1-value, 2-valid """
    n = G.shape[1]
    a = G.shape[0]
    f = Array(n, sint)
    f[:] = (G[0][:] == update_tuple[0]) * G[2][:] * update_tuple[2]

    G_n = Matrix(1, a, sint)
    G_n[0] = update_tuple
    sum_f = f.get_vector().sum()
    G_n[0][2] = G_n[0][2] * (1 - sum_f)

    G[1][:] = G[1][:] + f[:] * update_tuple[1]

    if outOpt == False:
        ri_mul_fi = Array(n, sint)
        ri_mul_fi[:] = G[1][:] * f[:]
        
        t_u_out1 = Array.create_from(update_tuple)
        t_u_out1[1] = ri_mul_fi.get_vector().sum()
        t_u_out1[2] = sum_f

        t_u_out2 = Array.create_from(t_u_out1)
        t_u_out2.assign(t_u_out1)
        t_u_out2[1] = t_u_out2[1] - update_tuple[1]

        t_u_out3 = G_n
        # t_u_out1 - insert, t_u_out2 - delete, t_u_out3 - insert
        return G.concat_columns(G_n.transpose()), t_u_out1, t_u_out2, t_u_out3
    else:
        return G.concat_columns(G_n.transpose())

def GroupBy_delete(G, update_tuple, outOpt):
    """ Attributes: 0-key, 1-value, 2-valid """
    n = G.shape[1]
    f = Array(n, sint)
    f[:] = (G[0][:] == update_tuple[0]) * G[2][:] * update_tuple[2]
    G[1][:] = G[1][:] - f[:] * update_tuple[1]

    if outOpt == False:
        ri_mul_fi = Array(n, sint)
        ri_mul_fi[:] = G[1][:] * f[:]
        
        t_u_out1 = Array.create_from(update_tuple)
        t_u_out1[1] = ri_mul_fi.get_vector().sum()
        t_u_out1[2] = f.get_vector().sum()

        t_u_out2 = Array.create_from(t_u_out1)
        t_u_out2[1] += update_tuple[1]
        return G, t_u_out1, t_u_out2
    else:
        return G

# Join
def Join_insert(update_tuple, A, B, J, target_table):
    """ Attributes of A and B: 0-key, 1-valid, ...-payload_A/B
        Attributes of J: 0-key, 1-valid, ...-payload_A, ...-payload_B """
    a_A = A.shape[0]
    a_B = B.shape[0]
    n_A = A.shape[1]
    n_B = B.shape[1]
    if target_table == 'A':
        update_tuple_mat = Matrix(1, a_A, sint)
        update_tuple_mat[0] = update_tuple
        A = A.concat_columns(update_tuple_mat.transpose())
        f = Array(n_B, sint)
        f[:] = (B[0][:]== update_tuple[0]) * B[1][:]
    else:
        update_tuple_mat = Matrix(1, a_B, sint)
        update_tuple_mat[0] = update_tuple
        B = B.concat_columns(update_tuple_mat.transpose())
        f = Array(n_A, sint)
        f[:] = (A[0][:]== update_tuple[0]) * A[1][:]

    a_J = a_A + a_B - 2
    t_out = Array(a_J, sint)
    t_out[0] = update_tuple[0]
    t_out[1] = update_tuple[1] * f.get_vector().sum()

    if target_table == 'A':
        # compute payloads from A
        @for_range_opt(a_A - 2)
        def _(j):
            t_out[2 + j] = update_tuple[2 + j]
        # compute payloads from B
        pB_mul_f = Matrix(a_B - 2, n_B, sint)
        @for_range_opt(a_B - 2)
        def _(j):
            pB_mul_f[j][:] = B[2 + j][:] * f[:]
            t_out[a_A + j] = pB_mul_f[j].get_vector().sum()
    elif target_table == 'B':
        # compute payloads from B
        @for_range_opt(a_B - 2)
        def _(j):
            t_out[a_A + j] = update_tuple[2 + j]
        # compute payloads from A
        pA_mul_f = Matrix(a_A - 2, n_A, sint)
        @for_range_opt(a_A - 2)
        def _(j):
            pA_mul_f[j][:] = A[2 + j][:] * f[:]
            t_out[2 + j] = pA_mul_f[j].get_vector().sum()

    t_out_mat = Matrix(1, a_J, sint)
    t_out_mat[0] = t_out
    return A, B, J.concat_columns(t_out_mat.transpose()), t_out

def Join_delete(update_tuple, A, B, J, target_table, outOpt):
    """ Attributes of A and B: 0-key, 1-valid, ...-payload_A/B
        Attributes of J: 0-key, 1-valid, ...-payload_A, ...-payload_B """
    a_A = A.shape[0]
    a_B = B.shape[0]
    a_J, n_J = J.shape[0], J.shape[1]
    t_out = Array(a_J, sint)
    J_f = Array(n_J, sint)
    if target_table == 'A':
        A[1][:] = (1 - ((A[0][:] == update_tuple[0]) * update_tuple[1])) * A[1][:]
        J_f[:] = (J[0][:] == update_tuple[0]) * J[1][:] * update_tuple[1]
        if outOpt == False:
            t_out[0] = update_tuple[0]  # key
            t_out[1] = update_tuple[1] * J_f.get_vector().sum() # valid
            # compute payloads from A
            @for_range_opt(a_A - 2)
            def _(j):
                t_out[2 + j] = update_tuple[2 + j]
            # compute payloads from B in J
            @for_range_opt(a_B - 2)
            def _(j):
                t_out[a_A + j] = (J[a_A + j][:] * J_f[:]).sum()
    elif target_table == 'B':
        B[1][:] = (1 - ((B[0][:] == update_tuple[0]) * update_tuple[1])) * B[1][:]
        J_f[:] = (J[0][:] == update_tuple[0]) * J[1][:] * update_tuple[1]
        if outOpt == False:
            t_out[0] = update_tuple[0]  # key
            t_out[1] = update_tuple[1] * J_f.get_vector().sum() # valid
            # compute payloads from B
            @for_range_opt(a_B - 2)
            def _(j):
                t_out[a_A + j] = update_tuple[2 + j]
            # compute payloads from A in J
            @for_range_opt(a_A - 2)
            def _(j):
                t_out[2 + j] = (J[2 + j][:] * J_f[:]).sum()
    else:
        raise ValueError("target_table must be either 'A' or 'B'")
    J[1][:] = (1 - J_f[:]) * J[1][:]
    return A, B, J, t_out

# Seletion
def Selection_insert(S, update_rows):
    """ Attributes: 0-valid, 1-id, ...-payload """
    def predicate(columns):
        # Example predicate: payload_1 > 10
        return columns[:] > sint(10)
    update_rows[0][:] = update_rows[0][:] * predicate(update_rows[2])
    S = S.concat_columns(update_rows)
    return S, update_rows

def Selection_delete(S, update_row):
    """ Attributes: 0-valid, 1-id, ...-payload """
    S[0][:] = S[0][:] * (1 - (S[1][:] == update_row[1]))
    return S

# Aggregation
def Aggregation_insert(res, values, func):
    def aggregate(values):
        if func == 'sum':
            return values.get_vector().sum()
        elif func == 'max':
            return max(values)
        elif func == 'min':
            return min(values)
    values[values.shape[0] - 1] = res
    return aggregate(values)

def Aggregation_delete(res, values):
    values.assign(-values)
    values[values.shape[0] - 1] = res
    return values.get_vector().sum()
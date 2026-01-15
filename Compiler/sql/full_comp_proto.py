from Compiler.library import *
from Compiler.sorting import radix_sort
from Compiler.types import *
from Compiler.util import *

def Orderby(T, limit=None):
    """ Attributes: 0-key, 1-valid, 2-id, ...-payload """
    key, valid = 0, 1
    D = T.transpose()
    radix_sort(D.get_column(key), D)
    radix_sort(1 - D.get_column(valid), D, n_bits=1)
    return D.transpose() if limit is None else D.get_part(0, limit).transpose()

# Groupby
def aggregate(a, b, agg_type='sum'):
    if agg_type == 'sum' or agg_type == '+':
        return a + b
    elif agg_type == 'sub' or agg_type == '-':
        return a - b
    elif agg_type == 'max':
        return (a > b).if_else(a, b)
    elif agg_type == 'min':
        return (a < b).if_else(a, b)
    else:
        raise ValueError(f"Unsupported aggregation type: {agg_type}")

def note_func(p1, p2, g1, g2=None, agg_type='sum'):
    """ 
    p1: 当前节点 (Target/Right)
    p2: 来源节点 (Source/Left)
    """
    agg = aggregate(p1, p2, agg_type)
    # 逻辑: p3 = p1 + g1 * (agg - p1)
    p3 = p1 + (agg - p1) * g1
    
    if g2 is not None:
        g3 = g1 * g2
        return p3, g3
    else:
        return p3

def Brent_Kung_network(x, g, agg_type='sum'):
    """
    Brent-Kung Network for Parallel Prefix Computation.
    """
    n = len(x)
    
    # 1. 初始化结果数组
    res = Array(n, x.value_type)
    res.assign(x)
    
    g_arr = Array(n, g.value_type)
    g_arr.assign(g)

    depth = math.ceil(math.log2(n)) if n > 1 else 0

    # ----------------------------------------------------
    # 2. Up-Sweep (Reduce 阶段)
    # ----------------------------------------------------
    for j in range(depth):
        step = 1 << (j + 1)
        off = 1 << j
        
        idx_right_py = list(range(step - 1, n, step))
        if not idx_right_py:
            continue
        idx_left_py = [i - off for i in idx_right_py]

        # 转换为 regint 向量
        idx_right = regint(idx_right_py)
        idx_left = regint(idx_left_py)

        # --- Gather (读取) ---
        p_right = res.get(idx_right)
        p_left = res.get(idx_left)
        g_right = g_arr.get(idx_right)
        g_left = g_arr.get(idx_left)

        # --- Compute (计算) ---
        p_new, g_new = note_func(p_right, p_left, g_right, g_left, agg_type)

        # --- Scatter (写回) ---
        addr_right_res = res.address + idx_right
        addr_right_g = g_arr.address + idx_right
        
        p_new.store_in_mem(addr_right_res)
        g_new.store_in_mem(addr_right_g)
        
        # 【关键修正】强制同步内存，防止指令重排导致读取旧数据
        break_point()

    # ----------------------------------------------------
    # 3. Down-Sweep (Distribute 阶段)
    # ----------------------------------------------------
    for j in range(depth - 2, -1, -1):
        step = 1 << (j + 1)
        dist = 1 << j
        
        idx_roots_py = []
        idx_children_py = []
        
        # 边界检查
        for i in range(step - 1, n, step):
            target = i + dist
            if target < n:
                idx_roots_py.append(i)
                idx_children_py.append(target)
        
        if not idx_children_py:
            continue

        idx_roots = regint(idx_roots_py)
        idx_children = regint(idx_children_py)

        # --- Gather ---
        p_root = res.get(idx_roots)
        p_child = res.get(idx_children)
        g_child = g_arr.get(idx_children)

        # --- Compute ---
        p_new_child = note_func(p_child, p_root, g_child, None, agg_type)

        # --- Scatter ---
        addr_children_res = res.address + idx_children
        p_new_child.store_in_mem(addr_children_res) # type: ignore
        
        # 【关键修正】强制同步内存
        break_point()
    return res

def Groupby(T):
    """ Attributes: 0-key, 1-value, 2-valid, 3-result """
    key, value, valid = 0, 1, 2
    n = T.shape[1]
    G = T.transpose()
    radix_sort(G.get_column(key), G)
    radix_sort(1 - G.get_column(valid), G, n_bits=1)

    g = Array(n, sint)
    g[1:n] = G[key][0:n-1] == G[key][1:n]

    G[valid][0:n-1] = G[valid][0:n-1] * (1 - g[1:n])

    G[value] = Brent_Kung_network(G[value], g)
    G.secure_shuffle()
    return G

def Join(A, B):
    """ Attributes of A and B: 0-key, 1-valid, ...-payload_A/B
        Attributes of J: 0-key, 1-valid, ...-payload_A, ...-payload_B """
    key, valid = 0, 1
    a_A, a_B = A.shape[0], B.shape[0]
    a_J, n_J = (a_A + a_B - 2), (A.shape[1] + B.shape[1])
    
    # Construct J by concatenating B and A
    J = Matrix(n_J, max(a_A, a_B), A.value_type)    
    @for_range(B.shape[1])
    def _(i):
        J[i] = B.get_column(i)
    @for_range(A.shape[1])
    def _(i):
        J[B.shape[1] + i] = A.get_column(i)

    radix_sort(J.get_column(key), J)

    # Extend J by enough columns
    J_prime = Matrix(a_J, n_J, A.value_type)    
    J_prime.assign(J.transpose())

    # Reset valid bits
    J_prime[valid][0] = sint(0)
    J_prime[valid][1:n_J] = J_prime[key][1:n_J] == J_prime[key][0 : n_J-1]

    # Propagate payloads of B from the previous rows
    for i in range(a_B - 2):
        J_prime[a_A + i][1:n_J] = if_else(J_prime[valid][1:n_J], J_prime[2 + i][0 : n_J-1], J_prime[a_A + i][1:n_J]) # J_prime[valid][1:n_J] * J_prime[2 + i][0 : n_J-1]
    
    # Shuffle
    J_prime = J_prime.transpose()
    J_prime.secure_shuffle()
    return J_prime.transpose()

def Selection(S):
    """ Attributes: 0-valid, 1-id, ...-payload """
    valid = 0
    def predicate(columns):
        # Example predicate:
        return columns[:] > sint(10)
    S[valid][:] = S[valid][:] * predicate(S[2])
    return S

def Aggregation(values, func):
    def aggregate(values):
        if func == 'sum':
            return values.get_vector().sum()
        elif func == 'max':
            return max(values)
        elif func == 'min':
            return min(values)
    return aggregate(values)
from Compiler.library import *

def reverse_matrix_by_rows(M):
    """ 按行反转矩阵M """
    R = M.same_shape()
    rows = M.shape[0]
    @for_range_opt(rows)
    def _(i):
        R[i][:] = M[rows - 1 - i][:]
    return R
import math

class ComplexityModel:
    """
    细粒度复杂度估算模型
    返回元组: (Ops, Bits, Rounds)
    """

    @staticmethod
    def _sort_complexity(n):
        # 基于 Bitonic Sort 的估算
        if n <= 1: return 0, 0, 0
        log_n = math.ceil(math.log2(n))
        # 比较次数 approx 0.5 * n * log^2(n)
        comparisons = 0.5 * n * (log_n ** 2)
        
        ops = comparisons * 10      # 每次比较的计算开销
        bits = comparisons * 128    # 每次比较的通信比特数 (secret sharing)
        rounds = (log_n ** 2) * 10  # 深度
        return ops, bits, rounds

    @staticmethod
    def get_fc_cost(op_type, n):
        """Full Computation: 全量计算"""
        if op_type == 'Sort' or op_type == 'Orderby':
            return ComplexityModel._sort_complexity(n)
        elif op_type == 'Join':
            # 假设 Sort-Merge Join: 2 * Sort + Scan
            ops_s, bits_s, rnd_s = ComplexityModel._sort_complexity(n)
            return ops_s * 2 + n * 5, bits_s * 2 + n * 64, rnd_s * 2 + 10
        elif op_type == 'Groupby':
            # Sort + Aggregate
            ops_s, bits_s, rnd_s = ComplexityModel._sort_complexity(n)
            return ops_s + n * 5, bits_s + n * 64, rnd_s + 10
        return 0, 0, 0

    @staticmethod
    def get_sic_cost(op_type, n, k):
        """Sequential Incremental: 逐条增量"""
        # 假设线性扫描或 ORAM 访问，每条更新 O(n) 或 O(log n)
        # 这里简化为线性扫描模型
        ops_one = n * 2
        bits_one = n * 32
        rounds_one = 5
        
        # k 次操作
        return k * ops_one, k * bits_one, k * rounds_one

    @staticmethod
    def get_bic_cost(op_type, n, k):
        """Batch Incremental: 批量增量"""
        # 通常涉及对 k 个更新排序，然后与 n 进行合并
        ops_sort_k, bits_sort_k, rnd_sort_k = ComplexityModel._sort_complexity(k)
        
        # 合并代价 O(n + k)
        ops_merge = (n + k) * 2
        bits_merge = (n + k) * 32
        rounds_merge = 10
        
        return (ops_sort_k + ops_merge, 
                bits_sort_k + bits_merge, 
                rnd_sort_k + rounds_merge)
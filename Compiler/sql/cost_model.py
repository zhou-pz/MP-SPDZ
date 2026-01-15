from .config import SystemParams, ModelParams
from .complexity import ComplexityModel

class CostPredictor:
    @staticmethod
    def predict(op_type, protocol, n, k=0):
        """
        预测特定协议的端到端延迟
        """
        ops, bits, rounds = 0, 0, 0
        
        # 1. 获取特征 (Feature Extraction)
        if protocol == 'FC':
            # 全量计算通常处理 n + k 的数据总量
            ops, bits, rounds = ComplexityModel.get_fc_cost(op_type, n + k)
        elif protocol == 'SIC':
            ops, bits, rounds = ComplexityModel.get_sic_cost(op_type, n, k)
        elif protocol == 'BIC':
            ops, bits, rounds = ComplexityModel.get_bic_cost(op_type, n, k)
            
        # 2. 应用代价公式 (Cost Model)
        # T = alpha * (Ops/f) + beta * (Bits/b) + gamma * (Rounds * d)
        
        f = SystemParams.cpu_freq
        b = SystemParams.bandwidth
        d = SystemParams.latency
        
        alpha = ModelParams.alpha
        beta = ModelParams.beta
        gamma = ModelParams.gamma
        
        t_comp = alpha * (ops / f)
        t_comm_bw = beta * (bits / b)
        t_comm_lat = gamma * (rounds * d)
        
        total_time = t_comp + t_comm_bw + t_comm_lat
        
        # 可选：添加 InputRefresh 开销
        if protocol == 'FC':
            # 假设删除操作需要 SetDummies 开销，这里简化处理
            pass 
            
        return total_time
import os
import subprocess
import re
import platform

# 尝试导入 psutil 用于获取 CPU 信息 (pip install psutil)
try:
    import psutil
except ImportError:
    psutil = None

class TCNetworkMonitor:
    """
    专门用于读取 Linux TC (Traffic Control) 配置的网络监视器
    适配 tbf + netem 的脚本结构
    """
    
    @staticmethod
    def get_network_params(interface="lo"):
        """
        返回: (bandwidth_bps, rtt_latency_sec)
        - bandwidth_bps: 带宽 (bits/second)
        - rtt_latency_sec: 往返延迟 (seconds) -> 注意：TC设置通常是单向，这里会乘以2
        """
        # 默认兜底值 (如果未设置TC，认为是无限快)
        bw = 100 * 10**9   # 100 Gbps
        delay = 0.00005    # 0.05 ms (50us)
        
        try:
            # 执行 tc 命令查看状态
            # 不需要 sudo，普通用户通常可以运行 show
            output = subprocess.check_output(
                ["tc", "qdisc", "show", "dev", interface], 
                universal_newlines=True,
                stderr=subprocess.STDOUT
            )
            
            # --- 1. 解析带宽 (TBF rate) ---
            # 匹配模式: rate 500Mbit 或 rate 10000Mbit
            # tc 输出单位通常是: bit, Kbit, Mbit, Gbit
            bw_match = re.search(r'rate (\d+\.?\d*)([KMG]?bit)', output)
            if bw_match:
                value = float(bw_match.group(1))
                unit = bw_match.group(2)
                bw = TCNetworkMonitor._convert_bw_to_bps(value, unit)

            # --- 2. 解析延迟 (Netem delay) ---
            # 匹配模式: delay 10.0ms 或 delay 150us
            lat_match = re.search(r'delay (\d+\.?\d*)([mu]?s)', output)
            if lat_match:
                value = float(lat_match.group(1))
                unit = lat_match.group(2)
                one_way_latency = TCNetworkMonitor._convert_lat_to_sec(value, unit)
                
                # 重要：TC netem 设置的是"单向"出站延迟
                # 对于本地回环通信 (lo)，数据包出去(delay)再回来(通常不经过tc或再经过一次)
                # 假设 A->B (lo output) 受 TC 限制。
                # 在 lo 接口上，tc root 对所有流出的包生效。
                # A发给B (delay)，B发回给A (delay)。
                # 所以 RTT ≈ 2 * delay
                delay = one_way_latency * 2
                
        except Exception as e:
            print(f"[TC Monitor] Warning: Could not read tc params: {e}")
            
        return bw, delay

    @staticmethod
    def _convert_bw_to_bps(value, unit):
        """将 tc 的带宽单位转换为 bits/second"""
        unit = unit.lower()
        if 'gbit' in unit:
            return value * 10**9
        elif 'mbit' in unit:
            return value * 10**6
        elif 'kbit' in unit:
            return value * 10**3
        else: # bit
            return value

    @staticmethod
    def _convert_lat_to_sec(value, unit):
        """将 tc 的时间单位转换为 seconds"""
        unit = unit.lower()
        if unit == 'ms':
            return value / 1000.0
        elif unit == 'us':
            return value / 1000000.0
        elif unit == 's':
            return value
        return value

class SystemParams:
    """
    系统环境参数
    b: Bandwidth (bits/second)
    d: Latency (seconds)
    f: CPU Frequency (Hz)
    """
    bandwidth = 100 * 10**6  # Default: 100 Mbps
    latency = 0.02           # Default: 20 ms
    cpu_freq = 3 * 10**9     # Default: 3 GHz

    @staticmethod
    def set_params(b, d, f):
        SystemParams.bandwidth = b
        SystemParams.latency = d
        SystemParams.cpu_freq = f

    @staticmethod
    def collect_realtime_params():
        """
        尝试实时采集系统参数。
        注意：带宽通常难以瞬间测量，建议通过环境变量 MPC_BANDWIDTH 配置。
        
        :param target_host: 用于测试网络延迟的目标主机 IP (例如 MPC 的另一方)
        """
        # 1. 获取 CPU 频率
        if psutil:
            try:
                freq = psutil.cpu_freq()
                if freq:
                    # psutil 返回 MHz，转换为 Hz
                    # 优先使用 current，如果为 0 (某些虚拟化环境) 则使用 max
                    current_freq = freq.current if freq.current > 0 else freq.max
                    if current_freq > 0:
                        SystemParams.cpu_freq = current_freq * 10**6
            except Exception as e:
                print(f"[Config] Warning: Failed to get CPU freq: {e}")
        
        # 2. 获取网络延迟 (Ping)
        interface="lo"
        """直接从 TC 读取网络环境"""
        bw, rtt = TCNetworkMonitor.get_network_params(interface)
        
        SystemParams.bandwidth = bw
        SystemParams.latency = rtt
        
        print(f"[Config] Loaded TC Params for '{interface}':")
        print(f"         Bandwidth : {bw / 10**6:.2f} Mbps")
        print(f"         RTT Latency: {rtt * 1000:.3f} ms (Calculated as 2 * one-way)")

class ModelParams:
    """
    代价模型回归系数 (Alpha, Beta, Gamma)
    用于校准理论公式与实际运行时间的偏差
    """
    alpha = 1.0 # 计算权重
    beta = 1.0  # 通信带宽权重
    gamma = 1.0 # 通信轮数权重

    @staticmethod
    def load_weights(a, b, g):
        ModelParams.alpha = a
        ModelParams.beta = b
        ModelParams.gamma = g
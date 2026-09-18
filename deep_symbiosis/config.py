"""
DeepSeek-PP & Harness 深水区融合系统 - 全局配置
定义系统常量、枚举和基础参数，消除魔法数字。
所有文件严格控制在 600 行以内。
"""
from enum import Enum
from dataclasses import dataclass

# ================= 系统常量 =================
MAX_FILE_LINES = 600
GPU_MEMORY_GB = 80
BLOCK_SIZE_KB = 16
NUM_NODES = 64
GPUS_PER_NODE = 8
TOTAL_GPUS = NUM_NODES * GPUS_PER_NODE

# ================= 拓扑常量 =================
NVLINK_BANDWIDTH = 900  # GB/s
PCIE_BANDWIDTH = 64     # GB/s
ROCE_BANDWIDTH = 200    # GB/s
LATENCY_NS = {"NVLINK": 500, "PCIE": 2000, "ROCE": 5000}

# ================= 调度常量 =================
TARGET_FLOPS_PER_MB = 1e12
MAX_TOKEN_VARIANCE = 0.1
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 32

# ================= 容错常量 =================
HEALTH_CHECK_INTERVAL = 1.0
DEGRADED_THRESHOLD_LATENCY = 1.5
HYSTERESIS_WINDOW_SEC = 5.0
RECOVERY_WINDOW_SEC = 10.0

# ================= 系统模式 =================
class SystemMode(Enum):
    TRAIN = "TRAIN"
    EVAL = "EVAL"
    INFERENCE = "INFERENCE"

# ================= 内存状态 =================
class MemoryState(Enum):
    FREE = "FREE"
    ACTIVE_ACTIVATION = "ACTIVE_ACTIVATION"
    ACTIVE_KVCACHE = "ACTIVE_KVCACHE"
    EVICTABLE = "EVICTABLE"

# ================= 健康度状态 (FSM) =================
class HealthState(Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"

# ================= 异常类型 =================
class AnomalyType(Enum):
    LATENCY_SPIKE = "LATENCY_SPIKE"
    COMPUTE_SLOWDOWN = "COMPUTE_SLOWDOWN"
    PACKET_LOSS = "PACKET_LOSS"
    XID_ERROR = "XID_ERROR"

# ================= 链接类型 =================
class LinkType(Enum):
    NVLINK = "NVLINK"
    PCIE = "PCIE"
    ROCE = "ROCE"

# ================= 硬件配置 =================
@dataclass
class HardwareConfig:
    num_nodes: int = NUM_NODES
    gpus_per_node: int = GPUS_PER_NODE
    memory_gb: int = GPU_MEMORY_GB

# ================= 全局配置单例 =================
CONFIG = HardwareConfig()

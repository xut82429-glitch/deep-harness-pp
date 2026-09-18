"""
State Bus 数据模型 - Protobuf-like 数据结构
控制面与执行面通信协议
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time

@dataclass
class MicroBatchProfile:
    """Micro-batch 元数据"""
    mb_id: int
    estimated_flops: float
    seq_length: int
    token_count: int

@dataclass
class TopologyMatrix:
    """物理拓扑代价矩阵"""
    nodes: List[int]
    bandwidth_matrix: List[List[float]]
    latency_matrix: List[List[float]]
    nvlink_domains: List[List[int]]

@dataclass
class HardwareAnomaly:
    """硬件微异常报告"""
    node_id: int
    gpu_id: int
    anomaly_type: str
    severity: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class PPControlCommand:
    """控制面 -> 执行面：任务指令"""
    task_id: str
    mode: str  # TRAIN/EVAL/INFERENCE
    memory_budget_gb: float
    mb_profiles: List[MicroBatchProfile] = field(default_factory=list)
    topo_matrix: Optional[TopologyMatrix] = None

@dataclass
class PPExecutionState:
    """执行面 -> 控制面：执行状态"""
    task_id: str
    stage_latency_ms: Dict[int, float] = field(default_factory=dict)
    bubble_rate: float = 0.0
    memory_usage_gb: float = 0.0
    anomalies: List[HardwareAnomaly] = field(default_factory=list)
    health_state: str = "NORMAL"

@dataclass
class MemoryRequest:
    """统一内存请求"""
    request_id: str
    size_bytes: int
    mem_type: str  # ACTIVATION/KV_CACHE
    owner: str  # PP_ENGINE/HARNESS

@dataclass
class MemoryResponse:
    """统一内存响应"""
    request_id: str
    success: bool
    block_ids: List[int] = field(default_factory=list)
    reused_zero_copy: bool = False

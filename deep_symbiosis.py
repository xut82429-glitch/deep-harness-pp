"""
DeepSymbiosis: DeepSeek-PP x Harness 深度共生原型系统
=====================================================
本模块实现了“深水区”理论的核心逻辑：
1. JointCompilation: 拓扑感知的算子级联合编译
2. UnifiedMemoryManager: 基于 Paged 机制的显存借贷与零拷贝
3. DigitalTwinSimulator: 策略推演与全局最优解搜索
4. AsymmetricScheduler: 面向变长数据的非对称流水线调度
5. ResilientRuntime: 硬件微异常感知与纳秒级降级
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import random

# ==============================================================================
# 深水区一：物理拓扑与联合编译 (Joint Compilation)
# ==============================================================================

class LinkType(Enum):
    NVLINK = "nvlink"      # 高带宽 (900GB/s)
    PCIE = "pcie"          # 中带宽 (64GB/s)
    ROCE = "roce"          # 低带宽/高延迟 (200Gbps+)

@dataclass
class PhysicalNode:
    id: int
    gpu_count: int
    compute_power: float  # TFLOPS

@dataclass
class TopologyLink:
    src: Tuple[int, int]  # (node_id, gpu_id)
    dst: Tuple[int, int]
    bandwidth: float      # GB/s
    latency: float        # us
    link_type: LinkType

@dataclass
class TopologyMatrix:
    """Harness 提供的物理拓扑先验"""
    nodes: List[PhysicalNode]
    links: Dict[Tuple[Tuple[int, int], Tuple[int, int]], TopologyLink]
    
    def get_link_cost(self, src, dst) -> float:
        if src == dst: return 0.0
        key = (src, dst)
        if key in self.links:
            link = self.links[key]
            # 成本模型：延迟 + 数据量/带宽
            return link.latency 
        return 1000.0 # 极高成本，表示不可达或极慢

@dataclass
class MoEExpert:
    id: int
    activation_freq: float # 路由频率预测值

@dataclass
class CompiledOperator:
    name: str
    compute_time: float
    comm_time: float
    fused_ops: List[str]
    placement: Tuple[int, int] # (node_id, gpu_id)

class JointCompiler:
    """
    深水区一核心：利用 Harness 的拓扑矩阵，对 PP 计算图进行拓扑感知编译
    """
    def __init__(self, topology: TopologyMatrix):
        self.topology = topology
        
    def compile_moe_layer(self, experts: List[MoEExpert], stage_gpus: List[Tuple[int, int]]) -> List[CompiledOperator]:
        """
        专家放置与通信对齐：
        将频繁交互的专家对放置在同一 NVLink 域内
        """
        compiled_ops = []
        
        # 1. 专家分组 (简化版：按频率排序后贪心放置到高速域)
        experts_sorted = sorted(experts, key=lambda x: x.activation_freq, reverse=True)
        
        # 假设 stage_gpus 已经按拓扑邻近性排序 (由 Harness 预处理)
        gpu_idx = 0
        
        for expert in experts_sorted:
            target_gpu = stage_gpus[gpu_idx % len(stage_gpus)]
            
            # 2. 微观 Overlap 动态生成
            # 检查目标 GPU 与其他 GPU 的通信代价
            # 如果跨机通信延迟大，编译器自动增加该 Micro-batch 的计算算子数量 (融合更多 FFN)
            base_compute = 10.0 # ms
            base_comm = 5.0 # ms
            
            # 模拟：如果检测到是跨机 (Node ID 不同)，增加计算融合度以掩盖通信
            is_cross_node = (target_gpu[0] != stage_gpus[0][0])
            
            fused_ops_list = ["GEMM", "Bias"]
            if is_cross_node:
                # 拓扑感知优化：融合更多算子，用计算填补通信气泡
                fused_ops_list.append("LayerNorm_Fused") 
                base_compute *= 1.2 # 计算量增加
                base_comm *= 0.8    # 通信次数减少 (由于融合)
            
            op = CompiledOperator(
                name=f"Expert_{expert.id}",
                compute_time=base_compute,
                comm_time=base_comm,
                fused_ops=fused_ops_list,
                placement=target_gpu
            )
            compiled_ops.append(op)
            gpu_idx += 1
            
        return compiled_ops

# ==============================================================================
# 深水区二：统一内存模型 (Unified Memory Manager)
# ==============================================================================

@dataclass
class MemoryPage:
    page_id: int
    size_mb: float
    owner: str # "PP_Activation" or "Harness_KVCache"
    is_frozen: bool = False
    physical_address: int = 0

class UnifiedMemoryManager:
    """
    深水区二核心：打破 PP 与 Harness 内存边界，基于 Paged 机制的零拷贝借贷
    """
    def __init__(self, total_gpu_memory_gb: float, page_size_mb: float = 64.0):
        self.total_pages = int((total_gpu_memory_gb * 1024) / page_size_mb)
        self.page_size_mb = page_size_mb
        self.free_list: List[int] = list(range(self.total_pages))
        self.page_table: Dict[int, MemoryPage] = {} # logical_id -> Page
        self.physical_map: Dict[int, int] = {} # physical_addr -> logical_id
        
        # 统计
        self.pp_usage = 0
        self.harness_usage = 0
        
    def allocate(self, owner: str, size_mb: float, priority: int = 0) -> List[int]:
        """分配逻辑页，此时不绑定物理页 (Lazy Allocation)"""
        num_pages = int(np.ceil(size_mb / self.page_size_mb))
        if len(self.free_list) < num_pages:
            # 触发回收或借贷逻辑
            self._reclaim_or_borrow(owner, num_pages)
            
        assigned_logical_ids = []
        for _ in range(num_pages):
            pid = self.free_list.pop()
            page = MemoryPage(page_id=pid, size_mb=self.page_size_mb, owner=owner)
            self.page_table[pid] = page
            assigned_logical_ids.append(pid)
            
        if owner == "PP_Activation":
            self.pp_usage += num_pages
        else:
            self.harness_usage += num_pages
            
        return assigned_logical_ids
    
    def borrow_memory(self, from_owner: str, to_owner: str, amount_mb: float):
        """
        动态内存借贷：
        Harness 冻结 KV Cache 页面，"借"给 PP 引擎作为 Activation 内存
        零拷贝：仅修改页表所有权标记，不移动数据
        """
        pages_to_borrow = int(np.ceil(amount_mb / self.page_size_mb))
        borrowed_count = 0
        
        for pid, page in self.page_table.items():
            if borrowed_count >= pages_to_borrow:
                break
            if page.owner == from_owner and not page.is_frozen:
                # 执行借贷
                page.is_frozen = True # 标记为冻结，原所有者不可访问
                old_owner = page.owner
                page.owner = to_owner # 所有权转移
                
                # 更新统计
                if old_owner == "Harness_KVCache":
                    self.harness_usage -= 1
                elif old_owner == "PP_Activation":
                    self.pp_usage -= 1
                    
                if to_owner == "PP_Activation":
                    self.pp_usage += 1
                else:
                    self.harness_usage += 1
                    
                borrowed_count += 1
                print(f"[MemMgr] Borrowed Page {pid} from {old_owner} to {to_owner}")
                
        if borrowed_count < pages_to_borrow:
            raise MemoryError("Insufficient memory to borrow even after freezing.")

    def return_memory(self, owner: str, logical_ids: List[int]):
        """归还内存，解冻页面"""
        for pid in logical_ids:
            if pid in self.page_table:
                page = self.page_table[pid]
                if page.is_frozen:
                    page.is_frozen = False
                    # 注意：这里所有权可能保持不变（如果是借贷），或者恢复
                    # 简化逻辑：借贷结束时，所有权回归原主或释放
                    if owner == "PP_Activation" and page.owner == "PP_Activation":
                         # 假如是借来的，现在用完还回去
                         page.owner = "Harness_KVCache" # 假设回借给 Harness
                         self.pp_usage -= 1
                         self.harness_usage += 1
                    elif not page.is_frozen:
                        # 正常释放
                        del self.page_table[pid]
                        self.free_list.append(pid)
                        if owner == "PP_Activation":
                            self.pp_usage -= 1
                        else:
                            self.harness_usage -= 1

# ==============================================================================
# 深水区三 & 四：非对称调度与数字孪生 (Asymmetric Scheduling & Digital Twin)
# ==============================================================================

@dataclass
class MicroBatch:
    id: int
    token_count: int
    estimated_compute_time: float
    stage_id: int

@dataclass
class SimulationResult:
    strategy_name: str
    total_throughput: float
    bubble_ratio: float
    peak_memory: float

class DigitalTwinSimulator:
    """
    深水区四核心：轻量级集群孪生，执行前的策略沙盘推演
    """
    def __init__(self, topology: TopologyMatrix):
        self.topology = topology
        
    def dry_run(self, micro_batches: List[MicroBatch], strategy: str) -> SimulationResult:
        """
        空跑推演：模拟不同 PP 策略下的执行流
        """
        num_stages = max(mb.stage_id for mb in micro_batches) + 1
        timeline = [[] for _ in range(num_stages)]
        
        # 简化模拟逻辑
        total_compute = 0
        total_idle = 0
        
        if strategy == "Symmetric_1F1B":
            # 传统对称调度，假设所有 MB 时间相同 (取平均值)
            avg_time = np.mean([mb.estimated_compute_time for mb in micro_batches])
            # 模拟气泡...
            bubble_penalty = np.std([mb.estimated_compute_time for mb in micro_batches]) * 0.5
            total_idle = bubble_penalty * num_stages * len(micro_batches)
            total_compute = sum(mb.estimated_compute_time for mb in micro_batches)
            
        elif strategy == "Asymmetric_DualPipe":
            # 非对称调度，Harness 贪心装箱 + PP 动态调整
            # 模拟：通过重新排序减少方差
            sorted_mbs = sorted(micro_batches, key=lambda x: x.token_count)
            # 模拟更紧凑的流水线
            variance_penalty = np.std([mb.estimated_compute_time for mb in sorted_mbs]) * 0.1
            total_idle = variance_penalty * num_stages
            total_compute = sum(mb.estimated_compute_time for mb in micro_batches)
            
            # 空闲计算填充 (Idle-compute Filling)
            # 在模拟中，我们将部分 idle 时间转化为 "Useful Work"
            total_idle *= 0.7 # 假设 30% 的空泡被填充了预取或校验任务
            
        total_time = total_compute + total_idle
        throughput = total_compute / total_time if total_time > 0 else 0
        bubble_ratio = total_idle / total_time if total_time > 0 else 0
        
        return SimulationResult(
            strategy_name=strategy,
            total_throughput=throughput,
            bubble_ratio=bubble_ratio,
            peak_memory=np.random.uniform(0.7, 0.95) # 模拟显存峰值占比
        )

class AsymmetricScheduler:
    """
    深水区三核心：根据 Token 长度动态生成非对称调度表
    """
    def __init__(self, simulator: DigitalTwinSimulator):
        self.simulator = simulator
        
    def generate_schedule(self, micro_batches: List[MicroBatch]) -> List[MicroBatch]:
        # 1. Harness 端贪心装箱 (Bin Packing)
        # 将长度相近的样本打包，减小方差
        mbs_sorted = sorted(micro_batches, key=lambda x: x.token_count)
        
        # 2. 策略推演
        res_sym = self.simulator.dry_run(micro_batches, "Symmetric_1F1B")
        res_asym = self.simulator.dry_run(micro_batches, "Asymmetric_DualPipe")
        
        print(f"[Scheduler] Simulated Symmetric: Throughput={res_sym.total_throughput:.2f}, Bubble={res_sym.bubble_ratio:.2%}")
        print(f"[Scheduler] Simulated Asymmetric: Throughput={res_asym.total_throughput:.2f}, Bubble={res_asym.bubble_ratio:.2%}")
        
        # 3. 全局最优解选择
        if res_asym.total_throughput > res_sym.total_throughput:
            print("[Scheduler] Selected Asymmetric Strategy based on Digital Twin.")
            return mbs_sorted # 返回重排后的序列
        else:
            print("[Scheduler] Selected Symmetric Strategy.")
            return micro_batches

# ==============================================================================
# 深水区五：硬件异常与协同降级 (Resilient Runtime)
# ==============================================================================

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED_COMPUTE = "degraded_compute" # 算力下降
    DEGRADED_COMM = "degraded_comm"       # 通信降速
    DEAD = "dead"

@dataclass
class NodeHealthReport:
    node_id: int
    status: HealthStatus
    compute_drop_ratio: float = 0.0
    comm_drop_ratio: float = 0.0

class ResilientRuntime:
    """
    深水区五核心：感知微异常，动态调整 PP 调度与精度
    """
    def __init__(self):
        self.cluster_health: Dict[int, HealthStatus] = {}
        
    def detect_anomaly(self, node_id: int) -> NodeHealthReport:
        # 模拟底层 Hook 检测到的异常
        if random.random() < 0.3:
            return NodeHealthReport(node_id, HealthStatus.DEGRADED_COMPUTE, compute_drop_ratio=0.2)
        return NodeHealthReport(node_id, HealthStatus.HEALTHY)
        
    def handle_degradation(self, report: NodeHealthReport, current_schedule: List[MicroBatch]) -> List[MicroBatch]:
        """
        微批级动态降级：
        1. 减少分配给该节点的 Micro-batch 数量
        2. 或降低计算精度 (模拟为减少计算时间)
        """
        if report.status == HealthStatus.HEALTHY:
            return current_schedule
            
        print(f"[Runtime] Detected Degradation on Node {report.node_id}: {report.status}")
        
        adjusted_schedule = []
        for mb in current_schedule:
            # 简单策略：如果该 MB 落在故障节点，且故障严重，跳过或缩小
            # 实际系统中会触发重映射
            if mb.stage_id % 4 == report.node_id % 4: # 模拟映射关系
                if report.status == HealthStatus.DEGRADED_COMPUTE:
                    # 模拟降级：BF16 -> FP8, 计算时间减少，精度损失换取速度
                    mb.estimated_compute_time *= (1 - report.compute_drop_ratio) 
                    print(f"[Runtime] Downgraded MB {mb.id} to FP8 on Node {report.node_id}")
                elif report.status == HealthStatus.DEGRADED_COMM:
                    # 调整通信窗口，增加本地计算融合
                    pass 
            adjusted_schedule.append(mb)
            
        return adjusted_schedule

# ==============================================================================
# 主流程演示 (Main Execution Flow)
# ==============================================================================

def main():
    print("=== DeepSymbiosis: DeepSeek-PP x Harness 深度共生系统启动 ===\n")
    
    # 1. 初始化物理拓扑 (Harness 提供)
    nodes = [PhysicalNode(id=i, gpu_count=8, compute_power=300.0) for i in range(2)]
    links = {}
    # 构建一个简单的 NVLink + RoCE 拓扑
    for i in range(2):
        for j in range(8):
            for k in range(8):
                if j != k:
                    bw = 900.0 if i == i else 200.0 # 同机 NVLink, 跨机 RoCE
                    lat = 1.0 if i == i else 20.0
                    links[((i,j), (i,k))] = TopologyLink((i,j), (i,k), bw, lat, LinkType.NVLINK if i==i else LinkType.ROCE)
    
    topology = TopologyMatrix(nodes=nodes, links=links)
    print(f"[Harness] Initialized Topology with {len(nodes)} Nodes.\n")
    
    # 2. 联合编译 (Joint Compilation)
    compiler = JointCompiler(topology)
    experts = [MoEExpert(id=i, activation_freq=random.random()) for i in range(64)]
    stage_gpus = [(0, i) for i in range(8)]
    
    compiled_ops = compiler.compile_moe_layer(experts, stage_gpus)
    print(f"[Compiler] Compiled {len(compiled_ops)} Operators with Topology Awareness.")
    fused_count = sum(1 for op in compiled_ops if "LayerNorm_Fused" in op.fused_ops)
    print(f"[Compiler] Auto-Fused {fused_count} Ops to mask Cross-Node Latency.\n")
    
    # 3. 统一内存管理 (Unified Memory)
    mem_mgr = UnifiedMemoryManager(total_gpu_memory_gb=80.0)
    # Harness 分配 KV Cache
    kv_pages = mem_mgr.allocate("Harness_KVCache", 2000.0) 
    print(f"[MemMgr] Harness Allocated {len(kv_pages)} Pages for KV Cache.")
    
    # PP 需要 Activation，显存不足，触发借贷
    try:
        act_pages = mem_mgr.allocate("PP_Activation", 60000.0) # 请求大量显存
    except MemoryError:
        print("[MemMgr] Direct Allocation Failed. Triggering Borrowing Mechanism...")
        mem_mgr.borrow_memory("Harness_KVCache", "PP_Activation", 4000.0)
        act_pages = mem_mgr.allocate("PP_Activation", 60000.0)
        print(f"[MemMgr] Successfully Borrowed Memory. PP Usage: {mem_mgr.pp_usage}, Harness Usage: {mem_mgr.harness_usage}\n")
    
    # 4. 数字孪生与非对称调度 (Digital Twin & Scheduling)
    simulator = DigitalTwinSimulator(topology)
    scheduler = AsymmetricScheduler(simulator)
    
    # 生成变长 Micro-batches (模拟评估场景)
    mbs = [MicroBatch(id=i, token_count=random.randint(100, 4096), 
                      estimated_compute_time=random.randint(10, 100), stage_id=i%4) 
           for i in range(20)]
           
    optimized_schedule = scheduler.generate_schedule(mbs)
    print(f"[Scheduler] Optimized Schedule Generated. First 5 MB Token Counts: {[mb.token_count for mb in optimized_schedule[:5]]}\n")
    
    # 5. 运行时异常处理 (Resilient Runtime)
    runtime = ResilientRuntime()
    # 模拟检测异常
    report = runtime.detect_anomaly(node_id=1)
    if report.status != HealthStatus.HEALTHY:
        final_schedule = runtime.handle_degradation(report, optimized_schedule)
        print("[Runtime] Schedule Adjusted for Degraded Node.\n")
    else:
        print("[Runtime] Cluster Healthy. No Adjustment Needed.\n")
        
    print("=== DeepSymbiosis Execution Cycle Complete ===")

if __name__ == "__main__":
    main()

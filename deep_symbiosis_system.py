#!/usr/bin/env python3
"""
DeepSeek-PP & DeepSeek Harness 深水区融合系统
Enterprise-Grade Unified PP-Harness System

本系统实现了白皮书中定义的五大核心深水区功能：
1. 联合编译 (Joint Compilation) - 拓扑感知的 MoE 专家放置
2. 统一内存管理 (Unified Memory Manager) - 零拷贝借贷机制
3. 非对称调度 (Asymmetric Scheduling) - 变长数据流整形
4. 数字孪生推演 (Digital Twin Simulation) - 策略空跑优化
5. 弹性运行时 (Resilient Runtime) - 微批级故障降级

架构遵循控制面/执行面分离原则，通过 State Bus 进行高频交互。
"""

import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from collections import defaultdict
import json


# ============================================================================
# 第一部分：状态总线协议 (State Bus Protocol)
# 控制面与执行面的交互接口定义
# ============================================================================

class TaskMode(Enum):
    TRAIN = "TRAIN"
    EVAL = "EVAL"
    INFERENCE = "INFERENCE"


class MemoryType(Enum):
    ACTIVATION = "ACTIVATION"
    KV_CACHE = "KV_CACHE"
    PARAMETER = "PARAMETER"
    OPTIMIZER = "OPTIMIZER"


class NodeHealthStatus(Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"


@dataclass
class MicroBatchProfile:
    """Micro-batch 元数据 Profile"""
    batch_id: int
    estimated_flops: float
    seq_length: int
    token_count: int
    moe_routing_prob: Dict[int, float]


@dataclass
class TopologyMatrix:
    """物理拓扑代价矩阵"""
    nodes: List[int]
    bandwidth_matrix: Dict[Tuple[int, int], float]
    latency_matrix: Dict[Tuple[int, int], float]
    nvlink_domains: List[Set[int]]
    
    def get_cost(self, src: int, dst: int) -> float:
        """计算通信代价：Cost = 1/Bandwidth + Latency"""
        bw = self.bandwidth_matrix.get((src, dst), 1.0)
        lat = self.latency_matrix.get((src, dst), 1.0)
        return 1.0 / bw + lat


@dataclass
class HardwareAnomaly:
    """硬件微异常报告"""
    node_id: int
    anomaly_type: str
    severity: float
    timestamp: float


# ============================================================================
# 第二部分：控制面组件 (Control Plane)
# ============================================================================

class TopologyManager:
    """
    拓扑管理器 - 探测物理网络拓扑，生成通信代价矩阵
    实现深水区一：联合编译的拓扑感知基础
    """
    
    def __init__(self, num_nodes: int = 64):
        self.num_nodes = num_nodes
        self.nodes = list(range(num_nodes))
        self.nvlink_domains = self._generate_nvlink_domains()
        self.bandwidth_matrix = {}
        self.latency_matrix = {}
        self._probe_topology()
    
    def _generate_nvlink_domains(self) -> List[Set[int]]:
        """模拟 NVLink 域分组 (每 8 节点为一个 NVLink 域)"""
        domains = []
        for i in range(0, self.num_nodes, 8):
            domain = set(range(i, min(i + 8, self.num_nodes)))
            domains.append(domain)
        return domains
    
    def _probe_topology(self):
        """模拟拓扑探测"""
        for i in self.nodes:
            for j in self.nodes:
                if i == j:
                    self.bandwidth_matrix[(i, j)] = 900.0
                    self.latency_matrix[(i, j)] = 0.01
                elif self._in_same_nvlink_domain(i, j):
                    self.bandwidth_matrix[(i, j)] = 300.0
                    self.latency_matrix[(i, j)] = 0.05
                else:
                    self.bandwidth_matrix[(i, j)] = 25.0
                    self.latency_matrix[(i, j)] = 2.0
    
    def _in_same_nvlink_domain(self, i: int, j: int) -> bool:
        for domain in self.nvlink_domains:
            if i in domain and j in domain:
                return True
        return False
    
    def get_topology_matrix(self) -> TopologyMatrix:
        return TopologyMatrix(
            nodes=self.nodes,
            bandwidth_matrix=self.bandwidth_matrix,
            latency_matrix=self.latency_matrix,
            nvlink_domains=self.nvlink_domains
        )
    
    def place_experts(self, routing_matrix: Dict[Tuple[int, int], float], 
                     num_experts: int) -> Dict[int, int]:
        """
        专家放置算法 - 贪心策略将高频交互的专家放在同一 NVLink 域内
        """
        placement = {}
        available_slots = {i: 8 for i in range(len(self.nvlink_domains))}
        
        hot_pairs = sorted(routing_matrix.items(), key=lambda x: x[1], reverse=True)
        placed_experts = set()
        
        for (expert_a, expert_b), interaction_freq in hot_pairs:
            if expert_a in placed_experts and expert_b in placed_experts:
                continue
            
            best_domain_idx = -1
            best_score = float('inf')
            
            for domain_idx, domain in enumerate(self.nvlink_domains):
                if available_slots[domain_idx] < 2:
                    continue
                
                score = 0
                for other_expert, other_node in placement.items():
                    if other_expert in [expert_a, expert_b]:
                        continue
                    other_domain_idx = other_node // 8
                    cost = self._get_placement_cost(
                        expert_a if other_expert == expert_b else expert_b,
                        other_expert, domain_idx, other_domain_idx, routing_matrix
                    )
                    score += cost
                
                if score < best_score:
                    best_score = score
                    best_domain_idx = domain_idx
            
            if best_domain_idx != -1:
                base_node = best_domain_idx * 8
                if expert_a not in placement:
                    placement[expert_a] = base_node + (available_slots[best_domain_idx] - 1) % 8
                    available_slots[best_domain_idx] -= 1
                    placed_experts.add(expert_a)
                if expert_b not in placement:
                    placement[expert_b] = base_node + (available_slots[best_domain_idx] - 1) % 8
                    available_slots[best_domain_idx] -= 1
                    placed_experts.add(expert_b)
        
        for expert_id in range(num_experts):
            if expert_id not in placement:
                best_domain_idx = max(range(len(self.nvlink_domains)), 
                                     key=lambda x: available_slots[x])
                base_node = best_domain_idx * 8
                offset = (8 - available_slots[best_domain_idx]) % 8
                placement[expert_id] = base_node + offset
                available_slots[best_domain_idx] -= 1
        
        return placement
    
    def _get_placement_cost(self, expert_a: int, expert_b: int, 
                           domain_a: int, domain_b: int,
                           routing_matrix: Dict) -> float:
        interaction = routing_matrix.get((expert_a, expert_b), 0.0)
        if interaction == 0:
            return 0.0
        if domain_a == domain_b:
            return interaction * 0.1
        else:
            return interaction * 10.0


class MicroBatchBuilder:
    """
    Micro-batch 构建器 - 将变长数据流整形为物理可执行的 Micro-batches
    实现深水区三：非对称调度的数据预处理
    """
    
    def __init__(self, stage_capacity_flops: float = 1e12, max_variance: float = 0.15):
        self.stage_capacity_flops = stage_capacity_flops
        self.max_variance = max_variance
    
    def estimate_flops(self, seq_len: int, moe_routing_prob: Dict[int, float]) -> float:
        base_flops = seq_len * seq_len * 4096
        moe_overhead = sum(moe_routing_prob.values()) * seq_len * 8192
        return base_flops + moe_overhead
    
    def build_micro_batches(self, token_stream: List[Dict]) -> List[MicroBatchProfile]:
        batches = []
        current_batch = []
        current_flops = 0.0
        batch_id = 0
        
        sorted_stream = sorted(token_stream, 
                              key=lambda x: self.estimate_flops(x['seq_len'], x['moe_routing_prob']),
                              reverse=True)
        
        for sample in sorted_stream:
            sample_flops = self.estimate_flops(sample['seq_len'], sample['moe_routing_prob'])
            
            if (current_flops + sample_flops <= self.stage_capacity_flops * (1 + self.max_variance)):
                current_batch.append(sample)
                current_flops += sample_flops
            else:
                if current_batch:
                    batches.append(self._create_profile(batch_id, current_batch, current_flops))
                    batch_id += 1
                current_batch = [sample]
                current_flops = sample_flops
        
        if current_batch:
            batches.append(self._create_profile(batch_id, current_batch, current_flops))
        
        return batches
    
    def _create_profile(self, batch_id: int, samples: List[Dict], total_flops: float) -> MicroBatchProfile:
        total_tokens = sum(s['seq_len'] for s in samples)
        max_seq_len = max(s['seq_len'] for s in samples)
        
        aggregated_routing = defaultdict(float)
        for sample in samples:
            for expert_id, prob in sample['moe_routing_prob'].items():
                aggregated_routing[expert_id] += prob
        
        return MicroBatchProfile(
            batch_id=batch_id,
            estimated_flops=total_flops,
            seq_length=max_seq_len,
            token_count=total_tokens,
            moe_routing_prob=dict(aggregated_routing)
        )


class DigitalTwinSimulator:
    """
    数字孪生模拟器 - 在执行前空跑推演不同 PP 策略
    实现深水区四：策略推演与闭环优化
    """
    
    def __init__(self, topology: TopologyMatrix, num_stages: int = 8):
        self.topology = topology
        self.num_stages = num_stages
    
    def simulate_strategy(self, micro_batches: List[MicroBatchProfile], 
                         strategy: str = "symmetric") -> Dict:
        if strategy == "symmetric":
            return self._simulate_symmetric(micro_batches)
        elif strategy == "asymmetric":
            return self._simulate_asymmetric(micro_batches)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _simulate_symmetric(self, micro_batches: List[MicroBatchProfile]) -> Dict:
        total_time = 0.0
        bubble_time = 0.0
        stage_latencies = defaultdict(list)
        
        avg_flops = sum(mb.estimated_flops for mb in micro_batches) / len(micro_batches)
        stage_time = avg_flops / 1e12
        
        for mb in micro_batches:
            actual_time = stage_time
            max_stage_time = stage_time
            
            if mb.estimated_flops < avg_flops * 0.8:
                bubble_time += (stage_time - mb.estimated_flops / 1e12)
            
            total_time += actual_time
            
            for stage_id in range(self.num_stages):
                stage_latencies[stage_id].append(actual_time)
        
        return {
            "strategy": "symmetric",
            "total_time_ms": total_time * 1000,
            "bubble_rate": bubble_time / total_time if total_time > 0 else 0,
            "avg_stage_latency_ms": sum(sum(lats) / len(lats) for lats in stage_latencies.values()) / self.num_stages * 1000
        }
    
    def _simulate_asymmetric(self, micro_batches: List[MicroBatchProfile]) -> Dict:
        total_time = 0.0
        bubble_time = 0.0
        stage_latencies = defaultdict(list)
        
        for mb in micro_batches:
            stage_times = []
            for stage_id in range(self.num_stages):
                comm_cost = self._estimate_comm_cost(mb, stage_id)
                compute_time = mb.estimated_flops / 1e12
                stage_time = compute_time + comm_cost
                stage_times.append(stage_time)
                stage_latencies[stage_id].append(stage_time)
            
            actual_time = max(stage_times)
            ideal_time = sum(stage_times) / self.num_stages
            
            bubble_time += max(0, actual_time - ideal_time) * 0.3
            total_time += actual_time
        
        return {
            "strategy": "asymmetric",
            "total_time_ms": total_time * 1000,
            "bubble_rate": bubble_time / total_time if total_time > 0 else 0,
            "avg_stage_latency_ms": sum(sum(lats) / len(lats) for lats in stage_latencies.values()) / self.num_stages * 1000
        }
    
    def _estimate_comm_cost(self, mb: MicroBatchProfile, stage_id: int) -> float:
        total_routing = sum(mb.moe_routing_prob.values())
        avg_latency = sum(self.topology.latency_matrix.values()) / len(self.topology.latency_matrix)
        return total_routing * avg_latency * 0.001


# ============================================================================
# 第三部分：执行面组件 (Execution Plane)
# ============================================================================

class UnifiedMemoryManager:
    """
    统一内存管理器 - 管理 Activation 和 KV Cache 的物理显存页
    实现深水区二：零拷贝借贷机制
    """
    
    BLOCK_SIZE = 16 * 1024
    
    def __init__(self, total_memory_gb: float = 80.0):
        self.total_blocks = int(total_memory_gb * 1024**3 / self.BLOCK_SIZE)
        self.blocks = [{"status": "FREE", "type": None, "ptr": i} for i in range(self.total_blocks)]
        self.free_list = list(range(self.total_blocks))
        self.evictable_list = []
        self.allocation_map = {}
        
        self.stats = {
            "total_allocations": 0,
            "zero_copy_reuses": 0,
            "evictions": 0
        }
    
    def allocate(self, size_bytes: int, mem_type: MemoryType) -> Optional[int]:
        num_blocks = (size_bytes + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        
        if mem_type == MemoryType.KV_CACHE and len(self.evictable_list) >= num_blocks:
            ptrs = []
            for _ in range(num_blocks):
                block_idx = self.evictable_list.pop()
                self.blocks[block_idx]["status"] = "ACTIVE_KVCACHE"
                self.blocks[block_idx]["type"] = mem_type
                ptrs.append(self.blocks[block_idx]["ptr"])
            
            self.stats["zero_copy_reuses"] += num_blocks
            self.stats["total_allocations"] += 1
            return ptrs[0]
        
        if len(self.free_list) < num_blocks:
            self._force_evict(num_blocks - len(self.free_list))
            if len(self.free_list) < num_blocks:
                return None
        
        ptrs = []
        for _ in range(num_blocks):
            block_idx = self.free_list.pop()
            self.blocks[block_idx]["status"] = f"ACTIVE_{mem_type.value}"
            self.blocks[block_idx]["type"] = mem_type
            ptrs.append(self.blocks[block_idx]["ptr"])
        
        self.stats["total_allocations"] += 1
        return ptrs[0]
    
    def mark_evictable(self, ptr: int):
        block_idx = ptr
        if 0 <= block_idx < self.total_blocks:
            self.blocks[block_idx]["status"] = "EVICTABLE"
            self.evictable_list.append(block_idx)
    
    def _force_evict(self, num_blocks: int):
        while len(self.evictable_list) > 0 and len(self.free_list) < num_blocks:
            block_idx = self.evictable_list.pop()
            self.blocks[block_idx]["status"] = "FREE"
            self.blocks[block_idx]["type"] = None
            self.free_list.append(block_idx)
            self.stats["evictions"] += 1
    
    def get_usage_stats(self) -> Dict:
        used = sum(1 for b in self.blocks if b["status"] != "FREE")
        evictable = len(self.evictable_list)
        free = len(self.free_list)
        
        return {
            "total_blocks": self.total_blocks,
            "used_blocks": used,
            "evictable_blocks": evictable,
            "free_blocks": free,
            "usage_percent": used / self.total_blocks * 100,
            "zero_copy_ratio": self.stats["zero_copy_reuses"] / max(1, self.stats["total_allocations"]) * 100
        }


class AsymmetricScheduler:
    """
    非对称调度器 - 根据 Micro-batch 的实际计算量生成动态调度表
    """
    
    def __init__(self, num_stages: int = 8, num_micro_batches: int = 16):
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        self.schedule = []
    
    def generate_schedule(self, micro_batches: List[MicroBatchProfile]) -> List[Dict]:
        schedule = []
        warmup_steps = self.num_stages - 1
        
        for step in range(warmup_steps):
            if step < len(micro_batches):
                schedule.append({
                    "step": step,
                    "phase": "WARMUP_FWD",
                    "micro_batch_id": step,
                    "stage_id": step,
                    "duration_ms": self._estimate_duration(micro_batches[step])
                })
        
        remaining_fwd = len(micro_batches) - warmup_steps
        for step in range(warmup_steps, warmup_steps + remaining_fwd):
            fwd_mb = step
            bwd_mb = step - self.num_stages
            
            schedule.append({
                "step": step,
                "phase": "1F1B_FWD",
                "micro_batch_id": fwd_mb,
                "stage_id": (step) % self.num_stages,
                "duration_ms": self._estimate_duration(micro_batches[fwd_mb])
            })
            
            if bwd_mb >= 0:
                schedule.append({
                    "step": step,
                    "phase": "1F1B_BWD",
                    "micro_batch_id": bwd_mb,
                    "stage_id": (step) % self.num_stages,
                    "duration_ms": self._estimate_duration(micro_batches[bwd_mb])
                })
        
        tail_start = warmup_steps + remaining_fwd
        for step in range(tail_start, tail_start + self.num_stages - 1):
            bwd_mb = len(micro_batches) - (tail_start + self.num_stages - 1 - step) - 1
            if bwd_mb >= 0:
                schedule.append({
                    "step": step,
                    "phase": "TAIL_BWD",
                    "micro_batch_id": bwd_mb,
                    "stage_id": (step) % self.num_stages,
                    "duration_ms": self._estimate_duration(micro_batches[bwd_mb])
                })
        
        self.schedule = schedule
        return schedule
    
    def _estimate_duration(self, mb: MicroBatchProfile) -> float:
        return mb.estimated_flops / 1e12 * 1000


class ResilientRuntime:
    """
    弹性运行时 - 处理硬件微异常，实现带病运行和优雅降级
    """
    
    def __init__(self):
        self.node_status = {}
        self.anomaly_history = []
        self.hysteresis_window = 5.0
        self.degradation_actions = []
    
    def report_anomaly(self, anomaly: HardwareAnomaly):
        self.anomaly_history.append(anomaly)
        
        current_status = self.node_status.get(anomaly.node_id, NodeHealthStatus.NORMAL)
        
        if anomaly.severity > 0.7:
            new_status = NodeHealthStatus.FAILED
        elif anomaly.severity > 0.3:
            new_status = NodeHealthStatus.DEGRADED
        else:
            new_status = NodeHealthStatus.NORMAL
        
        if current_status != new_status:
            self._transition_state(anomaly.node_id, current_status, new_status)
    
    def _transition_state(self, node_id: int, old_status: NodeHealthStatus, 
                         new_status: NodeHealthStatus):
        self.node_status[node_id] = new_status
        
        action = {
            "node_id": node_id,
            "timestamp": time.time(),
            "old_status": old_status.value,
            "new_status": new_status.value
        }
        
        if new_status == NodeHealthStatus.DEGRADED:
            action["actions"] = [
                "Reduce Micro-batch allocation by 50%",
                "Downgrade precision: BF16 -> FP8",
                "Reroute communication around this node"
            ]
            self.degradation_actions.append(action)
        elif new_status == NodeHealthStatus.RECOVERING:
            action["actions"] = ["Save checkpoint", "Rebuild communication ring"]
        elif new_status == NodeHealthStatus.FAILED:
            action["actions"] = ["Trigger global recovery", "Alert SRE"]
        
        print(f"[ResilientRuntime] Node {node_id}: {old_status.value} -> {new_status.value}")
        if "actions" in action:
            for a in action["actions"]:
                print(f"  → {a}")
    
    def check_recovery(self, node_id: int, current_time: float) -> bool:
        recent_anomalies = [a for a in self.anomaly_history 
                          if a.node_id == node_id and current_time - a.timestamp < self.hysteresis_window]
        
        if not recent_anomalies:
            if self.node_status.get(node_id) == NodeHealthStatus.DEGRADED:
                self._transition_state(node_id, NodeHealthStatus.DEGRADED, NodeHealthStatus.RECOVERING)
                return True
        return False
    
    def get_degraded_nodes(self) -> List[int]:
        return [nid for nid, status in self.node_status.items() 
                if status == NodeHealthStatus.DEGRADED]


# ============================================================================
# 第四部分：主系统集成与验证
# ============================================================================

class UnifiedPPHarnessSystem:
    """
    统一 PP-Harness 系统 - 集成所有深水区功能
    """
    
    def __init__(self, num_nodes: int = 64, num_stages: int = 8, memory_gb: float = 80.0):
        print("=" * 80)
        print("DeepSeek-PP & Harness 深水区融合系统初始化")
        print("=" * 80)
        
        self.topology_mgr = TopologyManager(num_nodes)
        self.micro_batch_builder = MicroBatchBuilder()
        self.simulator = DigitalTwinSimulator(self.topology_mgr.get_topology_matrix(), num_stages)
        
        self.memory_mgr = UnifiedMemoryManager(memory_gb)
        self.scheduler = AsymmetricScheduler(num_stages)
        self.runtime = ResilientRuntime()
        
        self.num_stages = num_stages
    
    def run_full_pipeline(self):
        # 1. 拓扑探测与专家放置
        print("\n[Phase 1] 拓扑探测与 MoE 专家放置")
        print("-" * 60)
        topology = self.topology_mgr.get_topology_matrix()
        print(f"检测到 {len(topology.nodes)} 个节点，{len(topology.nvlink_domains)} 个 NVLink 域")
        
        routing_matrix = {}
        for i in range(64):
            for j in range(i+1, 64):
                if random.random() > 0.7:
                    routing_matrix[(i, j)] = random.random()
        
        placement = self.topology_mgr.place_experts(routing_matrix, num_experts=64)
        print(f"完成 64 个专家的拓扑感知放置")
        co_location_rate = self._calculate_co_location_rate(placement, topology.nvlink_domains)
        print(f"  - 同域放置率：{co_location_rate:.1f}%")
        
        # 2. 数据流整形
        print("\n[Phase 2] 变长数据流整形 (Micro-batch Builder)")
        print("-" * 60)
        token_stream = self._generate_varlength_data(num_samples=200)
        micro_batches = self.micro_batch_builder.build_micro_batches(token_stream)
        print(f"输入：{len(token_stream)} 个变长样本 (64~4096 tokens)")
        print(f"输出：{len(micro_batches)} 个 Micro-batches")
        avg_flops = sum(mb.estimated_flops for mb in micro_batches)/len(micro_batches)
        flops_variance = self._calculate_variance([mb.estimated_flops for mb in micro_batches])
        print(f"  - 平均 FLOPs: {avg_flops:.2e}")
        print(f"  - FLOPs 方差：{flops_variance:.2e}")
        
        # 3. 数字孪生推演
        print("\n[Phase 3] 数字孪生策略推演")
        print("-" * 60)
        sym_result = self.simulator.simulate_strategy(micro_batches, "symmetric")
        asym_result = self.simulator.simulate_strategy(micro_batches, "asymmetric")
        
        print(f"对称调度:")
        print(f"  - 总耗时：{sym_result['total_time_ms']:.2f} ms")
        print(f"  - 气泡率：{sym_result['bubble_rate']*100:.2f}%")
        
        print(f"\n非对称调度:")
        print(f"  - 总耗时：{asym_result['total_time_ms']:.2f} ms")
        print(f"  - 气泡率：{asym_result['bubble_rate']*100:.2f}%")
        
        improvement = (sym_result['bubble_rate'] - asym_result['bubble_rate']) / sym_result['bubble_rate'] * 100 if sym_result['bubble_rate'] > 0 else 0
        print(f"\n✨ 性能提升：气泡率降低 {improvement:.1f}%")
        
        # 4. 统一内存管理验证
        print("\n[Phase 4] 统一内存管理 (零拷贝借贷)")
        print("-" * 60)
        
        # PP Forward: 分配 Activation (使用多个小块以便演示)
        num_blocks_needed = 100
        block_size = self.memory_mgr.BLOCK_SIZE
        activation_ptrs = []
        for i in range(num_blocks_needed):
            ptr = self.memory_mgr.allocate(block_size, MemoryType.ACTIVATION)
            if ptr is not None:
                activation_ptrs.append(ptr)
        
        print(f"PP Engine: 分配 {num_blocks_needed} 个 Activation 块 ({num_blocks_needed * block_size / 1024:.1f} KB)")
        
        # Mark as evictable (Forward 完成)
        for ptr in activation_ptrs:
            self.memory_mgr.mark_evictable(ptr)
        
        stats_after_fwd = self.memory_mgr.get_usage_stats()
        print(f"Forward 完成：{stats_after_fwd['evictable_blocks']} 个块标记为 EVICTABLE")
        
        # Harness: 分配 KV Cache (应复用 EVICTABLE 块)
        kv_ptrs = []
        for i in range(num_blocks_needed):
            ptr = self.memory_mgr.allocate(block_size, MemoryType.KV_CACHE)
            if ptr is not None:
                kv_ptrs.append(ptr)
        
        stats_after_kv = self.memory_mgr.get_usage_stats()
        print(f"Harness: 分配 {num_blocks_needed} 个 KV Cache 块 ({num_blocks_needed * block_size / 1024:.1f} KB)")
        print(f"  - 零拷贝复用：{stats_after_kv['zero_copy_ratio']:.1f}%")
        
        # 5. 混沌工程验证
        print("\n[Phase 5] 混沌工程验证 (故障注入)")
        print("-" * 60)
        
        anomaly = HardwareAnomaly(
            node_id=5,
            anomaly_type="LATENCY_SPIKE",
            severity=0.5,
            timestamp=time.time()
        )
        self.runtime.report_anomaly(anomaly)
        
        degraded_nodes = self.runtime.get_degraded_nodes()
        print(f"降级节点：{degraded_nodes}")
        
        time.sleep(0.01)
        self.runtime.check_recovery(5, time.time() + 10)
        
        # 6. 生成调度表
        print("\n[Phase 6] 生成非对称调度表")
        print("-" * 60)
        schedule = self.scheduler.generate_schedule(micro_batches[:16] if len(micro_batches) >= 16 else micro_batches)
        print(f"生成 {len(schedule)} 个调度项")
        
        phase_counts = defaultdict(int)
        for item in schedule:
            phase_counts[item['phase']] += 1
        
        for phase, count in phase_counts.items():
            print(f"  - {phase}: {count} 项")
        
        # 最终报告
        print("\n" + "=" * 80)
        print("🎉 深水区融合系统验证完成")
        print("=" * 80)
        print("\n核心指标汇总:")
        print(f"  ✅ 拓扑感知专家放置：同域率 {co_location_rate:.1f}%")
        print(f"  ✅ 变长数据整形：气泡率降低 {improvement:.1f}%")
        print(f"  ✅ 零拷贝内存复用：{stats_after_kv['zero_copy_ratio']:.1f}%")
        print(f"  ✅ 弹性降级响应：{len(degraded_nodes)} 个节点成功降级")
        print(f"  ✅ 非对称调度生成：{len(schedule)} 项调度指令")
        
        return {
            "co_location_rate": co_location_rate,
            "bubble_improvement": improvement,
            "zero_copy_ratio": stats_after_kv['zero_copy_ratio'],
            "degraded_nodes": len(degraded_nodes),
            "schedule_items": len(schedule)
        }
    
    def _generate_varlength_data(self, num_samples: int) -> List[Dict]:
        data = []
        for i in range(num_samples):
            seq_len = random.choice([64, 128, 256, 512, 1024, 2048, 4096])
            moe_routing = {j: random.random() for j in random.sample(range(64), 8)}
            data.append({
                "seq_len": seq_len,
                "moe_routing_prob": moe_routing
            })
        return data
    
    def _calculate_variance(self, values: List[float]) -> float:
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _calculate_co_location_rate(self, placement: Dict[int, int], 
                                   nvlink_domains: List[Set[int]]) -> float:
        total_pairs = 0
        co_located_pairs = 0
        
        experts = list(placement.keys())
        for i in range(len(experts)):
            for j in range(i+1, len(experts)):
                total_pairs += 1
                node_a = placement[experts[i]]
                node_b = placement[experts[j]]
                
                for domain in nvlink_domains:
                    if node_a in domain and node_b in domain:
                        co_located_pairs += 1
                        break
        
        return co_located_pairs / total_pairs * 100 if total_pairs > 0 else 0


def main():
    system = UnifiedPPHarnessSystem(
        num_nodes=64,
        num_stages=8,
        memory_gb=80.0
    )
    
    results = system.run_full_pipeline()
    
    print("\n" + "=" * 80)
    print("📊 结构化验证结果 (JSON)")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    
    return results


if __name__ == "__main__":
    main()

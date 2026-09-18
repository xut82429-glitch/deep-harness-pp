"""
深水区一：拓扑管理器 - 联合编译核心
探测物理拓扑，生成代价矩阵，实现 MoE 专家放置优化
"""
import random
from typing import List, Dict, Tuple
from proto.models import TopologyMatrix

class TopologyManager:
    """拓扑管理器：感知物理网络，优化专家放置"""
    
    def __init__(self, num_nodes: int = 64, gpus_per_node: int = 8):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.total_gpus = num_nodes * gpus_per_node
        self.nvlink_domains = self._build_nvlink_domains()
        
    def _build_nvlink_domains(self) -> List[List[int]]:
        """构建 NVLink 域：同一节点内 GPU 高速互联"""
        domains = []
        for node in range(self.num_nodes):
            domain = [node * self.gpus_per_node + i 
                     for i in range(self.gpus_per_node)]
            domains.append(domain)
        return domains
    
    def probe_topology(self) -> TopologyMatrix:
        """
        探测物理拓扑（模拟）
        返回带宽矩阵和延迟矩阵
        """
        nodes = list(range(self.total_gpus))
        bandwidth_matrix = [[0.0] * self.total_gpus for _ in range(self.total_gpus)]
        latency_matrix = [[0.0] * self.total_gpus for _ in range(self.total_gpus)]
        
        for i in range(self.total_gpus):
            for j in range(self.total_gpus):
                if i == j:
                    bandwidth_matrix[i][j] = float('inf')
                    latency_matrix[i][j] = 0
                elif self._same_nvlink_domain(i, j):
                    # NVLink: 900 GB/s, 500ns
                    bandwidth_matrix[i][j] = 900.0
                    latency_matrix[i][j] = 500.0
                elif self._same_node(i, j):
                    # PCIe: 64 GB/s, 2000ns
                    bandwidth_matrix[i][j] = 64.0
                    latency_matrix[i][j] = 2000.0
                else:
                    # RoCE: 200 GB/s, 5000ns
                    bandwidth_matrix[i][j] = 200.0
                    latency_matrix[i][j] = 5000.0
        
        return TopologyMatrix(
            nodes=nodes,
            bandwidth_matrix=bandwidth_matrix,
            latency_matrix=latency_matrix,
            nvlink_domains=self.nvlink_domains
        )
    
    def _same_nvlink_domain(self, gpu_a: int, gpu_b: int) -> bool:
        """检查两个 GPU 是否在同一 NVLink 域"""
        for domain in self.nvlink_domains:
            if gpu_a in domain and gpu_b in domain:
                return True
        return False
    
    def _same_node(self, gpu_a: int, gpu_b: int) -> bool:
        """检查两个 GPU 是否在同一节点"""
        return gpu_a // self.gpus_per_node == gpu_b // self.gpus_per_node
    
    def place_experts(self, 
                     num_experts: int, 
                     routing_matrix: Dict[Tuple[int, int], float],
                     topo: TopologyMatrix) -> Dict[int, int]:
        """
        深水区一核心：MoE 专家放置算法
        将频繁交互的专家对放置在同一 NVLink 域内
        """
        # 1. 提取高频交互专家对
        hot_pairs = sorted(
            routing_matrix.items(),
            key=lambda x: x[1],
            reverse=True
        )[:min(100, len(routing_matrix))]
        
        placement = {}
        assigned_domains = set()
        
        # 2. 贪心策略：优先安置高频交互对
        for (expert_a, expert_b), interaction_freq in hot_pairs:
            if expert_a in placement and expert_b in placement:
                continue
            
            # 寻找最优 NVLink 域
            best_domain = self._find_best_domain(
                expert_a, expert_b, 
                assigned_domains, 
                topo
            )
            
            if best_domain is not None:
                domain_gpus = topo.nvlink_domains[best_domain]
                free_gpu = self._find_free_gpu(
                    domain_gpus, 
                    placement, 
                    assigned_domains
                )
                if free_gpu is not None:
                    placement[expert_a] = free_gpu
                    placement[expert_b] = free_gpu
                    assigned_domains.add(best_domain)
        
        # 3. 剩余专家负载均衡填充
        all_gpus = list(range(self.total_gpus))
        for expert_id in range(num_experts):
            if expert_id not in placement:
                for gpu in all_gpus:
                    if gpu not in placement.values():
                        placement[expert_id] = gpu
                        break
        
        return placement
    
    def _find_best_domain(self, 
                         expert_a: int, 
                         expert_b: int,
                         assigned: set, 
                         topo: TopologyMatrix) -> int:
        """寻找最优 NVLink 域"""
        for idx, domain in enumerate(topo.nvlink_domains):
            if idx not in assigned:
                return idx
        return 0
    
    def _find_free_gpu(self, 
                      domain: List[int], 
                      placement: Dict[int, int],
                      assigned: set) -> int:
        """在域内寻找空闲 GPU"""
        used_gpus = set(placement.values())
        for gpu in domain:
            if gpu not in used_gpus:
                return gpu
        return domain[0]  # 降级：复用第一个
    
    def compute_cost_matrix(self, topo: TopologyMatrix) -> List[List[float]]:
        """计算通信代价矩阵：Cost = 1/Bandwidth + Latency"""
        cost_matrix = []
        for i in range(self.total_gpus):
            row = []
            for j in range(self.total_gpus):
                bw = topo.bandwidth_matrix[i][j]
                lat = topo.latency_matrix[i][j]
                if bw == float('inf'):
                    cost = 0.0
                else:
                    cost = 1.0 / bw + lat / 1e9  # 归一化
                row.append(cost)
            cost_matrix.append(row)
        return cost_matrix


def demo_topology_manager():
    """演示拓扑管理器功能"""
    print("=" * 60)
    print("深水区一：拓扑感知联合编译演示")
    print("=" * 60)
    
    mgr = TopologyManager(num_nodes=8, gpus_per_node=8)
    topo = mgr.probe_topology()
    
    print(f"集群规模：{mgr.num_nodes} 节点 × {mgr.gpus_per_node} GPU")
    print(f"NVLink 域数量：{len(topo.nvlink_domains)}")
    print(f"每个域包含：{len(topo.nvlink_domains[0])} GPU")
    
    # 模拟 MoE 路由矩阵
    random.seed(42)
    routing_matrix = {}
    for i in range(64):
        for j in range(i+1, 64):
            routing_matrix[(i, j)] = random.random()
    
    # 执行专家放置
    placement = mgr.place_experts(
        num_experts=64,
        routing_matrix=routing_matrix,
        topo=topo
    )
    
    # 统计同域率
    same_domain_count = 0
    total_pairs = 0
    for (a, b), freq in routing_matrix.items():
        if freq > 0.8:  # 高频交互对
            total_pairs += 1
            gpu_a, gpu_b = placement.get(a, -1), placement.get(b, -1)
            if mgr._same_nvlink_domain(gpu_a, gpu_b):
                same_domain_count += 1
    
    same_domain_rate = same_domain_count / max(total_pairs, 1) * 100
    print(f"\n高频交互专家对同域率：{same_domain_rate:.1f}%")
    print(f"✅ 拓扑感知专家放置完成")
    
    return mgr, topo, placement


if __name__ == "__main__":
    demo_topology_manager()

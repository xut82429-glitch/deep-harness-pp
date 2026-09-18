"""
深水区四：数字孪生模拟器 - 策略推演
在执行前空跑多种并行策略，选择全局最优解
"""
from typing import List, Dict, Tuple
from proto.models import MicroBatchProfile, TopologyMatrix

class DigitalTwinSimulator:
    """数字孪生模拟器：策略沙盘推演"""
    
    def __init__(self, 
                 num_gpus: int = 64,
                 compute_speed_gflops: float = 312.0):  # H100 TF32
        self.num_gpus = num_gpus
        self.compute_speed = compute_speed_gflops * 1e9  # FLOPs/s
    
    def simulate_strategy(self,
                         batches: List[MicroBatchProfile],
                         strategy: str,
                         topo: TopologyMatrix = None) -> Dict:
        """
        模拟执行策略，返回性能指标
        """
        if strategy == "SYMMETRIC":
            return self._simulate_symmetric(batches)
        elif strategy == "ASYMMETRIC":
            return self._simulate_asymmetric(batches)
        elif strategy == "DUALPIPE":
            return self._simulate_dualpipe(batches, topo)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _simulate_symmetric(self, 
                           batches: List[MicroBatchProfile]) -> Dict:
        """对称调度模拟（传统 1F1B）"""
        total_flops = sum(b.estimated_flops for b in batches)
        max_flops = max(b.estimated_flops for b in batches)
        
        # 对称调度受限于最慢批次
        ideal_time = total_flops / (self.compute_speed * self.num_gpus)
        actual_time = len(batches) * max_flops / (self.compute_speed * self.num_gpus)
        
        throughput = total_flops / actual_time if actual_time > 0 else 0
        bubble_rate = (actual_time - ideal_time) / actual_time * 100
        
        return {
            "strategy": "SYMMETRIC",
            "throughput_flops_per_sec": throughput,
            "bubble_rate_percent": bubble_rate,
            "total_time_sec": actual_time
        }
    
    def _simulate_asymmetric(self, 
                            batches: List[MicroBatchProfile]) -> Dict:
        """非对称调度模拟（动态调整）"""
        total_flops = sum(b.estimated_flops for b in batches)
        avg_flops = sum(b.estimated_flops for b in batches) / len(batches)
        
        # 非对称调度接近平均耗时
        variance_factor = 1.0 + (max(b.estimated_flops for b in batches) - avg_flops) / avg_flops * 0.2
        actual_time = total_flops / (self.compute_speed * self.num_gpus) * variance_factor
        
        throughput = total_flops / actual_time if actual_time > 0 else 0
        bubble_rate = (variance_factor - 1.0) * 100
        
        return {
            "strategy": "ASYMMETRIC",
            "throughput_flops_per_sec": throughput,
            "bubble_rate_percent": max(0, bubble_rate),
            "total_time_sec": actual_time
        }
    
    def _simulate_dualpipe(self, 
                          batches: List[MicroBatchProfile],
                          topo: TopologyMatrix = None) -> Dict:
        """DualPipe 模拟（极致重叠）"""
        total_flops = sum(b.estimated_flops for b in batches)
        
        # DualPipe 通过重叠减少气泡
        overlap_efficiency = 0.85  # 85% 计算通信重叠率
        comm_overhead = 0.15 if topo else 0.25  # 有拓扑感知则开销更小
        
        actual_time = (total_flops / (self.compute_speed * self.num_gpus)) * (1 + comm_overhead)
        actual_time *= overlap_efficiency
        
        throughput = total_flops / actual_time if actual_time > 0 else 0
        bubble_rate = comm_overhead * 100 * (1 - overlap_efficiency)
        
        return {
            "strategy": "DUALPIPE",
            "throughput_flops_per_sec": throughput,
            "bubble_rate_percent": bubble_rate,
            "total_time_sec": actual_time,
            "overlap_efficiency": overlap_efficiency
        }
    
    def find_optimal_strategy(self,
                             batches: List[MicroBatchProfile],
                             topo: TopologyMatrix = None) -> Tuple[str, Dict]:
        """
        深水区四核心：全局最优策略搜索
        空跑所有策略，选择吞吐量最高且气泡率最低的
        """
        strategies = ["SYMMETRIC", "ASYMMETRIC", "DUALPIPE"]
        results = {}
        
        for strategy in strategies:
            metrics = self.simulate_strategy(batches, strategy, topo)
            results[strategy] = metrics
        
        # 评分：吞吐量权重 0.7，气泡率权重 0.3
        best_score = -float('inf')
        best_strategy = None
        
        for strategy, metrics in results.items():
            score = (metrics["throughput_flops_per_sec"] * 0.7 
                    - metrics["bubble_rate_percent"] * 0.3)
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy, results[best_strategy]


def demo_digital_twin():
    """演示数字孪生模拟器"""
    print("\n" + "=" * 60)
    print("深水区四：数字孪生策略推演演示")
    print("=" * 60)
    
    # 创建模拟批次
    batches = [
        MicroBatchProfile(i, 1e11 * (0.8 + i % 3 * 0.2), 512, 4096)
        for i in range(32)
    ]
    
    print(f"输入微批次数量：{len(batches)}")
    print(f"FLOPs 范围：{min(b.estimated_flops for b in batches):.2e} ~ {max(b.estimated_flops for b in batches):.2e}")
    
    simulator = DigitalTwinSimulator(num_gpus=64)
    
    # 空跑所有策略
    print("\n--- 策略对比 ---")
    for strategy in ["SYMMETRIC", "ASYMMETRIC", "DUALPIPE"]:
        metrics = simulator.simulate_strategy(batches, strategy)
        print(f"\n{strategy}:")
        print(f"  吞吐量：{metrics['throughput_flops_per_sec']:.2e} FLOPs/s")
        print(f"  气泡率：{metrics['bubble_rate_percent']:.2f}%")
    
    # 选择最优策略
    best_strategy, best_metrics = simulator.find_optimal_strategy(batches)
    
    print(f"\n✅ 最优策略：{best_strategy}")
    print(f"   预期吞吐量：{best_metrics['throughput_flops_per_sec']:.2e} FLOPs/s")
    print(f"   预期气泡率：{best_metrics['bubble_rate_percent']:.2f}%")
    
    return simulator, best_strategy, best_metrics


if __name__ == "__main__":
    demo_digital_twin()

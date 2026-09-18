"""
深水区三：非对称调度器 - DualPipe 动态调度
根据 Micro-batch FLOPs 生成非对称调度表
"""
from typing import List, Dict
from proto.models import MicroBatchProfile

class AsymmetricScheduler:
    """非对称调度器：动态 DualPipe 调度"""
    
    def __init__(self, num_stages: int = 8, num_micro_batches: int = 16):
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
    
    def generate_schedule(self, 
                         batches: List[MicroBatchProfile]) -> List[Dict]:
        """
        生成非对称调度表
        每个条目包含：stage_id, mb_id, operation, estimated_time
        """
        schedule = []
        
        # 计算每个批次的相对耗时
        max_flops = max(b.estimated_flops for b in batches) if batches else 1
        relative_times = [b.estimated_flops / max_flops for b in batches]
        
        # 1F1B 调度：Warmup + Steady State + Cooldown
        warmup_batches = min(self.num_stages, len(batches))
        
        # Warmup 阶段：逐步填充流水线
        for stage in range(self.num_stages):
            for mb_idx in range(min(stage + 1, warmup_batches)):
                if mb_idx < len(batches):
                    schedule.append({
                        "stage_id": stage,
                        "mb_id": mb_idx,
                        "operation": "FWD",
                        "time_weight": relative_times[mb_idx]
                    })
        
        # Steady State：1F1B 交替
        steady_start = warmup_batches
        steady_end = len(batches)
        
        for mb_idx in range(steady_start, steady_end):
            for stage in range(self.num_stages):
                # Forward
                schedule.append({
                    "stage_id": stage,
                    "mb_id": mb_idx,
                    "operation": "FWD",
                    "time_weight": relative_times[mb_idx]
                })
                
                # Backward（滞后 num_stages 个批次）
                bwd_mb_idx = mb_idx - self.num_stages
                if bwd_mb_idx >= 0:
                    schedule.append({
                        "stage_id": stage,
                        "mb_id": bwd_mb_idx,
                        "operation": "BWD",
                        "time_weight": relative_times[bwd_mb_idx]
                    })
        
        # Cooldown 阶段：完成剩余反向传播
        for stage in reversed(range(self.num_stages)):
            for mb_idx in range(max(0, len(batches) - self.num_stages), len(batches)):
                schedule.append({
                    "stage_id": stage,
                    "mb_id": mb_idx,
                    "operation": "BWD",
                    "time_weight": relative_times[mb_idx]
                })
        
        return schedule
    
    def compute_pipeline_efficiency(self, 
                                   schedule: List[Dict],
                                   batches: List[MicroBatchProfile]) -> float:
        """计算流水线效率（1 - 气泡率）"""
        if not schedule or not batches:
            return 100.0
        
        # 按 stage 分组计算时间
        stage_times = {}
        for entry in schedule:
            stage = entry["stage_id"]
            time_w = entry["time_weight"]
            stage_times[stage] = stage_times.get(stage, 0) + time_w
        
        # 理想时间：所有 stage 负载均衡
        total_time = sum(stage_times.values())
        max_stage_time = max(stage_times.values()) if stage_times else 0
        ideal_time = total_time / self.num_stages
        
        # 效率 = 理想时间 / 实际时间（由最慢 stage 决定）
        efficiency = ideal_time / max_stage_time * 100 if max_stage_time > 0 else 100
        
        return efficiency
    
    def optimize_for_hardware(self, 
                             schedule: List[Dict],
                             stage_latencies: Dict[int, float]) -> List[Dict]:
        """
        根据硬件实际延迟优化调度表
        为慢速 stage 分配更多时间窗口
        """
        optimized = []
        
        avg_latency = sum(stage_latencies.values()) / len(stage_latencies) if stage_latencies else 1.0
        
        for entry in schedule:
            stage = entry["stage_id"]
            actual_latency = stage_latencies.get(stage, avg_latency)
            
            # 调整时间权重
            latency_factor = actual_latency / avg_latency
            optimized_entry = entry.copy()
            optimized_entry["adjusted_time"] = entry["time_weight"] * latency_factor
            
            optimized.append(optimized_entry)
        
        return optimized


def demo_asymmetric_scheduler():
    """演示非对称调度器"""
    print("\n" + "=" * 60)
    print("深水区三：非对称 DualPipe 调度演示")
    print("=" * 60)
    
    import random
    random.seed(42)
    
    # 创建变长微批次
    batches = [
        MicroBatchProfile(
            mb_id=i,
            estimated_flops=1e11 * (0.7 + random.random() * 0.6),
            seq_length=random.randint(256, 2048),
            token_count=random.randint(1000, 8000)
        )
        for i in range(32)
    ]
    
    print(f"微批次数量：{len(batches)}")
    print(f"FLOPs 范围：{min(b.estimated_flops for b in batches):.2e} ~ {max(b.estimated_flops for b in batches):.2e}")
    
    scheduler = AsymmetricScheduler(num_stages=8, num_micro_batches=32)
    
    # 生成调度表
    schedule = scheduler.generate_schedule(batches)
    print(f"\n调度表条目数：{len(schedule)}")
    
    # 统计操作分布
    fwd_count = sum(1 for e in schedule if e["operation"] == "FWD")
    bwd_count = sum(1 for e in schedule if e["operation"] == "BWD")
    print(f"Forward 操作：{fwd_count}")
    print(f"Backward 操作：{bwd_count}")
    
    # 计算效率
    efficiency = scheduler.compute_pipeline_efficiency(schedule, batches)
    bubble_rate = 100 - efficiency
    
    print(f"\n流水线效率：{efficiency:.2f}%")
    print(f"气泡率：{bubble_rate:.2f}%")
    
    # 模拟硬件感知优化
    stage_latencies = {i: 1.0 + i * 0.05 for i in range(8)}  # Stage 0 最快，Stage 7 最慢
    optimized_schedule = scheduler.optimize_for_hardware(schedule, stage_latencies)
    
    print(f"\n✅ 硬件感知优化完成")
    print(f"   Stage 延迟差异：{min(stage_latencies.values()):.2f}x ~ {max(stage_latencies.values()):.2f}x")
    
    return scheduler, schedule, optimized_schedule


if __name__ == "__main__":
    demo_asymmetric_scheduler()

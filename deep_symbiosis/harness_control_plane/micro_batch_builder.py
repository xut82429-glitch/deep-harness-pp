"""
深水区三：微批次构建器 - 变长数据整形
基于 FLOPs 的贪心装箱算法，最小化流水线气泡
"""
from typing import List, Dict
from proto.models import MicroBatchProfile

class Sample:
    """输入样本"""
    def __init__(self, sample_id: int, seq_len: int, moe_prob: float = 0.5):
        self.sample_id = sample_id
        self.seq_len = seq_len
        self.moe_prob = moe_prob

class MicroBatchBuilder:
    """微批次构建器：变长数据整形"""
    
    def __init__(self, 
                 stage_capacity_flops: float = 1e12,
                 max_variance: float = 0.1):
        self.stage_capacity = stage_capacity_flops
        self.max_variance = max_variance
    
    def estimate_flops(self, sample: Sample) -> float:
        """估算样本 FLOPs（简化模型）"""
        # FLOPs ≈ 2 * batch * seq_len^2 * hidden_dim
        # 这里简化为 seq_len^2 * moe_factor
        base_flops = sample.seq_len ** 2 * 1000
        moe_factor = 1.0 + sample.moe_prob * 0.5
        return base_flops * moe_factor
    
    def build_micro_batches(self, 
                           samples: List[Sample]) -> List[MicroBatchProfile]:
        """
        深水区三核心：贪心装箱算法
        按 FLOPs 切分，而非样本数
        """
        batches = []
        current_batch = []
        current_flops = 0.0
        mb_id = 0
        
        # 按序列长度排序（预处理优化）
        sorted_samples = sorted(samples, key=lambda s: s.seq_len, reverse=True)
        
        for sample in sorted_samples:
            sample_flops = self.estimate_flops(sample)
            
            # 检查是否超出容量容忍度
            if (current_flops + sample_flops <= 
                self.stage_capacity * (1 + self.max_variance)):
                current_batch.append(sample)
                current_flops += sample_flops
            else:
                # 打包当前批次
                if current_batch:
                    mb = self._pack_batch(current_batch, current_flops, mb_id)
                    batches.append(mb)
                    mb_id += 1
                
                # 开始新批次
                current_batch = [sample]
                current_flops = sample_flops
        
        # 处理剩余批次
        if current_batch:
            mb = self._pack_batch(current_batch, current_flops, mb_id)
            batches.append(mb)
        
        return batches
    
    def _pack_batch(self, 
                   samples: List[Sample], 
                   total_flops: float, 
                   mb_id: int) -> MicroBatchProfile:
        """打包微批次元数据"""
        total_tokens = sum(s.seq_len for s in samples)
        avg_seq_len = total_tokens / len(samples) if samples else 0
        
        return MicroBatchProfile(
            mb_id=mb_id,
            estimated_flops=total_flops,
            seq_length=int(avg_seq_len),
            token_count=total_tokens
        )
    
    def compute_bubble_rate(self, 
                           batches: List[MicroBatchProfile],
                           num_stages: int = 8) -> float:
        """
        计算流水线气泡率
        Bubble = (Max_Time - Avg_Time) / Max_Time
        """
        if not batches:
            return 0.0
        
        flops_list = [b.estimated_flops for b in batches]
        max_flops = max(flops_list)
        avg_flops = sum(flops_list) / len(flops_list)
        
        # 理想情况下所有批次耗时相同
        bubble = (max_flops - avg_flops) / max_flops if max_flops > 0 else 0
        return bubble * 100  # 百分比


def demo_micro_batch_builder():
    """演示微批次构建器"""
    print("\n" + "=" * 60)
    print("深水区三：变长数据整形与非对称调度演示")
    print("=" * 60)
    
    import random
    random.seed(42)
    
    # 生成变长样本（模拟评估场景）
    samples = []
    for i in range(200):
        # 长度分布：64 ~ 4096 tokens
        seq_len = random.choice([64, 128, 256, 512, 1024, 2048, 4096])
        moe_prob = random.random()
        samples.append(Sample(i, seq_len, moe_prob))
    
    print(f"输入样本数：{len(samples)}")
    print(f"样本长度范围：{min(s.seq_len for s in samples)} ~ {max(s.seq_len for s in samples)} tokens")
    
    builder = MicroBatchBuilder(
        stage_capacity_flops=1e12,
        max_variance=0.1
    )
    
    # 构建微批次
    batches = builder.build_micro_batches(samples)
    
    print(f"\n输出微批次数量：{len(batches)}")
    print(f"平均每批次样本数：{len(samples)/len(batches):.1f}")
    
    # 计算气泡率
    bubble_rate = builder.compute_bubble_rate(batches)
    print(f"流水线气泡率：{bubble_rate:.2f}%")
    
    # 对比：随机切分的气泡率
    random_batches = []
    batch_size = 16
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        flops = sum(builder.estimate_flops(s) for s in batch_samples)
        random_batches.append(MicroBatchProfile(
            mb_id=len(random_batches),
            estimated_flops=flops,
            seq_length=int(sum(s.seq_len for s in batch_samples)/len(batch_samples)),
            token_count=sum(s.seq_len for s in batch_samples)
        ))
    
    random_bubble = builder.compute_bubble_rate(random_batches)
    print(f"\n对比：随机切分气泡率：{random_bubble:.2f}%")
    print(f"气泡率降低：{(random_bubble - bubble_rate)/random_bubble*100:.1f}%")
    
    print(f"\n✅ 变长数据整形完成")
    
    return builder, batches


if __name__ == "__main__":
    demo_micro_batch_builder()

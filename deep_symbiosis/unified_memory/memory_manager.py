"""
深水区二：统一内存管理器 - 零拷贝借贷机制
统一管理 Activation 和 KV Cache，实现显存页复用
"""
from typing import Dict, List, Optional
from proto.models import MemoryRequest, MemoryResponse

class MemoryBlock:
    """显存页块"""
    def __init__(self, block_id: int, size_bytes: int):
        self.block_id = block_id
        self.size_bytes = size_bytes
        self.state = "FREE"  # FREE/ACTIVE_ACTIVATION/ACTIVE_KVCACHE/EVICTABLE
        self.owner = None  # PP_ENGINE/HARNESS
    
    def __repr__(self):
        return f"Block({self.block_id}, {self.state}, {self.owner})"

class UnifiedMemoryManager:
    """统一内存管理器：零拷贝借贷"""
    
    def __init__(self, 
                 total_memory_gb: float = 80.0,
                 block_size_kb: int = 16):
        self.total_memory_bytes = int(total_memory_gb * 1024**3)
        self.block_size_bytes = block_size_kb * 1024
        self.num_blocks = self.total_memory_bytes // self.block_size_bytes
        
        # 初始化显存池
        self.blocks: Dict[int, MemoryBlock] = {}
        for i in range(self.num_blocks):
            self.blocks[i] = MemoryBlock(i, self.block_size_bytes)
        
        # 空闲列表
        self.free_list = list(range(self.num_blocks))
        
        # 统计
        self.allocations = 0
        self.zero_copy_reuses = 0
    
    def allocate(self, request: MemoryRequest) -> MemoryResponse:
        """
        分配显存页
        优先从 EVICTABLE 池复用（零拷贝）
        """
        num_blocks_needed = (request.size_bytes + self.block_size_bytes - 1) // self.block_size_bytes
        
        # 1. 优先从 EVICTABLE 池复用（零拷贝关键）
        reused_blocks = self._find_evictable_blocks(num_blocks_needed)
        
        if len(reused_blocks) >= num_blocks_needed:
            # 零拷贝复用成功
            blocks_to_use = reused_blocks[:num_blocks_needed]
            self._mark_active(blocks_to_use, request.mem_type, request.owner)
            self.zero_copy_reuses += len(blocks_to_use)
            
            return MemoryResponse(
                request_id=request.request_id,
                success=True,
                block_ids=blocks_to_use,
                reused_zero_copy=True
            )
        
        # 2. 从空闲列表分配
        available_free = len(self.free_list)
        if available_free >= num_blocks_needed:
            blocks_to_use = [self.free_list.pop() for _ in range(num_blocks_needed)]
            self._mark_active(blocks_to_use, request.mem_type, request.owner)
            self.allocations += 1
            
            return MemoryResponse(
                request_id=request.request_id,
                success=True,
                block_ids=blocks_to_use,
                reused_zero_copy=False
            )
        
        # 3. 显存不足，触发紧急回收
        if self._try_emergency_eviction(num_blocks_needed):
            return self.allocate(request)  # 重试
        
        return MemoryResponse(
            request_id=request.request_id,
            success=False,
            block_ids=[],
            reused_zero_copy=False
        )
    
    def mark_evictable(self, block_ids: List[int], owner: str):
        """标记块为可回收（不释放数据，仅更新状态）"""
        for block_id in block_ids:
            if block_id in self.blocks:
                block = self.blocks[block_id]
                if block.owner == owner:
                    block.state = "EVICTABLE"
                    # 注意：不移动到 free_list，等待复用
    
    def _find_evictable_blocks(self, count: int) -> List[int]:
        """查找 EVICTABLE 状态的块"""
        evictable = [
            bid for bid, block in self.blocks.items()
            if block.state == "EVICTABLE"
        ]
        return evictable[:count]
    
    def _mark_active(self, 
                    block_ids: List[int], 
                    mem_type: str, 
                    owner: str):
        """标记块为活跃状态"""
        state_map = {
            "ACTIVATION": "ACTIVE_ACTIVATION",
            "KV_CACHE": "ACTIVE_KVCACHE"
        }
        for bid in block_ids:
            block = self.blocks[bid]
            block.state = state_map.get(mem_type, "ACTIVE_ACTIVATION")
            block.owner = owner
    
    def _try_emergency_eviction(self, needed: int) -> bool:
        """紧急回收：模拟将 EVICTABLE 数据卸载到 CPU"""
        evictable_count = sum(
            1 for b in self.blocks.values() 
            if b.state == "EVICTABLE"
        )
        
        if evictable_count >= needed:
            # 实际场景：异步拷贝到 CPU 内存
            for bid, block in self.blocks.items():
                if block.state == "EVICTABLE" and len(self.free_list) < needed:
                    block.state = "FREE"
                    block.owner = None
                    self.free_list.append(bid)
            return True
        
        return False
    
    def get_stats(self) -> Dict:
        """获取内存统计信息"""
        stats = {"FREE": 0, "ACTIVE_ACTIVATION": 0, "ACTIVE_KVCACHE": 0, "EVICTABLE": 0}
        for block in self.blocks.values():
            stats[block.state] += 1
        
        total_used = stats["ACTIVE_ACTIVATION"] + stats["ACTIVE_KVCACHE"]
        utilization = total_used / self.num_blocks * 100
        
        return {
            **stats,
            "total_blocks": self.num_blocks,
            "utilization_percent": utilization,
            "allocations": self.allocations,
            "zero_copy_reuses": self.zero_copy_reuses,
            "zero_copy_rate": self.zero_copy_reuses / max(self.allocations, 1) * 100
        }


def demo_unified_memory():
    """演示统一内存管理器"""
    print("\n" + "=" * 60)
    print("深水区二：统一内存管理与零拷贝借贷演示")
    print("=" * 60)
    
    # 模拟 80GB 显存，16KB 页
    manager = UnifiedMemoryManager(total_memory_gb=80.0, block_size_kb=16)
    
    print(f"总显存：80 GB")
    print(f"页大小：16 KB")
    print(f"总页数：{manager.num_blocks}")
    
    # 场景 1: PP Engine 申请 Activation 内存
    req1 = MemoryRequest(
        request_id="PP_001",
        size_bytes=1024 * 1024 * 1024,  # 1GB
        mem_type="ACTIVATION",
        owner="PP_ENGINE"
    )
    resp1 = manager.allocate(req1)
    print(f"\n[PP Engine] 申请 1GB Activation:")
    print(f"  成功：{resp1.success}, 分配页数：{len(resp1.block_ids)}")
    
    # 标记为可回收（Forward 完成）
    manager.mark_evictable(resp1.block_ids, "PP_ENGINE")
    print(f"  标记为 EVICTABLE（等待反向传播或复用）")
    
    # 场景 2: Harness 申请 KV Cache 内存
    req2 = MemoryRequest(
        request_id="HARNESS_001",
        size_bytes=1024 * 1024 * 1024,  # 1GB
        mem_type="KV_CACHE",
        owner="HARNESS"
    )
    resp2 = manager.allocate(req2)
    print(f"\n[Harness] 申请 1GB KV Cache:")
    print(f"  成功：{resp2.success}, 分配页数：{len(resp2.block_ids)}")
    print(f"  零拷贝复用：{resp2.reused_zero_copy}")
    
    if resp2.reused_zero_copy:
        print(f"  ✅ 零拷贝复用成功：直接复用 PP Engine 释放的页")
    
    # 获取统计
    stats = manager.get_stats()
    print(f"\n--- 内存统计 ---")
    print(f"FREE: {stats['FREE']} 页")
    print(f"ACTIVE_ACTIVATION: {stats['ACTIVE_ACTIVATION']} 页")
    print(f"ACTIVE_KVCACHE: {stats['ACTIVE_KVCACHE']} 页")
    print(f"EVICTABLE: {stats['EVICTABLE']} 页")
    print(f"显存利用率：{stats['utilization_percent']:.1f}%")
    print(f"零拷贝复用率：{stats['zero_copy_rate']:.1f}%")
    
    print(f"\n✅ 统一内存管理演示完成")
    
    return manager, stats


if __name__ == "__main__":
    demo_unified_memory()

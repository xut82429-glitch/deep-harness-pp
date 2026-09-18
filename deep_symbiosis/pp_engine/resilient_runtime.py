"""
深水区五：弹性运行时 - FSM 状态机与混沌工程
实现 NORMAL -> DEGRADED -> RECOVERING -> FAILED 状态流转
"""
import time
import random
from typing import List, Dict, Optional
from proto.models import HardwareAnomaly

class ResilientRuntime:
    """弹性运行时：基于 FSM 的故障管理"""
    
    def __init__(self, 
                 degraded_threshold: float = 1.5,
                 hysteresis_window: float = 5.0):
        self.state = "NORMAL"
        self.degraded_threshold = degraded_threshold
        self.hysteresis_window = hysteresis_window
        
        # 状态跟踪
        self.anomaly_history: List[HardwareAnomaly] = []
        self.state_enter_time = time.time()
        self.performance_cap = 1.0  # 100%
        
        # 降级动作记录
        self.degradation_actions = []
    
    def report_anomaly(self, anomaly: HardwareAnomaly) -> str:
        """
        报告硬件微异常
        根据严重程度和持续时间决定状态流转
        """
        self.anomaly_history.append(anomaly)
        
        # 计算最近异常的加权严重度
        recent_anomalies = [
            a for a in self.anomaly_history
            if time.time() - a.timestamp < self.hysteresis_window
        ]
        
        if not recent_anomalies:
            return self.state
        
        avg_severity = sum(a.severity for a in recent_anomalies) / len(recent_anomalies)
        
        # FSM 状态流转
        old_state = self.state
        self.state = self._transition_state(old_state, avg_severity, len(recent_anomalies))
        
        if self.state != old_state:
            self.state_enter_time = time.time()
            print(f"\n[FSM] 状态流转：{old_state} -> {self.state}")
            
            if self.state == "DEGRADED":
                self._execute_degradation(recent_anomalies)
            elif self.state == "RECOVERING":
                self._execute_recovery()
        
        return self.state
    
    def _transition_state(self, 
                         current: str, 
                         severity: float, 
                         count: int) -> str:
        """状态流转逻辑"""
        if current == "NORMAL":
            if severity > 0.7 or count >= 3:
                return "DEGRADED"
            elif severity > 0.4:
                return "DEGRADED"
        
        elif current == "DEGRADED":
            time_in_state = time.time() - self.state_enter_time
            
            # 持续恶化 -> FAILED
            if severity > 0.9 and time_in_state > self.hysteresis_window:
                return "FAILED"
            
            # 恢复正常 -> RECOVERING（需要迟滞窗口）
            if severity < 0.3 and time_in_state >= self.hysteresis_window:
                return "RECOVERING"
        
        elif current == "RECOVERING":
            time_in_state = time.time() - self.state_enter_time
            
            # 恢复完成 -> NORMAL
            if time_in_state >= self.hysteresis_window:
                return "NORMAL"
            
            # 再次恶化 -> DEGRADED
            if severity > 0.5:
                return "DEGRADED"
        
        elif current == "FAILED":
            # 需要人工干预或自动恢复流程
            pass
        
        return current
    
    def _execute_degradation(self, anomalies: List[HardwareAnomaly]):
        """执行降级动作"""
        self.performance_cap = 0.5  # 降低 50% 性能
        
        # 具体降级策略
        actions = []
        
        # 1. 精度降级：BF16 -> FP8
        actions.append("PRECISION: BF16 -> FP8")
        
        # 2. 减少 Micro-batch 分配
        actions.append("SCHEDULER: Reduce MB allocation by 50%")
        
        # 3. 通信环重构（绕过亚健康节点）
        affected_nodes = set(a.node_id for a in anomalies)
        actions.append(f"NETWORK: Reroute around nodes {affected_nodes}")
        
        self.degradation_actions = actions
        
        for action in actions:
            print(f"  [降级动作] {action}")
    
    def _execute_recovery(self):
        """执行恢复动作"""
        print("  [恢复动作] Restoring full performance...")
        print("  [恢复动作] Re-enabling BF16 precision")
        print("  [恢复动作] Rebalancing Micro-batch distribution")
        
        self.performance_cap = 1.0
        self.degradation_actions = []
    
    def get_health_report(self) -> Dict:
        """生成健康报告"""
        return {
            "current_state": self.state,
            "performance_cap": self.performance_cap,
            "anomaly_count": len(self.anomaly_history),
            "recent_anomalies": len([
                a for a in self.anomaly_history
                if time.time() - a.timestamp < self.hysteresis_window
            ]),
            "degradation_actions": self.degradation_actions
        }


def demo_resilient_runtime():
    """演示弹性运行时与混沌工程"""
    print("\n" + "=" * 60)
    print("深水区五：弹性运行时与混沌工程演示")
    print("=" * 60)
    
    runtime = ResilientRuntime(
        degraded_threshold=1.5,
        hysteresis_window=5.0
    )
    
    print(f"初始状态：{runtime.state}")
    print(f"迟滞窗口：{runtime.hysteresis_window}s")
    
    # 混沌工程：注入故障
    print("\n--- 混沌工程测试 ---")
    
    # 场景 1: 轻微延迟抖动（不应触发降级）
    print("\n[测试 1] 注入轻微延迟抖动...")
    anomaly1 = HardwareAnomaly(
        node_id=3, gpu_id=0,
        anomaly_type="LATENCY_SPIKE",
        severity=0.3
    )
    state1 = runtime.report_anomaly(anomaly1)
    print(f"当前状态：{state1} (预期：NORMAL)")
    
    # 场景 2: 多次中等异常（触发 DEGRADED）
    print("\n[测试 2] 注入连续中等异常...")
    for i in range(3):
        anomaly = HardwareAnomaly(
            node_id=5, gpu_id=i,
            anomaly_type="COMPUTE_SLOWDOWN",
            severity=0.6
        )
        state = runtime.report_anomaly(anomaly)
    
    print(f"当前状态：{runtime.state} (预期：DEGRADED)")
    print(f"性能限制：{runtime.performance_cap * 100:.0f}%")
    
    # 场景 3: 模拟恢复过程
    print("\n[测试 3] 模拟系统恢复...")
    # 等待迟滞窗口（模拟）
    runtime.state_enter_time = time.time() - runtime.hysteresis_window - 1
    
    anomaly_recovery = HardwareAnomaly(
        node_id=5, gpu_id=0,
        anomaly_type="COMPUTE_SLOWDOWN",
        severity=0.1  # 轻微
    )
    state_recovery = runtime.report_anomaly(anomaly_recovery)
    print(f"当前状态：{runtime.state} (预期：RECOVERING 或 NORMAL)")
    
    # 最终报告
    report = runtime.get_health_report()
    print(f"\n--- 健康报告 ---")
    print(f"当前状态：{report['current_state']}")
    print(f"性能限制：{report['performance_cap'] * 100:.0f}%")
    print(f"异常总数：{report['anomaly_count']}")
    print(f"最近异常：{report['recent_anomalies']}")
    
    if report['degradation_actions']:
        print(f"降级动作:")
        for action in report['degradation_actions']:
            print(f"  - {action}")
    
    print(f"\n✅ 弹性运行时演示完成")
    
    return runtime, report


if __name__ == "__main__":
    demo_resilient_runtime()

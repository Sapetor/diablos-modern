"""
Performance Profiling Utilities for DiaBloS

Usage:
    from lib.profiling import SimulationProfiler
    
    profiler = SimulationProfiler()
    profiler.start()
    # ... run simulation ...
    profiler.stop()
    profiler.report()
"""

import time
import logging
from collections import defaultdict
from functools import wraps

logger = logging.getLogger(__name__)


class SimulationProfiler:
    """
    Lightweight profiler for DiaBloS simulations.
    Tracks execution time of blocks and overall simulation metrics.
    """
    
    def __init__(self):
        self.block_times = defaultdict(float)
        self.block_calls = defaultdict(int)
        self.start_time = None
        self.end_time = None
        self.total_iterations = 0
        self.enabled = False
    
    def start(self):
        """Start profiling session."""
        self.block_times.clear()
        self.block_calls.clear()
        self.start_time = time.perf_counter()
        self.end_time = None
        self.total_iterations = 0
        self.enabled = True
        logger.info("Profiler started")
    
    def stop(self):
        """Stop profiling session."""
        self.end_time = time.perf_counter()
        self.enabled = False
        logger.info("Profiler stopped")
    
    def record_block(self, block_name: str, elapsed: float):
        """Record execution time for a block."""
        if self.enabled:
            self.block_times[block_name] += elapsed
            self.block_calls[block_name] += 1
    
    def increment_iteration(self):
        """Increment iteration counter."""
        if self.enabled:
            self.total_iterations += 1
    
    def report(self) -> str:
        """
        Generate and return profiling report.
        Also logs the report.
        """
        if self.start_time is None:
            return "No profiling data available"
        
        total_time = (self.end_time or time.perf_counter()) - self.start_time
        
        lines = []
        lines.append("=" * 60)
        lines.append("SIMULATION PROFILING REPORT")
        lines.append("=" * 60)
        lines.append(f"Total simulation time: {total_time:.3f} seconds")
        lines.append(f"Total iterations: {self.total_iterations}")
        if self.total_iterations > 0:
            lines.append(f"Average time per iteration: {total_time/self.total_iterations*1000:.3f} ms")
        lines.append("")
        
        # Sort blocks by total time (descending)
        sorted_blocks = sorted(
            self.block_times.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if sorted_blocks:
            lines.append("Block Execution Times (sorted by total time):")
            lines.append("-" * 60)
            lines.append(f"{'Block':<30} {'Calls':>8} {'Total(ms)':>12} {'Avg(ms)':>10}")
            lines.append("-" * 60)
            
            for block_name, total_ms in sorted_blocks[:20]:  # Top 20
                calls = self.block_calls[block_name]
                avg_ms = (total_ms / calls * 1000) if calls > 0 else 0
                total_ms_display = total_ms * 1000
                lines.append(f"{block_name:<30} {calls:>8} {total_ms_display:>12.3f} {avg_ms:>10.3f}")
        
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        logger.info("\n" + report)
        print(report)
        
        return report


# Global profiler instance
_profiler = None


def get_profiler() -> SimulationProfiler:
    """Get or create the global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = SimulationProfiler()
    return _profiler


def profile_block(func):
    """Decorator to profile block execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = get_profiler()
        if profiler.enabled:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            # Try to get block name from kwargs or args
            block_name = kwargs.get('block_name', 'unknown')
            profiler.record_block(block_name, elapsed)
            return result
        return func(*args, **kwargs)
    return wrapper

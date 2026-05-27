"""
FlexLLM: A flexible LLM inference scheduler with hybrid CPU-GPU KV Cache management.
Supports dynamic sequence scheduling, preemption, swap, and recomputation optimization
to maximize GPU memory utilization and inference throughput.
"""

__version__ = "0.1.0"

from .util import FlexLLMConfig
from .predictor import Predictor
from .swapper import Swapper
from .block_manager import FlexLLMBlockSpaceManager
from .scheduler import FlexScheduler
from .simulator import Simulator, Seq

__all__ = [
    "FlexLLMConfig",
    "Predictor",
    "Swapper",
    "FlexLLMBlockSpaceManager",
    "FlexScheduler",
    "Simulator",
    "Seq"
]

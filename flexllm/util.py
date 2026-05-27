# Global root directory for project data and model files
base_path = "/workspace/data"


class FlexLLMConfig:
    
    """
    Core configuration class for the FlexLLM project.
    Contains key parameters for KV Cache scheduling and GPU memory management.
    
    Args:
        model_name: Name or path of the target LLM model.
        onload_start: Trigger threshold, swap in user requests when KV Cache GPU memory ratio is below this value.
        onload_end: End threshold, stop user request swap-in when KV Cache GPU memory ratio reaches this value.
        backup_rate: Ratio of KV Cache data to offload when GPU memory is full.
        block_size: Block size for splitting and loading model weights, activation and kv cache.
    """

    def __init__(self, 
        model_name: str, 
        onload_start: float, 
        onload_end: float, 
        backup_rate: float, 
        block_size: int
    ) -> None: 
        self.model_name = model_name
        self.onload_start = onload_start
        self.onload_end = onload_end
        self.backup_rate = backup_rate
        self.block_size = block_size

    def _verify_args(self) -> None:
        assert self.model_name in ["llama-1-13b", "llama-2-13b", "llama-3-8b", "opt-13b"], (
            f"Invalid model: {self.model_name} ! \n"
            "Only llama-1-13b, llama-2-13b, llama-3-8b, opt-13b is supported by flexllm ! ")
        assert 0 < self.onload_start < self.onload_end < 1, (
            f"Invalid arguments: onload_start = {self.onload_start}, onload_end = {self.onload_end} \n"
            "The condition 0 < onload_start < onload_end < 1 must be satisfied ! ")
        assert 0 < self.backup_rate < 1, (
            f"Invalid argument: backup_rate = {self.backup_rate}"
            "The condition 0 < backup_rate < 1 must be satisfied ! ")
        assert self.block_size in [4, 8, 16, 32, 64, 128], (
            f"Invalid argument: block_size = {self.block_size}"
            "block size must be choosed from 4, 8, 16, 32, 64, 128 ! ")

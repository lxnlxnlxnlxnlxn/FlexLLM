from typing import Dict, Optional

from vllm.core.block_manager import BlockSpaceManager, BlockTable
from vllm.sequence import Sequence, SequenceGroup
from vllm.utils import Device


class FlexLLMBlockSpaceManager(BlockSpaceManager):
    
    """
    An extended block space manager designed specifically for FlexLLM, 
    built on top of vLLM's native BlockSpaceManager.
    It supports hybrid CPU-GPU KV Cache block management, including 
    incremental block backup, full sequence swap-out, and partial swap-in 
    with recomputation optimization.
    
    This class maintains two block tables for each sequence:
    1. GPU block table (inherited from parent) for active KV Cache.
    2. CPU block table (flexllm_block_tables) for backed-up KV Cache.
    
    It enables flexible memory scheduling: offloading blocks to CPU when 
    GPU memory is full, and restoring blocks when memory is underutilized.

    Args:
        block_size: Number of tokens that each memory block can hold.
        num_gpu_blocks: Total number of physical KV Cache blocks on GPU.
        num_cpu_blocks: Total number of physical KV Cache blocks on CPU.
        watermark: Memory watermark ratio for safe memory management.
        sliding_window: Sliding window size for attention (optional).
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None
    ) -> None:
        
        # Initialize parent block manager with basic CPU-GPU resources
        super().__init__(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            watermark=watermark,
            sliding_window=sliding_window)
        
        # Inherited: Maps sequence ID to its GPU-resident block table
        # self.block_tables: Dict[int, BlockTable] = {}
        
        # Custom FlexLLM: Maps sequence ID to its CPU-backed block table
        # All CPU blocks are sequentially copied from corresponding GPU blocks
        self.flexllm_block_tables: Dict[int, BlockTable] = {}
    
    def backup(self, seq_group: SequenceGroup) -> Dict[int, int]:
        """
        Perform incremental backup of KV Cache blocks:
        Only backs up the LAST block of the sequence from GPU to CPU.
        All earlier blocks are assumed to have already been backed up.
        
        If the CPU block table does not exist, it will be initialized.
        If the last GPU block is not yet backed up, a new CPU block is allocated.

        Args:
            seq_group: The sequence group containing the sequence to back up.
        
        Returns:
            A dictionary mapping the backed-up GPU block number 
            to the allocated CPU block number.
        """

        seq = seq_group.get_seqs()[0]
        gpu_block_table = self.block_tables[seq.seq_id]
        
        # Initialize empty CPU block table if not created yet
        if seq.seq_id not in self.flexllm_block_tables.keys():
            self.flexllm_block_tables[seq.seq_id] = list()
        cpu_block_table = self.flexllm_block_tables[seq.seq_id]
        
        # Get the last block of GPU KV Cache (the only one needing backup)
        gpu_block = gpu_block_table[-1]
        
        # Allocate new CPU block if the backup is incomplete
        if len(gpu_block_table) > len(cpu_block_table):
            cpu_block = self.cpu_allocator.allocate()
            cpu_block_table.append(cpu_block)
        
        # Reuse the last CPU block if already backed up
        else:
            cpu_block = cpu_block_table[-1]
        
        # Return GPU -> CPU block mapping for data transfer
        return {gpu_block.block_number: cpu_block.block_number}

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        """
        Fully swap out a sequence from GPU to CPU memory:
        1. First ensure the last block is safely backed up to CPU.
        2. Free all GPU blocks occupied by this sequence.
        3. Clear the GPU block table to release resources.
        
        This operation completely evicts the sequence from GPU.

        Args:
            seq_group: The sequence group to be evicted from GPU.
        
        Returns:
            Mapping of all GPU block numbers to their CPU backup blocks.
        """

        seq = seq_group.get_seqs()[0]
        
        # Ensure CPU backup table exists before swap-out
        assert seq.seq_id in self.flexllm_block_tables.keys()
        
        # Final backup to ensure the latest block is safe
        backup_map = self.backup(seq_group=seq_group)
        
        # Free all GPU blocks and empty the GPU block table
        for gpu_block in self.block_tables[seq.seq_id]:
            self.gpu_allocator.free(gpu_block)
        self.block_tables[seq.seq_id] = []
        
        return backup_map
    
    def swap_in(self, seq_group: SequenceGroup, sep: int) -> Dict[int, int]:
        """
        Partially swap a sequence from CPU back to GPU.
        Blocks BEFORE the separator index are regenerated by recomputation.
        Blocks AFTER the separator index are restored from CPU backup.
        
        This is the core optimization of FlexLLM: reduce swap-in overhead
        by combining recomputation and partial block loading.

        Args:
            seq_group: Sequence group to be restored to GPU.
            sep: Separator block index (split point between recompute and restore).
        
        Returns:
            Mapping from CPU backup blocks to GPU target blocks.
        """
        
        seq = seq_group.get_seqs()[0]
        
        # Ensure CPU backup exists before swap-in
        assert seq.seq_id in self.flexllm_block_tables.keys()
        mapping = {}

        # Get CPU-backed blocks and restore only blocks after separator
        cpu_block_table = self.flexllm_block_tables[seq.seq_id]
        for idx, cpu_block in enumerate(cpu_block_table[sep:]):
            gpu_block = self.block_tables[seq.seq_id][idx + sep]
            mapping[cpu_block.block_number] = gpu_block.block_number
            
        return mapping
    
    def free(self, seq: Sequence) -> None:
        """
        Completely release all resources for a finished sequence:
        1. Free all GPU blocks and delete the GPU block table.
        2. Free all CPU backup blocks and delete the CPU backup table.
        
        Ensures no memory leakage after sequence completion.

        Args:
            seq: The finished sequence to be fully freed.
        """

        # Free all GPU blocks and remove the GPU block table
        if seq.seq_id in self.block_tables.keys():
            for block in self.block_tables[seq.seq_id]:
                if block.device == Device.GPU:
                    self.gpu_allocator.free(block)
                else:
                    self.cpu_allocator.free(block)
            del self.block_tables[seq.seq_id]

        # Free all CPU backup blocks and remove the FlexLLM CPU table
        if seq.seq_id in self.flexllm_block_tables.keys():
            for block in self.flexllm_block_tables[seq.seq_id]:
                if block.device == Device.GPU:
                    self.gpu_allocator.free(block)
                else:
                    self.cpu_allocator.free(block)
            del self.flexllm_block_tables[seq.seq_id]
            
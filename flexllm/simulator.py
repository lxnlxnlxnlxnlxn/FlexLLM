import math
from typing import List

from util import FlexLLMConfig


class Seq:

    """
    Sequence class representing a single request in FlexLLM.
    Tracks prompt length, output length, and block usage for scheduling.

    Args:
        prompt_len: Length of the input prompt
    """

    def __init__(self, prompt_len: int) -> None:
        self.prompt_len = prompt_len
        self.output_len = 0
        self.block = 0
    
    def prefill(self, block_size: int) -> int:
        """
        Perform prefill (context encoding) for the sequence.
        Calculate and update block usage, then generate one token.
        
        Args:
            block_size: Size of each memory block
        
        Returns:
            Total blocks occupied after prefill
        """

        self.block = math.ceil((self.prompt_len + self.output_len + 1) / block_size)
        self.output_len += 1
        return self.block
    
    def decode(self, block_size: int) -> int:
        """
        Perform decode (token generation) step.
        Calculate new blocks and return the number of additional blocks needed.
        
        Args:
            block_size: Size of each memory block
        
        Returns:
            Number of new blocks added in this decode step
        """

        block = math.ceil((self.prompt_len + self.output_len + 1) / block_size)
        add_block = block - self.block
        self.block = block
        self.output_len += 1
        return add_block
    
    def get_total_block(self, block_size: int) -> int:
        """
        Calculate total blocks required for the sequence.
        
        Args:
            block_size: Size of each memory block
        
        Returns:
            Total number of blocks needed
        """

        return (self.prompt_len + self.output_len) / block_size


class Simulator:
    
    """
    FlexLLM scheduling simulator for GPU block management. Simulates request 
    scheduling, preemption, swap, and regeneration based on KV Cache memory 
    usage. Implements dynamic scheduling using onload_start, onload_end, 
    and backup_rate thresholds.

    Args:
        model_name: Name of the LLM model.
        flexllm_config: FlexLLM configuration (thresholds for scheduling).
        block_num: Total number of available GPU blocks.
        output_len: Target output length for each sequence.
    """

    def __init__(self,
                model_name: str, 
                flexllm_config: FlexLLMConfig,
                block_num: float,
                output_len: int
    ) -> None:    
        self.model_name = model_name
        self.flexllm_config = flexllm_config
        self.block_num = block_num
        self.output_len = output_len
        
        # Sequence queues
        self.running: List[Seq] = []        # Sequences running on GPU
        self.waiting: List[Seq] = []        # Waiting sequences
        self.swapped: List[Seq] = []        # Swapped/preempted sequences
        
        self.used_block = 0
        self.iter = 0

    def add_seq(self, prompt_len_list: List[int]) -> None:
        """
        Add new request sequences to the waiting queue.
        
        Args:
            prompt_len_list: List of prompt lengths for new sequences.
        """
        for prompt_len in prompt_len_list:
            seq = Seq(prompt_len=prompt_len)
            self.waiting.append(seq)
 
    def schedule(self) -> int:
        """
        Main scheduling loop: run until all sequences finish generation.
        
        Returns:
            Total iterations (time steps) used to complete all requests.
        """
        while self.running or self.swapped or self.waiting:
            self._schedule()
            self.iter += 1
        return self.iter

    def _schedule(self) -> float:
        """
        Internal core scheduling logic:
        1. Regenerate sequences if memory is low.
        2. Preempt if memory is insufficient.
        3. Run decode steps for running sequences.
        """
        
        # Regenerate swapped/waiting sequences if used blocks below onload_start
        if self.used_block < self.block_num * self.flexllm_config.onload_start:
            self._regen()
        else:
            # Preempt if running sequences exceed available blocks
            if len(self.running) > self.block_num - self.used_block:
                self._preempt()
            # Decode step: release blocks after generation
            for seq in self.running:
                self.used_block -= seq.decode()
        
        # Filter finished sequences (output reaches target length)
        running: List[Seq] = list()
        for seq in self.running:
            if seq.output_len < self.output_len:
                running.append(seq)
        self.running = running         

    def _regen(self) -> float:
        """
        Regenerate (swap in) sequences from swapped or waiting queues.
        Load sequences until GPU blocks reach onload_end threshold.
        """

        goal_block = math.floor(self.block_num * self.flexllm_config.onload_end)
        
        # Recover swapped sequences first
        while self.swapped:
            seq = self.swapped[0]
            block_num = seq.get_total_block(block_size=self.block_size)
            if block_num + self.used_block > goal_block:
                break
            
            self.swapped.pop(0)
            self.running.append(seq)
            self.used_block += block_num
        
        # Add new waiting sequences
        while self.waiting:
            seq = self.waiting[0]
            block_num = seq.get_total_block(block_size=self.block_size)
            if block_num + self.used_block > goal_block:
                break
            
            self.waiting.pop(0)
            self.running.append(seq)
            self.used_block += block_num
    
    def _preempt(self) -> None: 
        """
        Preempt (swap out) running sequences to free GPU blocks.
        Stop when used blocks drop below backup_rate threshold.
        """
             
        preempt_goal = self.block_num * self.flexllm_config.backup_rate

        while self.running:
            seq = self.running[0]
            block_num = seq.get_total_block(block_size=self.block_size)

            self.used_block -= block_num   
            self.swapped.append(seq)
            self.running.remove(seq)

            if self.used_block - block_num <= preempt_goal:
                break

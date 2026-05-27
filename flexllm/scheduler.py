import math
from typing import Dict, List

from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.config import CacheConfig, SchedulerConfig, ModelConfig
from vllm.sequence import SequenceGroup, SequenceStatus

from flexllm.util import FlexLLMConfig
from flexllm.predictor import Predictor
from flexllm.swapper import Swapper
from flexllm.block_manager import FlexLLMBlockSpaceManager


class FlexScheduler(Scheduler):
    
    """
    Custom scheduling module inherited from native vLLM scheduler.
    Implements hybrid CPU-GPU KV cache management strategy for FlexLLM 
    inference. Core capabilities include real-time time prediction, 
    PCIe data swap latency calculation, sequence backup, memory 
    preemption and recomputation-driven sequence swap-in. Dynamically 
    adjust running sequences according to GPU memory occupancy threshold, 
    balance computation overhead and data transmission overhead to improve 
    overall throughput.
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        model_config: ModelConfig,
    ) -> None:
        """
        Initialize FlexLLM enhanced scheduler instance.
        Inherit basic scheduling capability from vLLM base scheduler,
        instantiate auxiliary functional modules and custom block manager.

        Args:
            scheduler_config: Global scheduling constraint configuration of vLLM
            cache_config: KV cache block partition and capacity configuration
            model_config: Basic structural parameter configuration of large 
                language model
        """

        super().__init__(scheduler_config, cache_config)
        
        # Persist core configuration parameters
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config

        # Extract pure model name from full model storage path
        self.model_name = self.model_config.model.split('/')[-1]

        # Inference latency prediction module for prefill and decode stage
        self.predictor = Predictor(model_name=self.model_name)

        # PCIe bandwidth query and swap time calculation module
        self.swapper = Swapper()

        # Cache pending sequence group waiting for KV cache backup operation
        self.backup: List[SequenceGroup] = list()

        # Custom block manager supporting cross-device block backup, swap and restore
        self.block_manager = FlexLLMBlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window)

    def set_config(self, flexllm_config: FlexLLMConfig) -> None:
        """
        Load FlexLLM exclusive scheduling policy parameters,
        and initialize swap calculation module with model actual parameters.

        Args:
            flexllm_config: Custom configuration containing memory threshold 
                and block size
        """
        self.flexllm_config = flexllm_config

        # Pass model hidden dimension, layer count and data type to swapper for memory calculation
        self.swapper.set(hidden_size=self.model_config.hf_config.hidden_size,
                         num_layers=self.model_config.hf_config.num_hidden_layers,
                         block_size=self.flexllm_config.block_size,
                         dtype=2)
        
    def _schedule(self) -> SchedulerOutputs:
        """
        Main core scheduling execution function of FlexLLM.
        Execute multi-stage scheduling judgment and processing in fixed order:
        1. Complete scheduled KV cache backup for partial running sequences
        2. Trigger sequence swap-in recomputation when GPU memory usage is too low
        3. Trigger running sequence preemption when GPU memory resources are insufficient
        4. Allocate new inference slot for surviving running sequences
        Finally encapsulate scheduling result and return to inference engine.

        Returns:
            SchedulerOutputs: Structured scheduling result carrying sequence list,
                              block swap mapping and batch token statistical information
        """

        # Store block mapping relationship needing to offload from GPU to CPU
        blocks_to_swap_out: Dict[int, int] = {}
        
        # Store block copy relationship inside GPU device
        blocks_to_copy: Dict[int, List[int]] = {}

        # Count current total and available GPU physical blocks
        total_block = self.block_manager.num_total_gpu_blocks
        free_block = self.block_manager.gpu_allocator.get_num_free_blocks()
        cur_block = total_block - free_block
        
        # Execute pre-scheduled KV cache backup task
        self._backup_schedule(blocks_to_swap_out=blocks_to_swap_out)

        # Low memory utilization scenario: activate sequence regenerate swap-in strategy
        if cur_block <= total_block * self.flexllm_config.onload_start:
            scheduler_outputs = self._regenerate_schedule()
            if scheduler_outputs:
                self._backup_search()
                return scheduler_outputs
        
        # Insufficient memory scenario: activate running sequence preemption strategy
        if len(self.running) > self.block_manager.get_num_free_gpu_blocks():
            self._preempt_schedule(blocks_to_swap_out=blocks_to_swap_out)
            self._backup_search()
        
        # Guarantee valid running sequences exist for subsequent inference
        assert len(self.running) > 0
        
        # Apply new cache slot expansion for each ongoing inference sequence
        for seq_group in self.running:
            self._append_slot(seq_group, blocks_to_copy)
        
        # Accumulate total valid inference tokens of current batch
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)
        
        # Assemble complete scheduling output data packet
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=[],
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    def _backup_search(self) -> None:
        """
        Select eligible sequences from running queue to join backup waiting queue.
        Take the trade-off between model decode calculation time and CPU-GPU swap time as judgment basis.
        Continuously add sequences to backup list until swap overhead exceeds calculation overhead.
        Avoid excessive backup causing unnecessary data transmission delay.
        """

        # Count existing pending backup block quantity
        backup_now = len(self.backup)
        
        # Traverse all current running inference sequences
        for seq_group in self.running:
            
            # Skip sequences already added to backup queue
            if seq_group in self.backup:
                continue
            
            # Calculate block quantity occupied by current sequence
            seq_len = seq_group.get_seqs()[0].get_len()
            block_num = math.ceil(seq_len / self.flexllm_config.block_size)
            
            # Stop backup selection when calculation efficiency is higher than swap efficiency
            if self._get_decode_time(self.running) <= self.swapper.get_swap_time(
                    block_num=backup_now + block_num, type="dtoh") and backup_now != 0:
                break
            
            # Add qualified sequence to backup waiting list
            else:
                backup_now += block_num
                self.backup.append(seq_group)

    def _backup_schedule(self, blocks_to_swap_out: Dict[int, int]) -> None:
        """
        Formal execution of KV cache backup operation.
        Call custom block manager to copy latest cache block of target sequence 
        to CPU side, record corresponding GPU and CPU block number mapping for 
        subsequent swap-out operation.

        Args:
            blocks_to_swap_out: Dictionary used to collect block mapping relation 
                of backup behavior
        """

        # Process each sequence waiting for cache backup
        for seq_group in self.backup:
            mapping = self.block_manager.backup(seq_group)
            blocks_to_swap_out.update(mapping)

    def _preempt_schedule(self, blocks_to_swap_out: Dict[int, int]) -> None:       
        """
        Memory preemption processing logic.
        When GPU cache is saturated, actively evict backup-ready sequences from 
        GPU to CPU. Gradually release GPU blocks until memory usage drops to 
        preset backup safety threshold. Update sequence running state and 
        maintain internal queue order.

        Args:
            blocks_to_swap_out: Collect block mapping generated by preemption swap-out
        """

        # Acquire total available GPU block resources
        total = self.block_manager.num_total_gpu_blocks
        
        # Statistic block quantity already evicted to CPU
        backup_now = 0
        
        # Target memory occupancy ratio after preemption stabilization
        backup_goal = total * self.flexllm_config.backup_rate

        # Continuously evict sequences from backup queue
        while self.backup:
            seq_group = self.backup[0]
            seq = seq_group.get_seqs()[0]

            # Calculate actual block consumption of single sequence
            blk = math.ceil(seq.get_len() / self.flexllm_config.block_size)
            
            # Stop preemption action after reaching safety memory threshold
            if backup_now + blk > backup_goal and backup_now != 0:
                break
            backup_now += blk
            
            # Mark sequence as swapped-out state
            seq.status = SequenceStatus.SWAPPED
            
            # Complete cross-device block migration and obtain mapping relation
            mapping = self.block_manager.swap_out(seq_group)
            blocks_to_swap_out.update(mapping)
            
            # Adjust sequence queue distribution
            self.backup.pop(0)
            self.swapped.append(seq_group)
            self.running.remove(seq_group)          
    
    def _regenerate_schedule(self) -> SchedulerOutputs:
        """
        Sequence regenerate swap-in scheduling logic.
        Adopt combination strategy of partial recomputation and cache restore.
        Divide into three sequential processing phases:
        1. Screen recoverable sequences from swapped queue and waiting queue
        2. Solve optimal recomputation length to minimize maximum time cost
        3. Execute partial cache swap-in and inference slot allocation
        Return regenerated batch scheduling result to start prompt stage calculation.

        Returns:
            SchedulerOutputs: Scheduling packet of regenerated sequence batch,
                              return None if no eligible recoverable sequence
        """

        # First stage: filter valid sequences supporting regenerate recovery
        regen_list: List[SequenceGroup] = []
        ignored_list: List[SequenceGroup] = []
        self._regen_search(regen_list=regen_list, ignored_list=ignored_list)
        
        # Terminate regenerate flow without available sequence
        if len(regen_list) == 0:
            return None
        
        # Second stage: optimize recomputation segmentation length
        rcmp_l = self._regen_refine(regen_list=regen_list)
        rcmp_block = math.ceil(rcmp_l / self.flexllm_config.block_size)
        
        # Third stage: perform cache restore and slot allocation
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        for seq_group in regen_list:
            
            # Restore subsequent cache block data from CPU if sequence length exceeds split point
            if seq_group.get_seqs()[0].get_len() > rcmp_l:
                mapping = self.block_manager.swap_in(seq_group=seq_group, sep=rcmp_block)
                blocks_to_swap_in.update(mapping)
            
            # Allocate new inference cache slot for regenerated sequence
            self._append_slot(seq_group=seq_group, blocks_to_copy=blocks_to_copy)       

        # Assemble regenerate batch scheduling result
        return SchedulerOutputs(
            scheduled_seq_groups=regen_list,
            prompt_run=True,
            num_batched_tokens=rcmp_l * len(regen_list),
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_copy,
            blocks_to_copy=[],
            ignored_seq_groups=ignored_list)

    def _regen_search(
        self, 
        regen_list: List[SequenceGroup], 
        ignored_list: List[SequenceGroup]
    ) -> None:
        """
        The first step of regenerate strategy: candidate sequence screening.
        Follow FCFS priority principle, swapped-out sequences have higher 
        recovery priority. Check prompt length limit, block allocation 
        feasibility and batch capacity constraint. Stop adding new sequences 
        when GPU memory reaches onload upper threshold. Exceed-length illegal 
        sequences will be marked as ignored status.

        Args:
            regen_list: Temporary list storing sequences to be recovered
            ignored_list: Collect sequences exceeding maximum prompt length restriction
        """

        # Count current real-time GPU block usage
        total_block = self.block_manager.num_total_gpu_blocks
        free_block = self.block_manager.gpu_allocator.get_num_free_blocks()
        used_block = total_block - free_block
        
        # Upper limit of block quantity after completing sequence swap-in
        goal_block = math.floor(total_block * self.flexllm_config.onload_end)
        
        # Preferentially recover sequences stored in CPU swap queue
        while self.swapped:
            seq_group = self.swapped[0]
            seq = seq_group.get_seqs()[0]
            
            # Eliminate super-long prompt sequences beyond model processing capability
            num_prompt_tokens = seq.get_len()
            if num_prompt_tokens > self.prompt_limit:
                seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_list.append(seq_group)
                self.swapped.pop(0)
                continue
            
            # Estimate block resource consumption of single sequence
            block_num = math.ceil(seq.get_len() / self.flexllm_config.block_size)
            
            # Stop recovery after touching memory upper limit
            if used_block + block_num >= goal_block:
                break 
            
            # Skip sequence failing block allocation inspection
            if not self.block_manager.can_allocate(seq_group):
                break
            
            # Insufficient remaining block cannot accommodate new sequence
            if len(self.running) + block_num + 1 >= self.block_manager.get_num_free_gpu_blocks():
                break
            
            # Reach maximum concurrent inference sequence limit
            if len(self.running) == self.scheduler_config.max_num_seqs:
                break

            # Formal allocate block and join running queue
            self.swapped.pop(0)
            self._allocate(seq_group=seq_group)
            self.running.append(seq_group)
            
            used_block += block_num
            regen_list.append(seq_group)
        
        # Supplement new pending sequences from waiting queue
        while self.waiting:
            seq_group = self.waiting[0]
            seq = seq_group.get_seqs()[0]

            # Reject oversize prompt request
            num_prompt_tokens = seq.get_len()
            if num_prompt_tokens > self.prompt_limit:
                seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_list.append(seq_group)
                self.waiting.pop(0)
                continue
            
            block_num = math.ceil(seq.get_len() / self.flexllm_config.block_size)
            if used_block + block_num >= goal_block:
                break 
            if not self.block_manager.can_allocate(seq_group):
                break
            if len(self.running) + block_num + 1 >= self.block_manager.get_num_free_gpu_blocks():
                break
            if len(self.running) == self.scheduler_config.max_num_seqs:
                break
            
            # Complete sequence access and resource binding
            self.waiting.pop(0)
            self._allocate(seq_group=seq_group)
            self.running.append(seq_group)
            
            used_block += block_num
            regen_list.append(seq_group)

    def _regen_refine(
        self, 
        regen_list: List[SequenceGroup], 
    ) -> float:
        """
        The second step of regenerate strategy: optimal recomputation length 
        solving. Optimization objective: minimize the maximum value between 
        recompute time and swap time. Constraint condition: unified recomputation 
        length for all sequences in batch, avoid redundant padding calculation 
        and guarantee computing efficiency. Iteratively shrink recomputation 
        interval until time cost reaches balance state.

        Args:
            regen_list: Sequence batch waiting for recomputation segmentation optimization

        Returns:
            float: Optimal token length of front segment recomputation part
        """

        # Initialize time cost parameters
        rcmp_time = self._get_prefill_time(seq_group_list=regen_list)
        swap_time = 0
        
        # Initial recompute length takes maximum sequence length of current batch
        rcmp_l = max([seq_group.get_seqs()[0].get_len() for seq_group in regen_list])
        rcmp_block = math.ceil(rcmp_l / self.flexllm_config.block_size)
        
        # Minimum legal recompute length restricted by original prompt length
        min_rcmp_l = max([seq_group.get_seqs()[0].get_prompt_len() 
                          for seq_group in regen_list])
        min_rcmp_block = math.ceil(min_rcmp_l / self.flexllm_config.block_size)
        
        # Iteratively adjust split length to balance two types of time overhead
        while rcmp_time > swap_time and rcmp_block > min_rcmp_block:        
            rcmp_block -= 1
            rcmp_l = rcmp_block * self.flexllm_config.block_size
            
            # Refresh recomputation time consumption after length adjustment
            rcmp_time = self._get_prefill_time(seq_group_list=regen_list, seql=rcmp_l)
            
            # Accumulate total block quantity needing cross-device swap
            block_num = 0
            for seq_group in regen_list:
                seql = seq_group.get_seqs()[0].get_len()
                swapl = seql - rcmp_l
                block_num += math.ceil(swapl / self.flexllm_config.block_size)               
            
            # Calculate corresponding PCIe swap transmission time
            swap_time = self.swapper.get_swap_time(block_num=block_num, type="htod")

        return rcmp_l

    def free_finished_seq_groups(self) -> None:
        """
        Regular cleanup maintenance function.
        Filter and remove fully completed inference sequences from running 
        queue and backup queue. Timely release invalid sequence occupation 
        to avoid waste of scheduling and memory resources.
        """

        self.running = [seq_group for seq_group in self.running
                        if not seq_group.is_finished()]
        self.backup = [seq_group for seq_group in self.backup
                        if not seq_group.is_finished()]
    
    def _get_prefill_time(
        self, 
        seq_group_list: List[SequenceGroup], 
        seql: int = None
    ) -> float:
        """
        Invoke offline prediction model to obtain prefill stage inference latency.
        Take batch size and unified sequence length as prediction input features.

        Args:
            seq_group_list: Target sequence batch participating in prefill calculation
            seql: Custom specified sequence length, use batch max length if not assigned

        Returns:
            float: Predicted time consumption of prefill forward computation
        """

        if seql is None:
            seql = max([seq_group.get_seqs()[0].get_len() 
                        for seq_group in seq_group_list])
        bs = len(seq_group_list)

        return self.predictor.get_time(bs=bs, seql=seql, stage="p")

    def _get_decode_time(self, seq_group_list: List[SequenceGroup]) -> float:
        """
        Predict time overhead of token autoregressive decode stage.
        Calculate average sequence length of batch as statistical feature for prediction.

        Args:
            seq_group_list: Current active running sequence batch

        Returns:
            float: Predicted single step decode inference delay
        """

        l_list = [seq_group.get_seqs()[0].get_len() 
                  for seq_group in seq_group_list]
        seql = math.ceil(sum(l_list) / len(l_list))
        bs = len(seq_group_list)

        return self.predictor.get_time(bs=bs, seql=seql, stage="d")
    
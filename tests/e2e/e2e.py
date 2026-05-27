import json
import argparse

from transformers import logging
# Disable unnecessary logging output from transformers
logging.set_verbosity_error()

from vllm import LLM, SamplingParams


def main(args: argparse.ArgumentParser):

    # --------------------------
    # 1. Parse all input arguments
    # --------------------------
    model_name = args.model_name           # Name of the LLM model
    onload_start = args.onload_start       # GPU memory threshold to start swap-in (regeneration)
    onload_end = args.onload_end           # GPU memory threshold to stop swap-in
    backup_rate = args.backup_rate         # Target GPU memory ratio after preemption
    block_size = args.block_size           # KV Cache block size (tokens per block)
    dataset = args.dataset                 # Test dataset name
    gpu_block = args.gpu_block             # Number of GPU KV Cache blocks
    cpu_block = args.cpu_block             # Number of CPU KV Cache blocks
    batch_size = args.batch_size           # Maximum concurrent batch size
    testcase_num = args.testcase_num       # Number of test cases to run
    output_len = args.output_len           # Tokens to generate per prompt

    # --------------------------
    # 2. Load and process dataset
    # --------------------------
    dataset_path = f"/workspace/data/dataset/{dataset}.json"
    prompt_list = []
    
    # Read JSON dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    # Extract prompts (only first N test cases)
    for data in data_list[0:testcase_num]:
        prompt = data["prompt"]
        prompt_list.append(prompt)
    
    # --------------------------
    # 3. Load vLLM with FlexLLM enabled
    # --------------------------
    model_path = f"/workspace/data/models/{model_name}"
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,            # No tensor parallelism
        gpu_memory_utilization=0.90,       # GPU memory usage limit
        use_flexllm=True,                  # Enable FlexLLM scheduling
        block_size=block_size,             # KV Cache block size
        max_num_seqs=batch_size,           # Max batch size
        num_gpu_blocks=gpu_block,          # GPU block count
        num_cpu_blocks=cpu_block)          # CPU block count

    # --------------------------
    # 4. Set FlexLLM scheduling config
    # --------------------------
    llm.set_flexllm_config(
        model_name=model_name,
        onload_start=onload_start,
        onload_end=onload_end,
        backup_rate=backup_rate,
        block_size=block_size)
    
    # --------------------------
    # 5. Run batch inference
    # --------------------------
    sampling_params = SamplingParams(
        temperature=0,        # Deterministic generation
        top_p=1,              # No nucleus sampling
        max_tokens=output_len) # Max generation length
    
    # Start generation
    _ = llm.generate(
        prompt_list,
        sampling_params,
        use_tqdm=True)   # Show progress bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FlexLLM Inference Runner with vLLM backend")

    # Model & scheduling parameters
    parser.add_argument(
        '--model_name', 
        type=str,
        choices=["opt-13b", "llama-1-160m", "llama-1-13b", "llama-2-13b", "llama-3-8b"],
        required=True, 
        help="Name of the model to run"
    )
    parser.add_argument(
        '--onload_start', 
        type=float,
        required=True, 
        help="GPU memory ratio to start swapping sequences in"
    )
    parser.add_argument(
        '--onload_end', 
        type=float,
        required=True, 
        help="GPU memory ratio to stop swapping sequences in"
    )
    parser.add_argument(
        '--backup_rate', 
        type=float,
        required=True, 
        help="Target GPU memory ratio after preemption/swap-out"
    )
    parser.add_argument(
        '--block_size', 
        type=int, 
        choices=[4, 8, 16, 32, 64],
        required=True, 
        help="KV Cache block size in tokens"
    )

    # Dataset & resource settings
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=["alpaca", "chatbot", "gsm8k"],
        required=True, 
        help="Dataset to run inference on"
    )
    parser.add_argument(
        '--gpu_block', 
        type=int,
        required=True, 
        help="Number of GPU KV Cache blocks"
    )
    parser.add_argument(
        '--cpu_block', 
        type=int,
        required=True, 
        help="Number of CPU KV Cache blocks"
    )
    parser.add_argument(
        '--batch_size', 
        type=int,
        required=True, 
        help="Maximum batch size for inference"
    )
    parser.add_argument(
        '--testcase_num', 
        type=int,
        required=True, 
        help="Number of test cases to run"
    )
    parser.add_argument(
        '--output_len', 
        type=int,
        required=True, 
        help="Number of tokens to generate per prompt"
    )

    args = parser.parse_args()
    main(args=args)

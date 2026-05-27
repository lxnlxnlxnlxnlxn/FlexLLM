import json
import argparse
from typing import List

from transformers import AutoTokenizer

from simulator import Simulator
from util import FlexLLMConfig


def main(args: argparse.ArgumentParser):
    
    # Parse command-line arguments
    model_name = args.model_name          # Name of the LLM model used in simulation
    dataset = args.dataset                # Name of the dataset to simulate
    gpu_block = args.gpu_block            # Total number of GPU KV cache blocks available
    output_len = args.output_len          # Target number of tokens to generate per sequence

    # FlexLLM scheduling policy parameters
    onload_start = args.onload_start      # GPU mem threshold to trigger swap-in (regeneration)
    onload_end = args.onload_end          # GPU mem upper limit when stopping swap-in
    backup_rate = args.backup_rate        # Target GPU usage after preemption/swap-out
    block_size = args.block_size          # KV cache block size (tokens per block)

    # Initialize tokenizer (model path will be filled in real usage)
    tokenizer = AutoTokenizer.from_pretrained("")

    # Load dataset JSON file
    dataset_path = f"../../data/dataset/{dataset}.json"
    prompt_list: List[int] = []

    # Read and tokenize all prompts to get their lengths (in tokens)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    for data in data_list:
        prompt = data["prompt"]
        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=512
        )
        prompt_list.append(len(token_ids))  # Store prompt length in tokens
    
    # Initialize FlexLLM scheduling configuration
    flexllm_config = FlexLLMConfig(
        model_name=model_name, 
        onload_start=onload_start, 
        onload_end=onload_end, 
        backup_rate=backup_rate, 
        block_size=block_size
    )

    # Create the FlexLLM scheduling simulator
    simulator = Simulator(
        model_name=model_name, 
        flexllm_config=flexllm_config, 
        block_num=gpu_block, 
        output_len=output_len
    )

    # Add all sequence prompt lengths to the simulator
    simulator.add_seq(prompt_list)

    # Run the full scheduling simulation until all sequences finish
    t = simulator.schedule()

    # Print all configuration and final simulation result (total time/score)
    print(f"model_name = {model_name} \n"
          f"dataset = {dataset} \n"
          f"gpu_block = {gpu_block} \n"
          f"output_len = {output_len} \n"
          f"onload_start = {onload_start} \n"
          f"onload_end = {onload_end} \n"
          f"backup_rate = {backup_rate} \n"
          f"block_size = {block_size} \n"
          f"total_score = {int(t)}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="FlexLLM Scheduling Simulator")

    parser.add_argument(
        '--model_name', 
        type=str,
        choices=["opt-13b", "llama-1-160m", "llama-1-13b", "llama-2-13b", "llama-3-8b"],
        required=True,
        help="Name of the LLM model to simulate"
    )
    parser.add_argument(
        '--dataset', 
        type=str,
        choices=["alpaca", "chatbot", "gsm8k"],
        required=True,
        help="Dataset name for simulation requests"
    )
    parser.add_argument(
        '--gpu_block', 
        type=int,
        required=True,
        help="Total number of GPU KV cache blocks"
    )
    parser.add_argument(
        '--output_len', 
        type=int,
        required=True,
        help="Number of tokens to generate per request"
    )
    parser.add_argument(
        '--onload_start', 
        type=float,
        required=True,
        help="GPU memory ratio to start swapping in sequences"
    )
    parser.add_argument(
        '--onload_end', 
        type=float,
        required=True,
        help="GPU memory ratio to stop swapping in sequences"
    )
    parser.add_argument(
        '--backup_rate', 
        type=float,
        required=True,
        help="Target GPU memory ratio after preemption"
    )
    parser.add_argument(
        '--block_size', 
        type=int, 
        choices=[4, 8, 16, 32, 64],
        required=True,
        help="KV cache block size in tokens"
    )

    args = parser.parse_args()
    main(args=args)
    
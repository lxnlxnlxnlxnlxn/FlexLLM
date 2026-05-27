import os
import json
import math
import csv 
import argparse
from typing import List

from flexllm.predictor import Predictor
from flexllm.swapper import Swapper


def main(args: argparse.ArgumentParser):

    # Parse input arguments
    model_name = args.model_name        # Name of the LLM model
    block_size = args.block_size        # KV Cache block size (tokens per block)
    prompt_len = args.prompt_len        # Prompt length of the sequence
    output_len = args.output_len        # Target output length to generate
    step = args.step                    # Step size for iterating recomputation lengths

    # Initialize predictor (for prefill/decode time prediction) and swapper (for PCIe swap time)
    predictor = Predictor(model_name=model_name)
    swapper = Swapper()
    
    # Load model config (hidden_size, num_layers) from model directory
    model_path = f"/workspace/data/models/{model_name}"
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    
    # Initialize swapper with model parameters for memory calculation
    swapper.set(hidden_size=hidden_size, 
                num_layers=num_layers, 
                block_size=block_size, 
                dtype=2)

    # Total sequence length = prompt + generated tokens
    total_len = prompt_len + output_len
    
    # Lists to store results
    recomp_len_list: List[int] = []      # Recomputation lengths
    recomp_t_list: List[int] = []        # Recomputation times (ms)
    swap_t_list: List[int] = []          # PCIe swap times (ms)

    # Iterate over different recomputation lengths from prompt_len to total_len
    for l in range(prompt_len, total_len, step):
        recomp_len_list.append(l)

        # Predict recomputation time (prefill time)
        recomp_t = predictor.get_time(bs=1, seql=l, stage='p')
        recomp_t_list.append(round(recomp_t, 3))
        
        # Calculate number of blocks to swap from CPU to GPU
        block_num = math.ceil(total_len / block_size) - math.floor(l / block_size) 
        
        # Get PCIe swap time
        swap_t = swapper.get_swap_time(block_num=block_num, type='htod')
        swap_t_list.append(round(swap_t, 3))
    
    # Write all results to output.csv
    with open("output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
    
        writer.writerow(["title"])
        writer.writerow(["recomp_len"] + recomp_len_list)
        writer.writerow(["recomp_t"] + recomp_t_list)
        writer.writerow(["swap_t"] + swap_t_list)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="FlexLLM Recomputation vs Swap Time Analyzer")
    
    parser.add_argument(
        '--model_name', 
        type=str, 
        choices=["opt-13b", "llama-1-160m", "llama-1-13b", "llama-2-13b", "llama-3-8b"],
        required=True, 
        help="Name of the model to evaluate"
    )
    parser.add_argument(
        '--block_size', 
        type=int, 
        choices=[4, 8, 16, 32, 64],
        required=True, 
        help="KV Cache block size in tokens"
    )
    parser.add_argument(
        '--prompt_len', 
        type=int, 
        choices=[64, 128, 256, 512],
        required=True, 
        help="Prompt length of the sequence"
    )
    parser.add_argument(
        '--output_len', 
        type=int, 
        choices=[64, 128, 256, 512],
        required=True, 
        help="Target output generation length"
    )
    parser.add_argument(
        '--step', 
        type=int, 
        choices=[1, 2, 4, 8, 16],
        required=True, 
        help="Step size for iterating recomputation length"
    )

    args = parser.parse_args()
    main(args=args)
    
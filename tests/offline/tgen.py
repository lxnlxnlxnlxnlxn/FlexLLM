import random
from typing import List, Dict
import csv
import argparse

import torch
import gc
from transformers import logging
logging.set_verbosity_error()  # Disable unnecessary transformer warnings

from vllm.vllm import LLM, SamplingParams
from vllm.vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


def main(args: argparse.ArgumentParser):
    
    # Parse input arguments
    model_name = args.model_name
    model_path = f"/workspace/data/models/{model_name}"
    
    # Batch size configuration
    bs_step = args.bs_step          # Step size for iterating batch sizes
    bs_max = args.bs_max            # Maximum batch size to test
    
    # Sequence length configuration
    seqlen_step = args.seqlen_step  # Step size for iterating sequence lengths
    seqlen_max = args.seqlen_max    # Maximum sequence length to test
    
    # Test parameters
    group = args.group              # Number of test groups for stable averaging
    outlen = args.outlen            # Number of tokens to generate per sequence
    vocab_size = args.vocab_size    # Model vocabulary size
    flu = args.flu                  # Fluctuation threshold for result validation

    # Store benchmark results
    pres = list()  # Prefill time results
    dres = list()  # Decode time results

    # Iterate all batch sizes and sequence lengths
    for bs in range(bs_step, bs_max + bs_step, bs_step):
        for seql in range(seqlen_step, seqlen_max + seqlen_step, seqlen_step):
            
            # Generate random token prompts for testing
            prompt_list = gen(bs, seql, group, vocab_size)
            res = None
            
            # Run test until valid (stable) results are obtained
            while res is None:
                res = run(model_path, prompt_list, bs, group, outlen, flu)
            
            print(f"bs={bs},seql={seql},ptime={res['ptime']},dtime={res['dtime']}")
            
            # Collect results
            pres.append({'bs': bs, 'seql': seql, 't(ms)': res['ptime']})
            dres.append({'bs': bs, 'seql': seql, 't(ms)': res['dtime']})

    # Write results to CSV files
    write(fname=f"/workspace/data/forward_time/{model_name}/ptime.csv", data=pres)
    write(fname=f"/workspace/data/forward_time/{model_name}/dtime.csv", data=pres)


def gen(bs: int, seqlen: int, group: int, vocab: int) -> List[List[int]]:
    prompt_list = list()
    for _ in range(group):
        for _ in range(bs):
            prompt = list()
            # Generate random token IDs as test prompt
            for _ in range(seqlen):
                token_id = random.randint(5, vocab)
                prompt.append(token_id)
            prompt_list.append(prompt)
    return prompt_list


def run(model_path: str, prompt_list: List, bs: int, group: int, outlen: int, flu: int):
    
    # Initialize vLLM engine
    llm = LLM(model=model_path, max_num_seqs=bs, gpu_memory_utilization=0.90)
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=outlen)
    
    # Run inference and get time consumption
    _, use_time_list = llm.generate(
        prompt_token_ids=prompt_list,
        sampling_params=sampling_params,
        use_tqdm=False
    )
    
    # Validate time list length
    assert len(use_time_list) == outlen * group
    
    # Clean up vLLM and GPU memory
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Calculate average prefill and decode time
    ptime_list = list()
    dtime_list = list()
    
    for i in range(1, group):
        ptime_list.append(use_time_list[i * outlen])
        for j in range(1, outlen):
            dtime_list.append(use_time_list[i * outlen + j])
    
    ptime_res = sum(ptime_list) / len(ptime_list)
    dtime_res = sum(dtime_list) / len(dtime_list)
    
    # Check result stability (filter outliers)
    for ptime in ptime_list:
        if abs(ptime - ptime_res) > ptime_res * flu:
            print("ptime fail")
            return None
    for dtime in dtime_list:
        if abs(dtime - dtime_res) > dtime_res * flu:
            print("dtime fail")
            return None
        
    # Convert to milliseconds and return
    return {
        'ptime': round(ptime_res * 1000, 3),
        'dtime': round(dtime_res * 1000, 3)
    }


def write(fname: str, data: List[Dict[str, float]]):
    with open(fname, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['bs', 'seql', 't(ms)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Inference Latency Benchmark Tool")

    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True, 
        help="Name of the model to benchmark"
    )
    parser.add_argument(
        '--bs_step', 
        type=int, 
        required=True, 
        help="Step size for batch size iteration"
    )
    parser.add_argument(
        '--bs_max', 
        type=int, 
        required=True, 
        help="Maximum batch size to test"
    )
    parser.add_argument(
        '--seqlen_step', 
        type=int, 
        required=True, 
        help="Step size for sequence length iteration"
    )
    parser.add_argument(
        '--seqlen_max', 
        type=int, 
        required=True, 
        help="Maximum sequence length to test"
    )
    parser.add_argument(
        '--group', 
        type=int, 
        required=True, 
        help="Number of test groups for averaging"
    )
    parser.add_argument(
        '--outlen', 
        type=int, 
        required=True, 
        help="Number of tokens to generate per request"
    )
    parser.add_argument(
        '--vocab_size', 
        type=int, 
        required=True, 
        help="Model vocabulary size"
    )
    parser.add_argument(
        '--flu', 
        type=float, 
        required=True, 
        help="Fluctuation threshold for result validation"
    )

    args = parser.parse_args()
    main(args=args)
    
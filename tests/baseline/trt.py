import time
import pandas as pd
import torch
import gc
import argparse
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig


def main(args: argparse.ArgumentParser):

    # ======================
    # 1. Parse command-line arguments
    # ======================
    testcase_num = args.testcase_num    # Number of test prompts to run
    model = args.model                  # Model name for benchmark
    dataset = args.dataset              # Test dataset name
    output_len = args.output_len        # Number of tokens to generate per prompt

    # ======================
    # 2. Initialize TensorRT-LLM engine
    # ======================
    model_path = "/workspace/data/models/" + model
    
    # Configure KV Cache: block size, GPU memory usage limit
    kv_cfg = KvCacheConfig(tokens_per_block=16, free_gpu_memory_fraction=0.07)
    
    # Initialize LLM inference engine
    llm = LLM(
        model=model_path,
        kv_cache_config=kv_cfg,
        gpus_per_node=1,
        max_batch_size=16,
        max_seq_len=1638400
    )

    # ======================
    # 3. Load dataset and extract prompts
    # ======================
    dataset_path = f"/workspace/data/dataset/{dataset}.json"
    prompts = []
    raw_data_list = pd.read_json(dataset_path).values

    # Collect valid prompts (non-empty) up to testcase_num
    for raw_data in raw_data_list:
        prompt_text = raw_data[0]
        if len(prompt_text) > 0:
            prompts.append(prompt_text)
        if len(prompts) == testcase_num:
            break

    # ======================
    # 4. Run inference and measure time
    # ======================
    # Generation parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=output_len
    )

    # Start timing
    start_time = time.time()
    
    # Run batch inference
    _, use_time_list = llm.generate(prompts, sampling_params)
    
    # End timing
    end_time = time.time()

    # ======================
    # 5. Calculate performance metrics
    # ======================
    throughput = testcase_num / (end_time - start_time)    # Throughput: requests/second
    avg_jct = sum(use_time_list) / len(use_time_list)      # Average job completion time

    # ======================
    # 6. Save results to log file
    # ======================
    with open("/workspace/output_1.txt", "a", encoding='utf-8') as f:
        f.write(
            f"model = {model}, dataset = {dataset}, "
            f"output_len = {output_len}, "
            f"total_time = {round(end_time - start_time, 3)}, "
            f"throughput = {round(throughput, 3)}, "
            f"avg_jct = {round(avg_jct, 3)}\n"
        )

    # ======================
    # 7. Clean up GPU resources
    # ======================
    llm.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM Inference Benchmark for LLM Performance Testing"
    )

    # Test configuration parameters
    parser.add_argument(
        '--testcase_num', 
        type=int, 
        required=True, 
        help="Number of test prompts to run for inference"
    )
    parser.add_argument(
        '--model', 
        type=str,
        choices=["llama-3-8b"], 
        required=True, 
        help="Model name to benchmark"
    )
    parser.add_argument(
        '--dataset', 
        type=str,
        choices=["summary"], 
        required=True, 
        help="Dataset name to use for testing"
    )
    parser.add_argument(
        '--output_len', 
        type=int, 
        required=True, 
        help="Number of tokens to generate per prompt"
    )

    args = parser.parse_args()
    main(args)

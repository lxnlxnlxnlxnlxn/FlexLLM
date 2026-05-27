import time
import pandas as pd
import torch
import gc
import argparse
from lmdeploy import Pipeline, GenerationConfig, PytorchEngineConfig


def main(args: argparse.ArgumentParser):

    # --------------------------
    # 1. Parse command-line arguments
    # --------------------------
    testcase_num = args.testcase_num    # Number of test prompts to run
    model = args.model                  # Model name for testing
    dataset = args.dataset              # Dataset name for testing
    output_len = args.output_len        # Number of tokens to generate per prompt

    # --------------------------
    # 2. Initialize LMDeploy pipeline
    # --------------------------
    model_path = "/workspace/data/models/" + model
    backend_config = PytorchEngineConfig(
        block_size=16, 
        num_gpu_blocks=128, 
        max_batch_size=32
    )
    pipe = Pipeline(model_path, backend_config=backend_config)

    # --------------------------
    # 3. Load and process dataset
    # --------------------------
    dataset_path = f"/workspace/data/dataset/{dataset}.json"
    prompts = []
    
    # Read JSON dataset and collect prompts
    raw_data_list = pd.read_json(dataset_path).values
    for raw_data in raw_data_list:
        prompt_text = raw_data[0]
        if len(prompt_text) > 0:
            prompts.append(prompt_text)
        # Stop when reaching test case count
        if len(prompts) == testcase_num:
            break

    # --------------------------
    # 4. Run inference and measure performance
    # --------------------------
    start_time = time.time()
    
    # Generation configuration
    gen_config = GenerationConfig(
        max_new_tokens=output_len,
        temperature=0.7,
        top_p=0.9
    )
    
    # Execute batch inference
    _, use_time_list = pipe(
        prompts, 
        use_tqdm=True, 
        gen_config=gen_config
    )
    
    end_time = time.time()

    # --------------------------
    # 5. Calculate performance metrics
    # --------------------------
    throughput = testcase_num / (end_time - start_time)    # Requests per second
    avg_jct = sum(use_time_list) / len(use_time_list)    # Average completion time

    # --------------------------
    # 6. Save results to log file
    # --------------------------
    with open("/workspace/output.txt", "a", encoding='utf-8') as f:
        f.write(
            f"model = {model}, dataset = {dataset}, "
            f"output_len = {output_len}, "
            f"total_time = {round(end_time - start_time, 3)}, "
            f"throughput = {round(throughput, 3)}, "
            f"avg_jct = {round(avg_jct, 3)}\n"
        )

    # --------------------------
    # 7. Clean up resources
    # --------------------------
    pipe.close()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LMDeploy Inference Benchmark for LLM Performance Testing"
    )

    # Test configuration parameters
    parser.add_argument(
        '--testcase_num', 
        type=int, 
        required=True, 
        help="Number of test prompts to run"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        choices=["llama-1-13b", "llama-2-13b", "llama-3-8b"], 
        required=True, 
        help="Model name to benchmark"
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=["alpaca", "chatbot", "gsm8k", "summary"], 
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

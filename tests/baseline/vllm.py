import json
import argparse

from transformers import logging
logging.set_verbosity_error()

from vllm import LLM, SamplingParams


def main(args: argparse.ArgumentParser):

    # Parse input arguments from command line
    model_name = args.model_name
    dataset = args.dataset
    gpu_block = args.gpu_block
    cpu_block = args.cpu_block
    batch_size = args.batch_size
    testcase_num = args.testcase_num
    output_len = args.output_len

    # Load dataset from JSON file
    dataset_path = f"/workspace/data/dataset/{dataset}.json"
    prompt_list = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    # Extract prompts from the first N test cases
    for data in data_list[0:testcase_num]:
        prompt = data["prompt"]
        prompt_list.append(prompt)
    
    # Load vLLM model with specified configuration (FlexLLM disabled)
    model_path = f"/workspace/data/models/{model_name}"
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        use_flexllm=False,
        max_num_seqs=batch_size,
        num_gpu_blocks=gpu_block,
        num_cpu_blocks=cpu_block
    )
    
    # Set generation parameters for deterministic inference
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=output_len
    )
    
    # Run batch inference
    _ = llm.generate(
        prompt_list,
        sampling_params,
        use_tqdm=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM Native Inference Benchmark (FlexLLM Disabled)"
    )

    parser.add_argument(
        '--model_name', 
        type=str,
        choices=["opt-13b", "llama-1-160m", "llama-1-13b", "llama-2-13b", "llama-3-8b"],
        required=True, 
        help="Name of the model to run inference"
    )
    parser.add_argument(
        '--dataset', 
        type=str,
        choices=["alpaca", "chatbot", "gsm8k"],
        required=True, 
        help="Dataset name for testing prompts"
    )
    parser.add_argument(
        '--gpu_block', 
        type=int,
        required=True, 
        help="Number of GPU KV cache blocks"
    )
    parser.add_argument(
        '--cpu_block', 
        type=int,
        required=True, 
        help="Number of CPU KV cache blocks"
    )
    parser.add_argument(
        '--batch_size', 
        type=int,
        required=True, 
        help="Maximum batch size for concurrent inference"
    )
    parser.add_argument(
        '--testcase_num', 
        type=int,
        required=True, 
        help="Number of test prompts to run"
    )
    parser.add_argument(
        '--output_len', 
        type=int,
        required=True, 
        help="Number of tokens to generate per prompt"
    )

    args = parser.parse_args()
    main(args=args)
    
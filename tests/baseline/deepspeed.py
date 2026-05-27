import pandas as pd
import mii
import time
import argparse


def main(args: argparse.ArgumentParser):

    # --------------------------
    # Parse command-line arguments
    # --------------------------
    testcase_num = args.testcase_num    # Number of test samples to run
    model = args.model                  # Model name for testing
    dataset = args.dataset              # Dataset name for testing
    output_len = args.output_len        # Number of tokens to generate per prompt

    # Model path on local disk
    model_path = "/workspace/data/models/" + model
    
    # Initialize MII inference pipeline
    pipe = mii.pipeline(model_path)

    # --------------------------
    # Load and prepare prompt dataset
    # --------------------------
    prompts = list()
    file_path = f"/workspace/data/dataset/{dataset}.json"
    raw_data_list = pd.read_json(file_path).values
    
    # Extract the first 'testcase_num' prompts
    for raw_data in raw_data_list[:testcase_num]:
        prompts.append(raw_data[0])

    # --------------------------
    # Run inference and measure performance
    # --------------------------
    start_time = time.time()
    
    # Run batch generation using MII
    _, use_time_list = pipe(prompts, max_new_tokens=output_len)
    
    # Reset pipeline status for next run
    pipe.reset_request_status()
    
    end_time = time.time()

    # --------------------------
    # Calculate performance metrics
    # --------------------------
    throughput = testcase_num / (end_time - start_time)                # Requests per second
    avg_jct = sum(use_time_list) / len(use_time_list)                  # Average job completion time

    print(f"model = {model}, dataset = {dataset}, output_len = {output_len}, finished !")

    # --------------------------
    # Save results to output log file
    # --------------------------
    with open("/workspace/output.txt", "a", encoding='utf-8') as f:
        f.writelines([
            f"model = {model}, dataset = {dataset}, "
            f"output_len = {output_len}, "
            f"total_time = {round(end_time - start_time, 3)}, "
            f"throughput = {round(throughput, 3)}, "
            f"avg_jct = {round(avg_jct, 3)}\n"
        ])

    # Destroy pipeline and release resources
    pipe.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MII Inference Benchmark for LLM Performance Testing"
    )

    # Test configuration parameters
    parser.add_argument(
        '--testcase_num', 
        type=int, 
        required=True, 
        help="Number of test samples/prompts to run"
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

from vllm import LLM, SamplingParams


def main():

    # Sample prompts.
    prompts = ["I like you", "I don't like you", "do you like me?"]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=32)

    # Create an LLM
    model_path = "/workspace/data/models/llama-3-8b"
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        use_flexllm=True,
        block_size=16
    )
    llm.set_flexllm_config(
        model_name="llama-3-8b",
        onload_start=0.3,
        onload_end=0.6,
        backup_rate=0.2,
        block_size=16
    )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs[0]:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("---------------------------------------")
    print(outputs[1])


if __name__ == "__main__":
    main()

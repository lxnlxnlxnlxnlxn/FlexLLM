# Flexllm

[TOC]

## Introduction

We suppose FlexLLM, an efficient tensor swapping and re-computation framework for large language model inference. FlexLLM performs better on LLM inferences tasks compared with multiple baselines, including vLLM, LMDeploy, Deepspeed-MII and TensorRT. FlexLLM is constructed by three components: the Offloader, the Onloader, and the Memory Manager. The Offloader performs asynchronous backup during the execution of the inference task, which enables zero additional overhead by KV Cache eviction when GPU memory is over-utilized. The Onloader generates KV Cache for new or pre-evicted requests with the minimal delay after GPU memory is released by previous requests. Serving as the underlying support for the cooperation of the above two modules, the memory management module adopts adaptive block size configuration and block table operations to enable efficient dynamic scheduling of video memory resources and performance optimization for inference tasks. The Memory Manager provides kernel-level support for the collaboration between the Offloader and the Onloader via adaptive parameter search, enabling dynamic and efficient resource management.

## Implementation

We have implemented a prototype of FlexLLM with 2.6k lines of Python code, including 500 lines for the scheduler, 200 lines for the memory manager and 1900 lines for others. FlexLLM is built upon CUDA 11.8, PyTorch 2.0.1, Ray 2.7.1 and vLLM 0.2.5, while tested on an Ubuntu 22.04 server which is equipped with Intel(R) Xeon(R) CPUs and NVIDIA V100 40GB GPUs, as well as PCIe 4.0 for GPU-CPU data transmission. 

- Datasets
  - [Alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca)
  - [Chatbot](https://huggingface.co/datasets/alespalla/chatbot_instruction_prompts)
  - [GSM8K](https://hf-mirror.com/datasets/openai/gsm8k)
  - [Summary](https://huggingface.co/datasets/khwrali011/summary-dataset)
- Models
  - [LLaMA-1-13B](https://huggingface.co/huggyllama/llama-13b)
  - [LLaMA-2-13B](https://huggingface.co/meta-llama/Llama-2-13b)
  - [LLaMA-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- Baselines
  - [vLLM](https://github.com/vllm-project/vllm)
  - [LMDeploy](https://github.com/InternLM/lmdeploy)
  - [DeepSpeed-MII](https://github.com/deepspeedai/DeepSpeed-MII)
  - [TensorRT](https://github.com/NVIDIA/TensorRT)

## Installation

```
git clone https://github.com/lxnlxnlxnlxnlxn/Flexllm
cd Flexllm
pip install -e .
```

## Quick Start

##### 1、installation verification

```
python ./tests/backend/vllm_test.py
python ./tests/backend/flexllm_test.py
```

##### 2、offline data collection

```
bash ./scripts/pcie.sh       # pcie bandwidth test
bash ./scripts/tgen.sh       # single step inference time collection
bash ./scripts/train.sh      # single step inference time prediction
bash ./scripts/simulate.sh   # choose parameter by simulation 
```

##### 3、end to end performance test

```
bash ./e2e/e2e.sh
```

##### 4、baseline test

```
bash ./baseline/deepspeed.sh
bash ./baseline/lmdeploy.sh
bash ./baseline/trt.sh
bash ./baseline/vllm.sh
```

## Results

##### 1、End-to-end performance

FlexLLM outperforms all baselines on overall throughput, it achieves up to 2.7×, 2.9×, 2.1×, 2.8× times of performance improvement compared with vLLM, Deepspeed-MII, LMDeploy, TensorRT respectively.

![throughput](.\data\e2e\throughput.png)

FlexLLM outperforms all baselines on the average value of per-request latency, it achieves up to 2.7×, 3.5×, 2.1×, 2.8× times of performance improvement compared with vLLM, Deepspeed-MII, LMDeploy, TensorRT respectively.

![jct](.\data\e2e\jct.png)

##### 2、Parameter Selection

FlexLLM leverages a simulator to predict the execution process of inference tasks, identifying the optimal parameter combination based on performance feedback under various parameter settings. We conducted three groups of experiments. Each group of experimental results is presented in four figures. From left to right, the first and third figures show the normalized performance scores given by the simulator, while the second and fourth figures illustrate the normalized real performance data recorded by e2e experiments, with overall throughput as the evaluation metric.

As observed from the figures, the simulator achieves accurate performance predictions for all conditions and enables optimal parameter combinations selection efficiently which is enclosed with a black rectangular box.

![args](.\data\args\args.png)

##### 3、Preemption Overhead

When GPU memory is sufficient, FlexLLM regenerates the KV Cache of previously evicted user requests. Since the KV Cache of these requests has been backed up in CPU memory before its eviction, FlexLLM searches the split points within the sequences. The first half is recomputed via one prefill stage, while the second half is swapped in from CPU memory. Since the latency of re-computation and swapping process can be obtained offline, we can find the optimal split point for each sequence for fully parallelization, minimizing the total overhead as a result.

<img src=".\data\preempt\preempt.png" alt="preempt" style="zoom: 15%;" />
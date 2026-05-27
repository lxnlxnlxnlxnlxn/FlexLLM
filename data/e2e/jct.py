from typing import Any, List, Dict
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


def main() -> None:
    fig, axes = plt.subplots(3, 4, figsize=(30, 15), dpi=300)
    axes_list = axes.flatten()

    i = 0
    for model in ["llama-1-13b", "llama-2-13b", "llama-3-8b"]:
        for dataset in ["alpaca", "chatbot", "gsm8k", "summary"]:
            raw_data = read(model=model, dataset=dataset)
            normalized_data = normalize_data(raw_data)
            fit(ax=axes_list[i], data=normalized_data, model=model, dataset=dataset)
            i += 1

    fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.80)
    fig.suptitle("Average Inference Time", fontsize=30, fontweight="bold", y=0.90)
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, 
               fontsize=25, frameon=False, bbox_to_anchor=(0.5, 0.88))
    plt.savefig(".\\data\\e2e\\jct.png", dpi=300, bbox_inches='tight')


def normalize_data(data: Dict[str, List[float]]) -> Dict[str, List[float]]:
    normalized = {}
    flex = data["flexllm"]
    for k, v in data.items():
        normalized[k] = [round(vi / fi, 3) if fi != 0 else 0 for vi, fi in zip(v, flex)]
    return normalized


def fit(ax: Any, data: Dict[str, List[float]], model: str, dataset: str) -> None:
    output_lengths = [16, 32, 64, 128, 256, 512]
    frameworks = list(data.keys())
    n_frameworks = len(frameworks)
    color_config = {
        "flexllm": "#2E86AB",
        "vllm": "#A23B72",
        "lmdeploy": "#F18F01",
        "ds-mii": "#C73E1D",
        "trt": "#6A994E"}

    bar_width = 0.15
    x = np.arange(len(output_lengths))
    x_offsets = [x + i * bar_width - (n_frameworks - 1) * bar_width / 2 for i in range(n_frameworks)]

    for i, fw in enumerate(frameworks):
        ax.bar(x_offsets[i], data[fw], width=bar_width, color=color_config[fw], 
               alpha=0.85, edgecolor="white", linewidth=0.8, label=fw)

    ax.set_xticks(x)
    ax.set_xticklabels(output_lengths, fontsize=24)
    ax.text(0.5, -0.15, "output length", transform=ax.transAxes, 
            ha="center", va="top", fontsize=22, fontweight="bold")

    ax.set_ylabel("Avg. Overhead(s)", fontsize=25, fontweight="bold")
    ax.tick_params(axis='y', labelsize=25)

    all_values = []
    for fw in frameworks:
        all_values.extend(data[fw])
    y_max = max([v for v in all_values if np.isfinite(v)]) * 1.1
    ax.set_ylim(0, y_max)

    ax.set_title(f"{model}/{dataset}", fontsize=28, fontweight="bold", pad=10)

    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if ax.get_legend() is not None:
        ax.get_legend().remove()


def read(model: str, dataset: str) -> Dict[str, List[float]]:
    data_path = os.path.join(r".\data\e2e", f"{model}-{dataset}.xlsx")
    df = pd.read_excel(data_path, sheet_name="jct", header=None, engine="openpyxl")
    raw = df.values
    return {
        "flexllm": raw[4, 2:8].astype(float).tolist(),
        "vllm": raw[5, 2:8].astype(float).tolist(),
        "lmdeploy": raw[6, 2:8].astype(float).tolist(),
        "ds-mii": raw[7, 2:8].astype(float).tolist(),
        "trt": raw[8, 2:8].astype(float).tolist()
    }


if __name__ == "__main__":
    main()
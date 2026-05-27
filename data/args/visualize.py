from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Patch

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


def main():
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=300)
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.5, wspace=0.4, top=0.80, bottom=0.05)

    fig.suptitle("Performance Comparison under Different Parameters ", fontsize=32, fontweight='bold', y=0.93)
    cases = ["llama-1-13b-alpaca", "llama-2-13b-chatbot", "llama-3-8b-gsm8k"]
    row_titles = ["llama-1-13b/alpaca", "llama-2-13b/chatbot", "llama-3-8b/gsm8k"]
    
    idx = 0
    for row_id, case in enumerate(cases):
        fig.text(0.5, 0.82 - row_id * 0.28, row_titles[row_id], 
                 ha='center', fontsize=25, fontweight='bold')
        for arg_name in ["rate", "block"]:
            for test_name in ["arg", "res"]:
                parse(ax=axes[idx], case=case, arg_name=arg_name, test_name=test_name)
                idx += 1

    legend_elements = [Patch(facecolor="#2c3e50", label="0~0.5"), 
                       Patch(facecolor="#8b5a2b", label="0.5~0.7"), 
                       Patch(facecolor="#2e8b57", label="0.7~0.8"), 
                       Patch(facecolor="#b8860b", label="0.8~0.9"), 
                       Patch(facecolor="#2f6a91", label="0.9~0.95"), 
                       Patch(facecolor="#a52a2a", label="0.95~1")]    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.90), 
               ncol=6, fontsize=18, frameon=False)

    plt.savefig(".\\args\\args.png", dpi=300, bbox_inches='tight')


def parse(ax: Any, case: str, arg_name: str, test_name: str) -> None:
    file_path = f".\\args\\{case}.xlsx"
    sheet_name = f"{arg_name}-{test_name}"
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    raw_data = []
    row_start = 4
    row_end = 9 if arg_name == "rate" else 4
    col_start = 2
    col_end = 7

    for i in range(row_start, row_end + 1):
        row = []
        for j in range(col_start, col_end + 1):
            val = df.iloc[i, j]
            if val == "/" or pd.isna(val):
                row.append(np.nan)
            else:
                row.append(float(val))
        raw_data.append(row)
    
    perf_matrix = np.array(raw_data, dtype=np.float64)

    if test_name == "arg":
        best_perf = np.nanmin(perf_matrix)
        ratio_matrix = 1 / perf_matrix * best_perf
    else:
        best_perf = np.nanmax(perf_matrix)
        ratio_matrix = perf_matrix / best_perf
    
    perf_matrix = np.flipud(perf_matrix)
    ratio_matrix = np.flipud(ratio_matrix)
    valid_rows = ~np.all(np.isnan(perf_matrix), axis=1)
    perf_matrix = perf_matrix[valid_rows]
    ratio_matrix = ratio_matrix[valid_rows]

    fit(ax=ax, x_label="regen thres", y_label="stop thres", 
        ratio_matrix=ratio_matrix, perf_matrix=perf_matrix, is_block=(arg_name == "block"))


def fit(ax: Any, x_label: str, y_label: str, ratio_matrix=None, perf_matrix=None, is_block=False) -> None:
    color_mat = np.array(ratio_matrix, dtype=np.float32)
    color_mat = np.where(np.isnan(perf_matrix), np.nan, color_mat)
    colors = ["#2c3e50", "#8b5a2b", "#2e8b57", 
              "#b8860b", "#2f6a91", "#a52a2a"]
    bounds = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    
    _ = ax.imshow(color_mat, cmap=cmap, norm=norm, aspect='equal')

    n_rows, n_cols = color_mat.shape
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))

    if is_block:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)
        ax.set_xlabel("block size", fontsize=20, fontweight='bold')
        ax.set_xticklabels([4, 8, 16, 32, 64, 128], fontsize=15)
    else:
        ax.set_xlabel(x_label, fontsize=20, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=20, fontweight='bold')
        ax.set_xticklabels([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], fontsize=15)
        y_ticks = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4][:n_rows]
        ax.set_yticklabels(y_ticks, fontsize=15)

    if ratio_matrix is not None:
        val_mat = np.ma.masked_invalid(ratio_matrix)
        best_val = val_mat.max()
        y_idx, x_idx = np.unravel_index((val_mat == best_val).argmax(), val_mat.shape)
        rect = Rectangle((x_idx - 0.5, y_idx - 0.5), 1, 1,
                         linewidth=5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        n_rows, n_cols = color_mat.shape
        ax.set_xlim(-0.6, n_cols - 0.4)
        ax.set_ylim(-0.6, n_rows - 0.4)


if __name__ == "__main__":
    main()
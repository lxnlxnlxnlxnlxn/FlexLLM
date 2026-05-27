from typing import Any, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


def main() -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 15), 
                                        height_ratios=[1, 1, 1])
    
    fit(id=1, ax=ax1, model_name="llama-1-13b")
    fit(id=2, ax=ax2, model_name="llama-2-13b")
    fit(id=3, ax=ax3, model_name="llama-3-8b")

    lines, labels = ax1.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), 
               fontsize=25, frameon=True, fancybox=True, shadow=True, ncol=2)
    fig.suptitle("Preemption Overhead", fontsize=28, 
                 fontweight='bold', fontfamily='Arial')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(hspace=0.6)
    fig.savefig(".\preempt\preempt.png", dpi=300)


def fit(id: int, ax: Any, model_name: str) -> None:
    file_name = f".\preempt\{model_name}.xlsx"
    df = pd.read_excel(file_name, header=None)
    
    recomp_len_list = (pd.to_numeric(df.iloc[1, :], errors="coerce").tolist())
    recomp_t_list = (pd.to_numeric(df.iloc[2, :], errors="coerce").tolist())
    swap_t_list = (pd.to_numeric(df.iloc[3, :], errors="coerce").tolist())

    ax.plot(recomp_len_list, recomp_t_list, color="orange", linewidth=4, 
            marker='o', markersize=6, markerfacecolor='red',
            markeredgecolor="red", markeredgewidth=3,
            label='recomputation time', zorder=3)
    ax.plot(recomp_len_list, swap_t_list, color="green", linewidth=4, 
            marker='o', markersize=6, markerfacecolor='black',
            markeredgecolor="blue", markeredgewidth=3,
            label='swapping time', zorder=3)
    
    x_intersect, y_intersect = find_mid(ax=ax, x=recomp_len_list, 
                                        y1=recomp_t_list, y2=swap_t_list)
    if id == 1:
        x_text = x_intersect - 35
        y_text = y_intersect + 20
    elif id == 2:
        x_text = x_intersect - 20
        y_text = y_intersect - 15
    else:
        x_text = x_intersect - 20
        y_text = y_intersect + 7

    ax.text(x_text, y_text, f'({x_intersect:.0f}, {y_intersect:.0f})',
            fontsize=18, fontweight='bold', color='white', 
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    ax.set_xlabel('recompute length', fontsize=25, fontweight='bold', 
                  fontfamily='Times New Roman', color='#333333', labelpad=15)
    ax.set_ylabel('t/ms', fontsize=25, fontweight='bold', 
                  fontfamily='Times New Roman', color='#333333', labelpad=15)
    ax.tick_params(axis='x', labelsize=20, labelcolor='#555555', labelrotation=0)
    ax.tick_params(axis='y', labelsize=20, labelcolor='#555555', width=2, length=6)
    ax.set_title(model_name, fontsize=26, fontweight="bold", pad=15)
    

def find_mid(ax: Any, x: List[int], y1: List[float], y2: List[float]) -> Tuple[float, float]:
    x, y1, y2 = np.array(x), np.array(y1), np.array(y2)
    mask = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
    x, y1, y2 = x[mask], y1[mask], y2[mask]

    x_intersect = None
    y_intersect = None

    for i in range(len(x) - 1):
        d_curr = y1[i] - y2[i]
        d_next = y1[i+1] - y2[i+1]
        
        if d_curr * d_next <= 0:
            dx = x[i+1] - x[i]
            dy1 = y1[i+1] - y1[i]
            t = abs(d_curr) / (abs(d_curr) + abs(d_next))
            x_intersect = x[i] + t * dx
            y_intersect = y1[i] + t * dy1
            break

    if x_intersect is not None and np.isfinite(x_intersect) and np.isfinite(y_intersect):
        ax.scatter(
            x_intersect, y_intersect,
            color='purple', s=200, zorder=6,
            edgecolors='purple', linewidth=3
        )

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y1), np.max(y1)        
        rect_w = (x_max - x_min) * 0.1
        rect_h = (y_max - y_min) * 0.2
        rect_x = x_intersect - rect_w / 2
        rect_y = y_intersect - rect_h / 2

        if np.isfinite(rect_x) and np.isfinite(rect_y):
            ax.add_patch(Rectangle((rect_x, rect_y), 
                                   rect_w, rect_h, 
                        linewidth=4, edgecolor='black', 
                        facecolor='none', zorder=5))
    return x_intersect, y_intersect


if __name__ == "__main__":
    main()

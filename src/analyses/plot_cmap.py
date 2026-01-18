import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_results(data_dict, title_prefix, xlabel, ylabel, filename, 
                 xticklabels=None, yticklabels=None, class_ref_csv=None):

    class_mapping = {}
    if class_ref_csv and os.path.exists(class_ref_csv):
        try:
            df = pd.read_csv(class_ref_csv)
            class_mapping = dict(zip(df['ID'], df['Label']))
        except Exception as e:
            print(e)

    n_plots = len(data_dict)
    n_cols = 5
    n_rows = (n_plots + 1) // n_cols if n_plots > 0 else 1
    n_rows = max(n_rows, 5) 

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows)) 
    axs = axs.flatten()

    vmin, vmax = -1, 1
    cmap = "coolwarm"

    for idx, cls in enumerate(sorted(data_dict.keys())):
        ax = axs[idx]

        sns.heatmap(data_dict[cls], ax=ax, cmap=cmap, center=0,
                    vmin=vmin, vmax=vmax, square=True,
                    xticklabels=xticklabels, yticklabels=yticklabels,
                    cbar=False) 
        
        class_name = class_mapping.get(cls, f"Class {cls}")
        
        ax.set_title(f"{class_name}", fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        
        ax.tick_params(axis='both', which='major', labelsize=10)

    for i in range(len(data_dict), len(axs)):
        axs[i].axis("off")

    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 

    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.ax.tick_params(labelsize=14) 

    plt.savefig(filename)
    plt.show()
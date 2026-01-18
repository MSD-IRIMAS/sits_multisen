import numpy as np
import pandas as pd
from src.utils.utils import get_data_paths

def compute_avg_channel_correlation(
    base_path, 
    folds, 
    included_classes, 
    normalize=False, 
    stats_csv=None
):
    if normalize:
        df = pd.read_csv(stats_csv)
        df = df.sort_values('band_idx')

        min_vals = df['global_min'].values.reshape(-1, 1)
        max_vals = df['global_max'].values.reshape(-1, 1)
        
        denom = max_vals - min_vals
        denom[denom == 0] = 1.0

    sum_corr = {i: np.zeros((10, 10)) for i in included_classes}
    counts = {i: 0 for i in included_classes}

    for x_path, y_path in get_data_paths(base_path, folds):
        
        y = np.load(y_path).item()
        
        if y in included_classes:
            x = np.load(x_path).astype(np.float32)

            if normalize:
                x = (x - min_vals) / (denom + 1)

            corr_matrix = np.corrcoef(x, rowvar=True)
            
            if np.isnan(corr_matrix).any():
                corr_matrix = np.nan_to_num(corr_matrix)

            sum_corr[y] += corr_matrix
            counts[y] += 1

    avg_corr = {}
    for cls in included_classes:
        if counts[cls] > 0:
            avg_corr[cls] = sum_corr[cls] / counts[cls]

    print(f"Terminé. Classes traitées : {list(avg_corr.keys())}")
    return avg_corr, counts


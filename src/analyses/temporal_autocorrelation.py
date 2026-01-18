import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from src.utils.utils import get_data_paths

def compute_avg_autocorrelation(
    base_path, 
    folds, 
    included_classes, 
    max_lag,
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

    sum_acf = {i: np.zeros((10, max_lag)) for i in included_classes}
    counts = {i: 0 for i in included_classes}

    for x_path, y_path in get_data_paths(base_path, folds):

        y = np.load(y_path).item()
        
        if y in included_classes:
            x = np.load(x_path).astype(np.float32)

            if normalize:
                x = (x - min_vals) / (denom + 1e-8)

            channels_acf = []
            for channel in x:
                try:
                    res = acf(channel, nlags=max_lag, fft=True)
                    if np.isnan(res).any():
                        res = np.zeros_like(res)
                except Exception:
                    res = np.zeros(max_lag + 1)
                
                channels_acf.append(res[1:])#no use of lag 0 
            
            acf_matrix = np.array(channels_acf)
            
            sum_acf[y] += acf_matrix
            counts[y] += 1

    avg_acf = {}
    for cls in included_classes:
        if counts[cls] > 0:
            avg_acf[cls] = sum_acf[cls] / counts[cls]
            
    print(f"Terminé. Classes traitées : {list(avg_acf.keys())}")
    return avg_acf, counts
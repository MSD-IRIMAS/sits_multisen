import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_global_stats_from_one_fold(
    base_path,
    fold_to_scan=0, 
    sub_folder_x="x",
    nb_bands=10,
    output_csv="global_dataset_stats.csv"
):
    splits_to_scan = ["train", "test"]
    
    print(f"--- Calcul des statistiques globales (via Fold {fold_to_scan}) ---")
    print(f"Racine : {base_path}")

    global_min = np.full(nb_bands, np.inf)
    global_max = np.full(nb_bands, -np.inf)

    for split in splits_to_scan:
        data_dir = os.path.join(base_path, "folds", f"fold_{fold_to_scan}", split, sub_folder_x)
        
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

        for filename in tqdm(files, desc=f"Scanning {split}", unit="img"):
            file_path = os.path.join(data_dir, filename)
            
            try:
                data = np.load(file_path)
                data = data.T
                
                local_min = np.min(data, axis=1)
                local_max = np.max(data, axis=1)
                
                global_min = np.minimum(global_min, local_min)
                global_max = np.maximum(global_max, local_max)
                
            except Exception as e:
                print(e)

    stats_list = []
    for b in range(nb_bands):
        stats_list.append({
            "band_idx": b,
            "global_min": global_min[b],
            "global_max": global_max[b]
        })

    df = pd.DataFrame(stats_list)
    output_path = os.path.join(base_path, output_csv)
    df.to_csv(output_path, index=False)

    print(df)



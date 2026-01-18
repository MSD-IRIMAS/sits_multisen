import os
import numpy as np
import rasterio
import glob
import pandas as pd
import tqdm

def get_data_paths(base_path, folds):
    for fold in folds:
        x_dir = os.path.join(base_path, fold, "test", "x")
        y_dir = os.path.join(base_path, fold, "test", "y")
        
        if not os.path.exists(x_dir): continue

        for fname in os.listdir(x_dir):
            if fname.endswith(".npy"):
                yield os.path.join(x_dir, fname), os.path.join(y_dir, fname)


#returns the array list with all the bands of patches_s list
def read_bands(folder_path_s, patches_s, NB_BANDS):
    image_list = []
    
    for patch_name in patches_s:
        image_bands = []
        with rasterio.open(folder_path_s + patch_name) as src:
            for i in range(NB_BANDS):
                image_bands.append(src.read(i + 1))
        image_list.append(image_bands)    
    return np.array(image_list, dtype=object)

def get_date_filename(filename):
    return filename.split("_")[1]

 #construct ground reference filename from label filename
def construct_gr_filename(label_filename):
    parts = label_filename.split("_")
    tile = parts[0]
    x = parts[1]
    y = parts[2].split(".")[0]

    return tile + "_GR_" + x + "_" + y + ".tif"

#deconstruct ground reference filename into variables : tile, x, y
def deconstruct_gr_filename(filename):
    parts = filename.split("_")
    tile = parts[0]
    x = parts[2]
    y = parts[3].split(".")[0]
    return tile, x, y

def get_fold_filepaths(
    base_path, 
    fold_id=0, 
    split="train", 
    x_folder_name="x", 
    y_folder_name="y"
):
    
    dir_x = os.path.join(base_path, "folds", f"fold_{fold_id}", split, x_folder_name)
    dir_y = os.path.join(base_path, "folds", f"fold_{fold_id}", split, y_folder_name)

    #we sort to be sure that x and y are ordered the same way
    x_paths = sorted(glob.glob(os.path.join(dir_x, "*.npy")))
    y_paths = sorted(glob.glob(os.path.join(dir_y, "*.npy")))
    
    return x_paths, y_paths

def load_fold_data(
    base_path, 
    fold_id=0, 
    split="train", 
    data_type="x",
    normalize=False, 
    stats_csv_path=None
):
    folder_path = os.path.join(base_path, "folds", f"fold_{fold_id}", split, data_type)

    file_paths = sorted(glob.glob(os.path.join(folder_path, "*.npy")))

    data_list = []
    for f in tqdm(file_paths, unit="img"):
        arr = np.load(f)
        data_list.append(arr)
    
    full_data = np.array(data_list)
    
    if normalize and data_type == 'x':
        df = pd.read_csv(stats_csv_path)
        
        min_vals = df['global_min'].values
        max_vals = df['global_max'].values
        
        #reshape for broadcasting
        min_vals = min_vals.reshape(1, -1, 1)
        max_vals = max_vals.reshape(1, -1, 1)

        denominator = max_vals - min_vals
        denominator[denominator == 0] = 1.0
        
        full_data = (full_data - min_vals) / (denominator + 1)
        
        full_data = full_data.astype(np.float32)

    return full_data
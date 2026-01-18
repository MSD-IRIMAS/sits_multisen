import os
import json
import numpy as np
import rasterio
from utils.utils import construct_gr_filename, deconstruct_gr_filename, read_bands, get_date_filename

def extract_pixel_timeseries(
    label_list,
    paths,
    satellite="corresponding_s2",
    nb_bands=10,
    img_size=(256, 256)
):
    subfolders = ["tiles", "timestamps", "coordonates", "multisenge_s_y", "multisenge_s_x"]
    for sub in subfolders:
        os.makedirs(os.path.join(paths['output'], sub), exist_ok=True)

    counter = 0

    for label in label_list:
        try:
            gr_filename = construct_gr_filename(label)
            json_path = os.path.join(paths['labels'], label)
            gr_path = os.path.join(paths['ground_reference'], gr_filename)

            if not os.path.exists(json_path) or not os.path.exists(gr_path):
                print(f"Fichier manquant pour {label}, passage au suivant.")
                continue

            with open(json_path, 'r') as file:
                label_json = json.load(file)

            patches_s = label_json[satellite].split(';')
            tile, x, y_coord = deconstruct_gr_filename(gr_filename)

            patch_list = read_bands(paths['s2'], patches_s, nb_bands)

            timestamps = [get_date_filename(p) for p in patches_s]
            timestamps_np = np.array(timestamps)
            sorted_indices = np.argsort(timestamps_np)
            sorted_timestamps = timestamps_np[sorted_indices]
            # --------------------

            with rasterio.open(gr_path) as src:
                image_labels = src.read(1)

                for i in range(img_size[0]):
                    for j in range(img_size[1]):
                        
                        tiles = np.array([tile])
                        coords = np.array([(x, y_coord)])                       
                        current_y = np.array([image_labels[i][j]])

                        multisenge_s_X = []
                        
                        for b in range(nb_bands):
                            band_values_over_time = [patch_list[p][b][i][j] for p in range(len(patch_list))]
                            multisenge_s_X.append(band_values_over_time)

                        X_array = np.array(multisenge_s_X) # shape: (Bands, Time)
                        sorted_multisenge_s_X = X_array[:, sorted_indices]

                        clean_label_name = os.path.splitext(label)[0]
                        sat_suffix = satellite.split("_")[1] if "_" in satellite else satellite
                        ts_name = f"{clean_label_name}_{sat_suffix}_{i}_{j}.npy"

                        base_out = paths['output']
                        np.save(os.path.join(base_out, "tiles", ts_name), tiles)
                        np.save(os.path.join(base_out, "timestamps", ts_name), sorted_timestamps)
                        np.save(os.path.join(base_out, "coordonates", ts_name), coords)
                        np.save(os.path.join(base_out, "multisenge_s_y", ts_name), current_y)
                        np.save(os.path.join(base_out, "multisenge_s_x", ts_name), sorted_multisenge_s_X)
                        

            counter += 1
            print(f"Label {label} terminé ! Total traités : {counter}")
            
        except Exception as e:
            print(f"Erreur critique sur le label {label} : {e}")
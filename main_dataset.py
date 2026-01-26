import os
import logging
from omegaconf import DictConfig

from src.dataset.creation import extract_pixel_timeseries_with_folds
from src.dataset.min_max_calcul import calculate_global_stats_from_one_fold

log = logging.getLogger(__name__)

def main_dataset(args: DictConfig):
    
    output_dir = args.dataset.base_path

    paths = {
        'folds': args.dataset.paths.folds_csv,
        'output': output_dir,
        'labels': args.dataset.paths.labels,
        'ground_reference': args.dataset.paths.ground_reference,
        's2': args.dataset.paths.s2_images
    }

    for key, path in paths.items():
        if key != 'output' and not os.path.exists(path):
            raise FileNotFoundError()

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        label_list = [f for f in os.listdir(paths['labels']) if f.endswith('.json')]
    except Exception as e:
        raise RuntimeError(e)

    n_folds = len(args.dataset.folds) if isinstance(args.dataset.folds, list) else 5

    extract_pixel_timeseries_with_folds(
        label_list=label_list,
        paths=paths,
        n_folds=n_folds,
        satellite=args.dataset.satellite,
        nb_bands=args.analysis.channel_count,
        img_size=(args.dataset.img_size, args.dataset.img_size),
        link_method=args.dataset.link_method
    )
    
    stats_full_path = args.dataset.stats_csv_path
    stats_filename = os.path.basename(stats_full_path)

    calculate_global_stats_from_one_fold(
        base_path=output_dir,
        fold_to_scan=0,
        sub_folder_x="x",
        nb_bands=args.analysis.channel_count,
        output_csv=stats_filename 
    )
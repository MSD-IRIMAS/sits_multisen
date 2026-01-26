from omegaconf import DictConfig
import numpy as np
import os
import hydra.utils 

from src.analyses.temporal_autocorrelation import compute_avg_autocorrelation
from src.analyses.inter_channel_correlation import compute_avg_channel_correlation
from src.analyses.plot_cmap import plot_results

def main_analyses(cfg: DictConfig):

    if cfg.output.save_plots:
        os.makedirs(cfg.output.dir, exist_ok=True)

    all_classes = range(1, cfg.dataset.n_classes + 1)
    included_classes = [c for c in all_classes if c not in cfg.dataset.excluded_classes]
    
    class_ref_csv_path = None
    if "class_ref_path" in cfg.dataset and cfg.dataset.class_ref_path:
        class_ref_csv_path = hydra.utils.to_absolute_path(cfg.dataset.class_ref_path)

    stats_csv_path = None
    if "stats_csv_path" in cfg.dataset and cfg.dataset.stats_csv_path:
        stats_csv_path = hydra.utils.to_absolute_path(cfg.dataset.stats_csv_path)

    do_normalize = cfg.dataset.get("normalize", False)

    #autocorrelation
    if cfg.run.autocorrelation:
        acf_results, _ = compute_avg_autocorrelation(
            base_path=cfg.dataset.base_path,
            folds=cfg.dataset.folds,
            included_classes=included_classes,
            max_lag=cfg.analysis.max_lag,
            normalize=do_normalize,
            stats_csv=stats_csv_path
        ) 
        
        if cfg.output.save_plots:
            full_out_path = os.path.join(cfg.output.dir, "autocorrelation.pdf")
            
            plot_results(
                data_dict=acf_results,
                title_prefix="AutoCorr",
                xlabel="Lag",
                ylabel="Channel",
                filename=full_out_path,
                xticklabels=np.arange(1, cfg.analysis.max_lag + 1),
                yticklabels=np.arange(1, cfg.analysis.channel_count + 1),
                class_ref_csv=class_ref_csv_path
            )

    # inter-channel correlation
    if cfg.run.inter_channel:
        corr_results, _ = compute_avg_channel_correlation(
            base_path=cfg.dataset.base_path,
            folds=cfg.dataset.folds,
            included_classes=included_classes,
            normalize=do_normalize,
            stats_csv=stats_csv_path
        )

        if cfg.output.save_plots:
            full_out_path = os.path.join(cfg.output.dir, "channel_correlation.pdf")
            
            plot_results(
                data_dict=corr_results,
                title_prefix="ChanCorr",
                xlabel="Channel",
                ylabel="Channel",
                filename=full_out_path,
                xticklabels=np.arange(1, cfg.analysis.channel_count + 1),
                yticklabels=np.arange(1, cfg.analysis.channel_count + 1),
                class_ref_csv=class_ref_csv_path
            )

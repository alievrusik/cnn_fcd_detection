import torch
import numpy as np
import os
from datetime import datetime
import pandas as pd
import wandb

def calculate_top_k_metric(target, predicted, k=20):
    top_indices = np.argsort(predicted)[::-1][:k]
    return (target[top_indices] > 0).mean()


def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(torch.sigmoid(output), min=1e-8, max=1-1e-8)
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))


def setup_experiment(title, config):
    current_time = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    experiment_name = "{}@{}".format(title, current_time)
    os.makedirs(f'./plots/{experiment_name}', exist_ok=True)
    os.makedirs(f'./checkpoints/{experiment_name}', exist_ok=True)
    os.makedirs(f'./data/predicted_masks/{experiment_name}', exist_ok=True)

    os.environ['WANDB_API_KEY'] = '7232fe584829d234576e43351c359921cab60b1b'
    wandb.init(project="cnn-fcd-detection", name=experiment_name, notes=config.message)
    wandb.config.update(config)
    return experiment_name, current_time


def log_experiment(config, current_time, top_k_metric):
    log_df = pd.read_csv('./log_dataframe.csv')
    curr_idx = len(log_df)
    log_df.loc[curr_idx] = [
        current_time, 
        config.title,
        config.height,
        config.width,
        config.use_coronal,
        config.use_sagital,
        config.augment,
        config.hard_labeling,
        config.lr,
        config.batch_size,
        config.latent_size,
        config.nb_epochs,
        config.weight_decay,
        config.weight_of_class,
        config.dropout_rate, 
        config.temporal_division,
        config.nb_of_modalities,
        top_k_metric
    ]
    log_df.to_csv('./log_dataframe.csv', index=False)


def get_brain_name_by_idx(idx, fcd=True):
    if fcd:
        return os.path.join('./data/fcd_brains/', f'fcd_{str(idx).zfill(2)}.nii.gz')
    else:
        return os.path.join('./data/fcd_brains/', f'nofcd_{str(idx).zfill(2)}.nii.gz')


def get_mask_name_by_idx(idx):
    return os.path.join('./data/masks/', f'mask_{str(idx).zfill(2)}.nii.gz')


def get_pred_mask_name_by_idx(experiment_name, idx, fcd=True):
    if fcd:     
        return os.path.join(
            './data/predicted_masks/', experiment_name, f'predicted_mask_{str(idx).zfill(2)}.nii.gz'
        )
    else:
        return os.path.join(
            './data/predicted_masks/', experiment_name, f'nofcd_predicted_mask_{str(idx).zfill(2)}.nii.gz'
        )



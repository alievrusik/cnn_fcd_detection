from models import PatchAutoEncoder, PatchModel
from mask_generator import FCDMaskGenerator
from utils import *
from dataset import PatchTrainDataset, PatchValDataset, dummy_collate
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import wandb
import random

# TEMPORAL_IDXS = [0, 3, 6, 9, 11, 12, 13]
# NON_TEMPORAL_IDXS = [1, 2, 4, 5, 7, 8, 10, 14]
NB_OF_FCD_SUBJECTS = 26
NB_OF_NOFCD_SUBJECTS = 15
NB_OF_CONTROL_SUBJECTS = 100500
DEFAULT_NB_OF_PATCHES = 8394
NUM_WORKERS = 12

CAPTION_DICT = {
    0: 't1 original',
    1: 't1 mirrored',
    2: 't2 original',
    3: 't2 mirrored',
    4: 'flair original',
    5: 'flair mirrored',
}


def train_ae(mods,
             h, w, use_coronal, use_sagital, latent_dim, batch_size,
             lr, n_epochs, p, loo_idx, parallel, experiment_name):
    nb_of_dims = 1 + 1*int(use_coronal) + 1*int(use_sagital)

    X_train_fcd = PatchTrainDataset('./data/saved_patches/', True, 2*mods*nb_of_dims, h, w, batch_size, loo_idx)
    X_train_nofcd = PatchTrainDataset('./data/saved_patches/', False, 2*mods*nb_of_dims, h, w, batch_size, None)
    X_train = X_train_fcd
    X_train.images += X_train_nofcd.images
    X_val = PatchValDataset('./data/saved_patches/', True, 2*mods*nb_of_dims, h, w, loo_idx, DEFAULT_NB_OF_PATCHES, batch_size)

    train_dataloader = data.DataLoader(X_train, batch_size=1, shuffle=True,  num_workers=NUM_WORKERS,
                                       pin_memory=True, drop_last=False, collate_fn=dummy_collate)
    val_dataloader = data.DataLoader(X_val, batch_size=1, shuffle=True,  num_workers=NUM_WORKERS,
                                     pin_memory=True, drop_last=False, collate_fn=dummy_collate)

    ae = PatchAutoEncoder(h, w, mods, nb_of_dims, latent_dim, p).cuda()
    if parallel:
        ae = nn.DataParallel(ae)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ae.eval()
    overall_val_loss = 0
    for batch in val_dataloader:
        x = batch[0].cuda()
        x_hat = ae(x)
        loss = criterion(x_hat, x)
        overall_val_loss += loss.item()
    overall_val_loss /= len(val_dataloader)
    wandb.log({
        f'input-val-ae-images-{loo_idx}': [wandb.Image(x[0][i].detach().cpu().numpy(), caption=CAPTION_DICT[i])
                                for i in range(2 * mods * nb_of_dims)],
        f'output-val-ae-images-{loo_idx}': [wandb.Image(x_hat[0][i].detach().cpu().numpy(), caption=CAPTION_DICT[i])
                                 for i in range(2 * mods * nb_of_dims)]
    }, commit=False)

    overall_train_loss = 0
    for batch in train_dataloader:
        x = batch[0].cuda()
        x_hat = ae(x)
        loss = criterion(x_hat, x)
        overall_train_loss += loss.item()
    overall_train_loss /= len(train_dataloader)
    wandb.log({
        f'input-train-ae-images-{loo_idx}': [wandb.Image(x[0][i].detach().cpu().numpy(), caption=CAPTION_DICT[i])
                                  for i in range(2 * mods * nb_of_dims)],
        f'output-train-ae-images-{loo_idx}': [wandb.Image(x_hat[0][i].detach().cpu().numpy(), caption=CAPTION_DICT[i])
                                   for i in range(2 * mods * nb_of_dims)]
    }, commit=False)
    wandb.log({f'val-ae-{loo_idx}': overall_val_loss, f'train-ae-{loo_idx}': overall_train_loss})

    for epoch in range(n_epochs):
        ae.train()
        overall_train_loss = 0
        for batch in train_dataloader:
            x = batch[0].cuda()
            x_hat = ae(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            overall_train_loss += loss.item()
        overall_train_loss /= len(train_dataloader)
        wandb.log({
            f'input-train-ae-images-{loo_idx}': [wandb.Image(x[0][i].detach().cpu().numpy(), caption=CAPTION_DICT[i])
                                      for i in range(2 * mods * nb_of_dims)],
            f'output-train-ae-images-{loo_idx}': [wandb.Image(x_hat[0][i].detach().cpu().numpy(), caption=CAPTION_DICT[i])
                                       for i in range(2 * mods * nb_of_dims)]
        }, commit=False)

        ae.eval()
        overall_val_loss = 0
        for batch in val_dataloader:
            x = batch[0].cuda()
            x_hat = ae(x)
            loss = criterion(x_hat, x)
            overall_val_loss += loss.item()
        overall_val_loss /= len(val_dataloader)
        wandb.log({
            f'input-val-ae-images-{loo_idx}': [wandb.Image(x[0][i].detach().cpu().numpy(), caption=CAPTION_DICT[i])
                                    for i in range(2 * mods * nb_of_dims)],
            f'output-val-ae-images-{loo_idx}': [wandb.Image(x_hat[0][i].detach().cpu().numpy(), caption=CAPTION_DICT[i])
                                     for i in range(2 * mods * nb_of_dims)]
        }, commit=False)

        wandb.log({f'val-ae-{loo_idx}': overall_val_loss, f'train-ae-{loo_idx}': overall_train_loss})


    if parallel:
        torch.save(ae.module.encoder.state_dict(),
                   f'./checkpoints/{experiment_name}/encoder_{str(loo_idx).zfill(2)}.pth')
    else:
        torch.save(ae.encoder.state_dict(),
                   f'./checkpoints/{experiment_name}/encoder_{str(loo_idx).zfill(2)}.pth')


def train_model(mods, use_ae,
                h, w, use_coronal, use_sagital, use_controls, latent_dim, batch_size,
                lr, weight_decay, weight_of_class, n_epochs, n_epochs_ae, p, save_masks, parallel,
                experiment_name, temporal_division, seed):

    # set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def launch():
        train_loss_history = []
        val_loss_range = [0]
        val_loss_history = []
        model.eval()

        if use_ae:
            if parallel:
                model.module.encoder.load_state_dict(
                    torch.load(f'./checkpoints/{experiment_name}/encoder_{str(idx).zfill(2)}.pth')
                )
                for param in model.module.encoder.parameters():
                    param.requires_grad = False
            else:
                model.encoder.load_state_dict(
                    torch.load(f'./checkpoints/{experiment_name}/encoder_{str(idx).zfill(2)}.pth')
                )
                for param in model.encoder.parameters():
                    param.requires_grad = False

        overall_val_loss = 0
        for i, batch in enumerate(val_dataloader):
            X_batch, y_batch = batch[0].cuda(), batch[1].cuda()
            logits = model(X_batch)
            loss = criterion(logits[:, 0], y_batch)
            overall_val_loss += loss.item()
        overall_val_loss = overall_val_loss/len(val_dataloader)

        overall_train_loss = 0
        for i, batch in enumerate(train_dataloader):
            X_batch, y_batch = batch[0].cuda(), batch[1].cuda()
            logits = model(X_batch)
            loss = criterion(logits[:, 0], y_batch)
            overall_train_loss += loss.item()
        overall_train_loss = overall_train_loss/len(train_dataloader)

        wandb.log({f'val-classification-{idx}': overall_val_loss, f'train-classification-{idx}': overall_train_loss})

        for epoch in range(n_epochs):
            if use_ae:
                if epoch == 1:
                    for param in model.encoder.parameters():
                        param.requires_grad = True

            model.train()
            overall_train_loss = 0
            for i, batch in enumerate(train_dataloader):
                X_batch, y_batch = batch[0].cuda(), batch[1].cuda()
                y_predicted = model(X_batch)
                loss = criterion(y_predicted[:, 0], y_batch)
                loss.backward()
                overall_train_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                train_loss_history.append(loss.item())

            model.eval()
            overall_loss = 0
            y_pred = []
            y_val = []
            for i, batch in enumerate(val_dataloader):
                X_batch, y_batch = batch[0].cuda(), batch[1].cuda()
                logits = model(X_batch)
                predicted_labels = torch.sigmoid(logits[:, 0])

                loss = criterion(logits[:, 0], y_batch)
                overall_loss += loss.item()
                y_pred += list(predicted_labels.detach().cpu().numpy())
                y_val += list(y_batch.detach().cpu().numpy())
            wandb.log(
                {f'val-classification-{idx}': overall_loss/len(val_dataloader),
                 f'train-classification-{idx}': overall_train_loss/len(train_dataloader)})

            y_val = np.array(y_val)
            y_pred = np.array(y_pred)

            val_loss_history.append(overall_loss/len(val_dataloader))
            val_loss_range.append(val_loss_range[-1]+len(train_dataloader))

            # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            # ax.semilogy(np.arange(len(train_loss_history)), train_loss_history, label='train loss')
            # ax.semilogy(val_loss_range, val_loss_history, 'r-*', label='val loss')
            # ax.set_title('Model loss history')
            # ax.legend()
            # fig.savefig(f'./plots/{experiment_name}/patchmodel_loss_{str(idx).zfill(2)}.png')
            # plt.close(fig)

        torch.save(model.state_dict(), f'./checkpoints/{experiment_name}/model_{str(idx).zfill(2)}.pth')

        top_k_score = calculate_top_k_metric(y_val, y_pred)
        top_k_scores.append(top_k_score)

        wandb.log({f'top_k_scores': top_k_score})

    nb_of_dims = 1 + 1 * int(use_coronal) + 1 * int(use_sagital)
    top_k_scores = []

    for idx in np.arange(NB_OF_FCD_SUBJECTS):
        print(f'Model training, doint subject: ', idx)
        if use_ae:
            train_ae(
                mods=mods,
                h=h,
                w=w,
                use_coronal=use_coronal,
                use_sagital=use_sagital,
                latent_dim=latent_dim,
                batch_size=batch_size,
                lr=lr,
                n_epochs=n_epochs_ae,
                p=p,
                loo_idx=idx,
                parallel=parallel,
                experiment_name=experiment_name
            )
        deleted_idxs = [idx]

        if use_ae:
            deleted_idxs += [i for i in range(NB_OF_FCD_SUBJECTS, NB_OF_FCD_SUBJECTS + NB_OF_NOFCD_SUBJECTS)]

        train_dataset = PatchTrainDataset('./data/saved_patches/', True, 2 * mods, h, w, batch_size, idx)
        val_dataset = PatchValDataset('./data/saved_patches/', True, 2 * mods, h, w, idx, DEFAULT_NB_OF_PATCHES,
                                      batch_size)

        train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS,
                                           pin_memory=True, drop_last=False, collate_fn=dummy_collate)
        val_dataloader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
                                         pin_memory=True, drop_last=False, collate_fn=dummy_collate)

        model = PatchModel(h, w, mods, nb_of_dims, latent_dim, p).cuda()

        if parallel:
            model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        weights = torch.FloatTensor([1., weight_of_class]).cuda()
        criterion = lambda output, target: weighted_binary_cross_entropy(output, target, weights=weights)

        launch()

        print('Top-k score: ', top_k_scores[-1])

        mask_generator = FCDMaskGenerator(
            h=h,
            w=w,
            mods=mods,
            nb_of_dims=nb_of_dims,
            latent_dim=latent_dim,
            use_coronal=use_coronal,
            use_sagital=use_sagital,
            p=p,
            experiment_name=experiment_name,
            parallel=parallel,
            model_weights=f'./checkpoints/{experiment_name}/model_{str(idx).zfill(2)}.pth'
        )

        mask_generator.get_probability_masks(idx, save_masks=save_masks)
    top_k_scores = np.array(top_k_scores)
    wandb.finish()
    return top_k_scores

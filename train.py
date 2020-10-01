from models import PatchAutoEncoder, PatchModel
from mask_generator import FCDMaskGenerator
from utils import *

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt

import os

from nilearn import plotting
import nibabel as nib
from scipy.signal import convolve

from pathlib import Path
from torchio.transforms import HistogramStandardization
import torchio
from torchio.transforms import ZNormalization

import random

def train_ae(list_of_all_tensors, list_of_all_labels, 
             h, w, use_coronal, use_sagital, latend_dim, batch_size, 
             lr, n_epochs, p, loo_idx, parallel, experiment_name):
    DEFAULT_NB_OF_PATCHES = list_of_all_tensors[-1].shape[0]
    nb_of_dims = 1 + 1*int(use_coronal) + 1*int(use_sagital)

    loo_list_of_tensors = np.delete(list_of_all_tensors, loo_idx).copy()
    loo_list_of_labels = np.delete(list_of_all_labels, loo_idx).copy()

    X_train_np = np.concatenate(loo_list_of_tensors).copy()
    y_train_np = np.concatenate(loo_list_of_labels).copy()

    X_val_np = list_of_all_tensors[loo_idx].copy()[:DEFAULT_NB_OF_PATCHES]
    y_val_np = list_of_all_labels[loo_idx].copy()[:DEFAULT_NB_OF_PATCHES]

    X_train = data.TensorDataset(torch.FloatTensor(X_train_np))
    X_val = data.TensorDataset(torch.FloatTensor(X_val_np))

    train_dataloader = data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(X_val, batch_size=batch_size)

    ae = PatchAutoEncoder(h, w, nb_of_dims, latend_dim, p).cuda()
    if parallel:
        ae = nn.DataParallel(ae)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loss_history = []
    val_loss_history = []

    ae.eval()
    overall_val_loss = 0
    for batch in val_dataloader:
        x = batch[0].cuda()
        x_hat = ae(x)
        loss = criterion(x_hat, x)
        overall_val_loss += loss.item()
    overall_val_loss /= len(val_dataloader)
    val_loss_history.append(overall_val_loss)

    for epoch in range(n_epochs):
        ae.train()
        for batch in train_dataloader:
            x = batch[0].cuda()
            x_hat = ae(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_history.append(loss.item())

        ae.eval()
        overall_val_loss = 0
        for batch in val_dataloader:
            x = batch[0].cuda()
            x_hat = ae(x)
            loss = criterion(x_hat, x)
            overall_val_loss += loss.item()
        overall_val_loss /= len(val_dataloader)
        val_loss_history.append(overall_val_loss)


        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        ax.semilogy(np.arange(len(train_loss_history)), train_loss_history, label='train loss')
        ax.semilogy(np.arange(0, len(train_loss_history)+1, len(train_dataloader)), val_loss_history, 'r-*', label='val loss')
        ax.set_title('Autoencoder loss history')
        ax.legend()
        fig.savefig(f'./plots/{experiment_name}/ae_loss_{str(loo_idx).zfill(2)}.png')
        plt.close(fig)

    if parallel:
        torch.save(ae.module.encoder.state_dict(), 
                   f'./checkpoints/{experiment_name}/encoder_{str(loo_idx).zfill(2)}.pth')       
    else:
        torch.save(ae.encoder.state_dict(), 
                   f'./checkpoints/{experiment_name}/encoder_{str(loo_idx).zfill(2)}.pth')   


def train_model(list_of_all_tensors, list_of_all_labels, use_ae, 
                h, w, use_coronal, use_sagital, use_controls, latend_dim, batch_size, 
                lr, weight_decay, weight_of_class, n_epochs, n_epochs_ae, p, save_masks, parallel, 
                experiment_name, temporal_division, seed):
    
    ### set seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ###

    def launch():
        train_loss_history = []
        val_loss_range = [0]
        val_loss_history = []
        model.eval()
        overall_loss = 0
        for i, batch in enumerate(val_dataloader):
            X_batch, y_batch = batch[0].cuda(), batch[1].cuda()
            logits = model(X_batch)
            loss = criterion(logits[:, 0], y_batch)
            overall_loss += loss.item()
        val_loss_history.append(overall_loss/len(val_dataloader))

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



        for epoch in range(n_epochs):
            if use_ae:
                if epoch == 2:
                    for param in model.parameters():
                        param.requires_grad = True
                    for g in optimizer.param_groups:
                        g['lr'] = lr/3

            model.train()
            for i, batch in enumerate(train_dataloader):
                X_batch, y_batch = batch[0].cuda(), batch[1].cuda()
                y_predicted = model(X_batch)
                #print(y_predicted[:, 0].size())
                #print(y_batch.size())
                loss = criterion(y_predicted[:, 0], y_batch)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                train_loss_history.append(loss.item())

            model.eval()
            correct = 0
            overall_loss = 0
            y_pred = []
            for i, batch in enumerate(val_dataloader):
                X_batch, y_batch = batch[0].cuda(), batch[1].cuda()
                logits = model(X_batch)
                predicted_labels = torch.sigmoid(logits[:, 0])

                loss = criterion(logits[:, 0], y_batch)
                overall_loss += loss.item()
                y_pred += list(predicted_labels.detach().cpu().numpy())
                
            y_pred = np.array(y_pred)
                

            val_loss_history.append(overall_loss/len(val_dataloader))
            val_loss_range.append(val_loss_range[-1]+len(train_dataloader))


            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.semilogy(np.arange(len(train_loss_history)), train_loss_history, label='train loss')
            ax.semilogy(val_loss_range, val_loss_history, 'r-*', label='val loss')
            ax.set_title('Model loss history')
            ax.legend()
            fig.savefig(f'./plots/{experiment_name}/patchmodel_loss_{str(idx).zfill(2)}.png')
            plt.close(fig)
        
        torch.save(model.state_dict(), f'./checkpoints/{experiment_name}/model_{str(idx).zfill(2)}.pth')
        
        top_k_score = calculate_top_k_metric(y_val, y_pred)
        top_k_scores.append(top_k_score)
        
    DEFAULT_NB_OF_PATCHES = list_of_all_tensors[-1].shape[0]
    TEMPORAL_IDXS = [0, 3, 6, 9, 11, 12, 13]
    NON_TEMPORAL_IDXS = [1, 2, 4, 5, 7, 8, 10, 14]
    
    nb_of_dims = 1 + 1*int(use_coronal) + 1*int(use_sagital)
    top_k_scores = []
    
    for idx in np.arange(15): # there are 15 fcd subjects
        print(f'Model training, doint subject: ', idx)
        if use_ae:
            train_ae(
                list_of_all_tensors=list_of_all_tensors, 
                list_of_all_labels=list_of_all_labels, 
                h=h, 
                w=w, 
                use_coronal=use_coronal, 
                use_sagital=use_sagital, 
                latend_dim=latend_dim, 
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
            deleted_idxs += [i for i in range(15, 30)]
            
        if temporal_division:
            deleted_idxs += NON_TEMPORAL_IDXS if idx in TEMPORAL_IDXS else TEMPORAL_IDXS
        
        if use_controls and use_ae: 
            deleted_idxs += [i for i in range(30, 47)]
        
        if use_controls and not use_ae: 
            deleted_idxs += [i for i in range(15, 32)]
            
        loo_list_of_tensors = np.delete(list_of_all_tensors, deleted_idxs).copy()
        loo_list_of_labels = np.delete(list_of_all_labels, deleted_idxs).copy()

        
        X_train = np.concatenate(loo_list_of_tensors).copy()
        y_train = np.concatenate(loo_list_of_labels).copy()

        X_val = list_of_all_tensors[idx].copy()[:DEFAULT_NB_OF_PATCHES]
        y_val = list_of_all_labels[idx].copy()[:DEFAULT_NB_OF_PATCHES]

        train_dataset = data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
        val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

        model = PatchModel(h, w, nb_of_dims, latend_dim, p).cuda()
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
            nb_of_dims=nb_of_dims, 
            latent_dim=latend_dim, 
            use_coronal=use_coronal, 
            use_sagital=use_sagital,
            p=p,
            experiment_name=experiment_name,
            parallel=parallel,
            model_weights = f'./checkpoints/{experiment_name}/model_{str(idx).zfill(2)}.pth'
        )
        validation_brain_name = get_brain_name_by_idx(idx)
        validation_mask_name = get_mask_name_by_idx(idx)
        if save_masks:
            pred_mask_name = get_pred_mask_name_by_idx(experiment_name, idx)
        else:
            pred_mask_name = None
        side_mask_np, mid_mask_np, _ = mask_generator.detection_pipeline(validation_brain_name, validation_mask_name, pred_mask_name, probs=True)
    top_k_scores = np.array(top_k_scores)
    return top_k_scores

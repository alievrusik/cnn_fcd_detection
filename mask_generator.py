import numpy as np
import torch
from torch import nn
import torch.utils.data as data
import os
from scipy.signal import convolve
import nibabel as nib
import torchio

from patch_extraction import get_image_patches
from models import PatchModel

FCD_FOLDER = './data/fcd_brains/'
CONTROL_FOLDER = './data/control_brains/'


class FCDMaskGenerator:
    def __init__(self, h, w, mods, nb_of_dims, latent_dim, use_coronal, use_sagital, p, experiment_name, parallel,
                 model_weights='best_model.pth', thr=.5):
        self.model = PatchModel(h, w, mods, nb_of_dims, latent_dim, p).cuda()
        if parallel:
            self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(model_weights))
        self.model.eval()
        self.h = h
        self.w = w
        self.mods = mods
        self.best_t = thr
        self.nb_of_dims = nb_of_dims
        self.use_coronal = use_coronal
        self.use_sagital = use_sagital
        self.experiment_name = experiment_name
        gray_matter_template = nib.load('./data/MNI152_T1_0.5mm_brain_gray.nii.gz')
        self.gmpm = gray_matter_template.get_fdata() > 0

    def get_probability_masks(self, nb, save_masks=True):
        number = str(nb).zfill(2)
        list_of_patches_per_modality = []
        for m in range(1, self.mods + 1):
            X, _ = get_image_patches(
                input_img_name=os.path.join(FCD_FOLDER, f'fcd_{number}.{m}.nii.gz'),
                mod_nb=m,
                input_mask_name=None,
                gmpm=self.gmpm,
                h=self.h,
                w=self.w,
                augment=False,
                use_coronal=self.use_coronal,
                use_sagital=self.use_sagital
            )
            list_of_patches_per_modality += [X]

        inp_dataset = data.TensorDataset(torch.FloatTensor(np.concatenate(list_of_patches_per_modality, axis=1)))
        inp_dataloader = data.DataLoader(inp_dataset, batch_size=128, shuffle=False)

        y_pred = []
        for i, batch in enumerate(inp_dataloader):
            X_batch = batch[0].cuda()
            logits = self.model(X_batch)
            predicted_labels = torch.sigmoid(logits[:, 0])
            y_pred += list(predicted_labels.detach().cpu().numpy())
        y_pred = np.array(y_pred, dtype='float64')

        side_mask_np, mid_mask_np = get_image_patches(
            input_img_name=os.path.join(FCD_FOLDER, f'fcd_{number}.{m}.nii.gz'),
            mod_nb=m,
            input_mask_name=None,
            gmpm=self.gmpm, h=self.h, w=self.w, augment=False,
            use_coronal=self.use_coronal, use_sagital=self.use_sagital,
            record_results=True, pred_labels=y_pred
        )

        if save_masks:
            img = nib.load(os.path.join(FCD_FOLDER, f'fcd_{number}.{m}.nii.gz'))
            self.save_nii_mask(
                side_mask_np,
                img,
                f'./data/predicted_masks/{self.experiment_name}/side_predicted_mask_{nb}.nii.gz'
            )
            self.save_nii_mask(
                mid_mask_np,
                img,
                f'./data/predicted_masks/{self.experiment_name}/mid_predicted_mask_{nb}.nii.gz'
            )

    # bullshit is below

    def save_nii_mask(self, mask, img, save_mask_name):
        pred_mask_nii = nib.Nifti1Image(mask, img.affine)
        nib.save(pred_mask_nii, save_mask_name)

    def _infer_patch(self, patch):
        patch_torch = torch.FloatTensor(patch[None]).cuda()
        logits = self.model(patch_torch)
        probas = torch.sigmoid(logits)
        return probas

    def _get_predictions_per_batches(self, img):
        patch_map_tensor = np.zeros((4, self.gmpm.shape[1] // self.h, self.gmpm.shape[2]))
        target_np = img
        for i in range(self.gmpm.shape[2]):
            if i - self.w // 2 < 0:  # condition so coronal slices will fit
                continue

            if self.gmpm[:, :, i].sum() == 0.:
                continue

            for j in range(0, self.gmpm.shape[1], self.h):

                if (16 <= i <= 30 and j >= 118):
                    continue
                if j + self.h > self.gmpm.shape[1]:
                    continue

                if self.gmpm[:, j: j + self.h, i].sum() == 0.:  # just black stride is useless
                    continue

                rodon = self.gmpm[:, j: j + self.h, i].sum(1) > 0
                start_idx = rodon.argmax()
                mid_idx = self.gmpm.shape[0] // 2 - self.w - 2

                assert start_idx != 0
                # side patches
                if start_idx < mid_idx:
                    patch_1_axial = np.stack((
                        target_np[start_idx: start_idx + self.w, j: j + self.h, i],  # axial  slice
                        target_np[-start_idx - 1: -start_idx - self.w - 1: -1, j: j + self.h, i],
                        # mirrored axial slice
                    ))

                    patch_1_coronal = np.stack((
                        target_np[start_idx: start_idx + self.w, j + self.h // 2, i - self.h // 2: i + self.h // 2],
                        # coronal slice
                        target_np[-start_idx - 1: -start_idx - self.w - 1: -1, j + self.h // 2,
                        i - self.h // 2: i + self.h // 2]  # mirrored coronal slice
                    ))

                    patch_1_sagital = np.stack((
                        target_np[start_idx + self.w // 2, j: j + self.h, i - self.w // 2: i + self.w // 2],
                        # sagital  slice
                        target_np[-start_idx - self.w // 2, j: j + self.h, i - self.w // 2: i + self.w // 2],
                        # mirrored sagital slice
                    ))
                    patch_1_sagital = np.transpose(patch_1_sagital, axes=(0, 2, 1))

                    patch_1 = [patch_1_axial]

                    if self.use_coronal:
                        patch_1.append(patch_1_coronal)

                    if self.use_sagital:
                        patch_1.append(patch_1_sagital)

                    patch_1 = np.concatenate(tuple(patch_1))

                    patch_2_axial = np.stack((
                        target_np[-start_idx - self.w: -start_idx, j: j + self.h, i],
                        target_np[start_idx + self.w - 1: start_idx - 1: -1, j: j + self.h, i],
                    ))

                    patch_2_coronal = np.stack((
                        target_np[-start_idx - self.w: -start_idx, j + self.h // 2, i - self.h // 2: i + self.h // 2],
                        target_np[start_idx + self.w - 1: start_idx - 1: -1, j + self.h // 2,
                        i - self.h // 2: i + self.h // 2],
                    ))

                    patch_2_sagital = np.stack((
                        target_np[-start_idx - self.w // 2, j: j + self.h, i - self.w // 2: i + self.w // 2],
                        target_np[start_idx + self.w // 2, j: j + self.h, i - self.w // 2: i + self.w // 2],
                    ))
                    patch_2_sagital = np.transpose(patch_2_sagital, axes=(0, 2, 1))

                    patch_2 = [patch_2_axial]

                    if self.use_coronal:
                        patch_2.append(patch_2_coronal)

                    if self.use_sagital:
                        patch_2.append(patch_2_sagital)

                    patch_2 = np.concatenate(tuple(patch_2))

                    if start_idx < mid_idx:
                        patch_map_tensor[0, j // self.h, i] = self._infer_patch(patch_1)
                        patch_map_tensor[3, j // self.h, i] = self._infer_patch(patch_2)

                if not (i <= 44 and j >= 118):
                    # middle patches
                    patch_3_axial = np.stack((
                        target_np[mid_idx: mid_idx + self.w, j: j + self.h, i],
                        target_np[mid_idx + 2 * self.w - 1: mid_idx + self.w - 1: -1, j: j + self.h, i]
                    ))

                    patch_3_coronal = np.stack((
                        target_np[mid_idx: mid_idx + self.w, j + self.h // 2, i - self.h // 2: i + self.h // 2],
                        target_np[mid_idx + 2 * self.w - 1: mid_idx + self.w - 1: -1, j + self.h // 2,
                        i - self.h // 2: i + self.h // 2]
                    ))

                    patch_3_sagital = np.stack((
                        target_np[mid_idx + self.w // 2, j: j + self.h, i - self.w // 2: i + self.w // 2],
                        target_np[mid_idx + self.w + self.w // 2, j: j + self.h, i - self.w // 2: i + self.w // 2],
                    ))
                    patch_3_sagital = np.transpose(patch_3_sagital, axes=(0, 2, 1))

                    patch_3 = [patch_3_axial]

                    if self.use_coronal:
                        patch_3.append(patch_3_coronal)

                    if self.use_sagital:
                        patch_3.append(patch_3_sagital)

                    patch_3 = np.concatenate(tuple(patch_3))

                    patch_4_axial = np.stack((
                        target_np[mid_idx + self.w: mid_idx + 2 * self.w, j: j + self.h, i],
                        target_np[mid_idx + self.w - 1: mid_idx - 1: -1, j: j + self.h, i],
                    ))

                    patch_4_coronal = np.stack((
                        target_np[mid_idx + self.w: mid_idx + 2 * self.w, j + self.h // 2,
                        i - self.h // 2: i + self.h // 2],
                        target_np[mid_idx + self.w - 1: mid_idx - 1: -1, j + self.h // 2,
                        i - self.h // 2: i + self.h // 2]
                    ))

                    patch_4_sagital = np.stack((
                        target_np[mid_idx + self.w + self.w // 2, j: j + self.h, i - self.w // 2: i + self.w // 2],
                        target_np[mid_idx + self.w // 2, j: j + self.h, i - self.w // 2: i + self.w // 2],
                    ))
                    patch_4_sagital = np.transpose(patch_4_sagital, axes=(0, 2, 1))

                    patch_4 = [patch_4_axial]

                    if self.use_coronal:
                        patch_4.append(patch_4_coronal)

                    if self.use_sagital:
                        patch_4.append(patch_4_sagital)

                    patch_4 = np.concatenate(tuple(patch_4))

                    patch_map_tensor[1, j // self.h, i] = self._infer_patch(patch_3)
                    patch_map_tensor[2, j // self.h, i] = self._infer_patch(patch_4)

        return patch_map_tensor

    def _postprocess(self, patch_map_tensor):
        count_neighbs = .25 * np.array([[
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]])
        res = convolve(patch_map_tensor, count_neighbs, mode='same')
        change_to_pos = np.isclose(res, 1.)
        change_to_neg = np.isclose(res, 0.)
        patch_map_tensor[change_to_pos] = 1
        patch_map_tensor[change_to_neg] = 0
        return patch_map_tensor

    def _prob_masking(self, img, patch_map_tensor):
        side_mask_np, mid_mask_np = np.zeros_like(img), np.zeros_like(img)
        for i in range(self.gmpm.shape[2]):
            if i - self.w // 2 < 0:  # condition so coronal slices will fit
                continue

            if self.gmpm[:, :, i].sum() == 0.:
                continue

            for j in range(0, self.gmpm.shape[1], self.h):
                if (16 <= i <= 30 and j >= 118):
                    continue

                if j + self.h > self.gmpm.shape[1]:
                    continue

                if self.gmpm[:, j: j + self.h, i].sum() == 0.:  # just black stride is useless
                    continue

                rodon = self.gmpm[:, j: j + self.h, i].sum(1) > 0
                start_idx = rodon.argmax()
                mid_idx = self.gmpm.shape[0] // 2 - self.w - 2

                if start_idx < mid_idx:
                    side_mask_np[start_idx: start_idx + self.w, j:j + self.h, i] = patch_map_tensor[0, j // self.h, i]
                    side_mask_np[-start_idx - self.w: -start_idx, j:j + self.h, i] = patch_map_tensor[3, j // self.h, i]
                if not (i <= 44 and j >= 118):
                    mid_mask_np[mid_idx: mid_idx + self.w, j:j + self.h, i] = patch_map_tensor[1, j // self.h, i]
                    mid_mask_np[mid_idx + self.w: mid_idx + 2 * self.w, j:j + self.h, i] = patch_map_tensor[
                        2, j // self.h, i]
        return side_mask_np, mid_mask_np

    def get_mask(self, img):
        patch_map_tensor = self._get_predictions_per_batches(img)
        patch_map_tensor = patch_map_tensor > self.best_t
        patch_map_tensor2 = self._postprocess(patch_map_tensor)
        side_mask_np, mid_mask_np = self._prob_masking(img, patch_map_tensor2)
        return mask

    def get_prob_masks(self, img):
        patch_map_tensor = self._get_predictions_per_batches(img)
        patch_map_tensor[patch_map_tensor < 1e-4] = 0.
        side_mask_np, mid_mask_np = self._prob_masking(img, patch_map_tensor)
        return side_mask_np, mid_mask_np

    def get_iou(self, side_mask_np, mid_mask_np, true_mask):
        assert pred_mask.shape == true_mask.shape, 'Wrong shape of masks'
        pred_mask = np.logical_or(side_mask_np, mid_mask_np)
        intersection = np.logical_and(pred_mask, true_mask)
        union = np.logical_or(pred_mask, true_mask)
        return intersection.sum() / union.sum()

    def save_nii_mask(self, mask, img, save_mask_name):
        pred_mask_nii = nib.Nifti1Image(mask, img.affine)
        nib.save(pred_mask_nii, save_mask_name)

    def detection_pipeline(self, input_img_name, input_mask_name=None, save_mask_name='pred_mask.nii.gz', probs=False):
        img = nib.load(input_img_name)
        subject_dict = {
            'mri': torchio.Image(input_img_name, torchio.INTENSITY),
        }
        subject = torchio.Subject(subject_dict)
        zimage = self.transform(subject)
        img_np = zimage['mri'].data[0].numpy()

        if not probs:
            side_mask_np, mid_mask_np = self.get_mask(img_np)
            if input_mask_name is not None:
                true_mask = nib.load(input_mask_name)
                true_mask_np = true_mask.get_fdata() > 0
                iou = self.get_iou(side_mask_np, mid_mask_np, true_mask_np)
                print('Intersection over union = {:.5f}'.format(iou))
            else:
                iou = None

            self.save_nii_mask(pred_mask_np, img, save_mask_name)
            return side_mask_np, mid_mask_np, iou
        else:
            side_mask_np, mid_mask_np = self.get_prob_masks(img_np)
            if save_mask_name is not None:
                self.save_nii_mask(side_mask_np, img, os.path.join(f'./data/predicted_masks/{self.experiment_name}',
                                                                   'side_' + os.path.basename(save_mask_name)))
                self.save_nii_mask(mid_mask_np, img, os.path.join(f'./data/predicted_masks/{self.experiment_name}',
                                                                  'mid_' + os.path.basename(save_mask_name)))
            return side_mask_np, mid_mask_np, None

    def classification_pipeline(self, input_img_name, input_mask_name=None):

        pred_mask_np, iou = self.detection_pipeline(input_img_name, input_mask_name)

        if input_mask_name is not None:
            if iou > 0:
                have_fcd = True  # true positive
            else:
                have_fcd = False  # false negative
        else:
            if pred_mask_np.sum() < self.minimum_mask_volume:
                have_fcd = False  # true negative
            else:
                have_fcd = True  # false negative
        return have_fcd

import nibabel as nib
import numpy as np
import os
from pathlib import Path
import torchio
from torchio.transforms import HistogramStandardization
from torchio.transforms import ZNormalization
from numba import njit, jit
import pickle

FCD_FOLDER = './data/fcd_brains/'
CONTROL_FOLDER = './data/control_brains/'
MASK_FOLDER = './data/masks/'
NB_OF_FCD_SUBJECTS = 26
NB_OF_NOFCD_SUBJECTS = 15
NB_OF_CONTROL_SUBJECTS = 100500


@jit(nopython=True)
def get_patches_and_labels(target_np: np.array, gmpm: np.array, mask_np: np.array, use_coronal=False, use_sagital=False,
                           h=16, w=32, coef=.2, max_counter=25000, augment=True,
                           record_results=False, pred_labels=None):
    nb_of_dims = 2 + 2 * int(use_coronal) + 2 * int(use_sagital)
    all_patches = np.zeros((max_counter, nb_of_dims, w, h))
    all_labels = np.zeros((max_counter,))
    counter = 0
    if pred_labels is None:
        pred_labels = np.zeros(max_counter)
    side_mask_np, mid_mask_np = np.zeros(target_np.shape), np.zeros(target_np.shape)
    rep = (h - 1) * augment + 1
    for k in range(0, rep):  # if augment, then k in [0..h-1], else k in [0..0]
        for i in range(gmpm.shape[2]):
            # if i - w // 2 <= 0:  # condition so sagital slices will fit
            #     continue

            if gmpm[:, :, i].sum() == 0.:
                continue

            for j in range(0, gmpm.shape[1], h):
                if j + k + h > gmpm.shape[1]:
                    continue

                # if 16 <= i <= 30 and j + k >= 118:
                #     continue

                if gmpm[:, j + k: j + k + h, i].sum() == 0.:  # just black stride is useless
                    continue

                rodon = gmpm[:, j + k: j + k + h, i].sum(1) > 0
                start_idx = rodon.argmax()
                mid_idx = gmpm.shape[0] // 2 - w

                assert start_idx != 0

                # side patches
                if start_idx < mid_idx:
                    patch_1_axial = np.stack((
                        target_np[start_idx: start_idx + w, j + k: j + k + h, i],
                        # axial slice
                        target_np[-start_idx - 1: -start_idx - w - 1: -1, j + k: j + k + h, i],
                        # mirrored axial slice
                    ))

                    patch_1_coronal = np.stack((
                        target_np[start_idx: start_idx + w, j + k + h // 2, i - h // 2: i + h // 2],
                        # coronal slice
                        target_np[-start_idx - 1: -start_idx - w - 1: -1, j + k + h // 2, i - h // 2: i + h // 2],
                        # mirrored coronal slice
                    ))

                    patch_1_sagital = np.stack((
                        target_np[start_idx + w // 2, j + k: j + k + h, i - w // 2: i + w // 2],
                        # sagital slice
                        target_np[-start_idx - w // 2, j + k: j + k + h, i - w // 2: i + w // 2],
                        # mirrored sagital slice
                    ))
                    patch_1_sagital = np.transpose(patch_1_sagital, axes=(0, 2, 1))

                    patch_1 = [patch_1_axial]
                    label_1 = mask_np[start_idx: start_idx + w, j + k: j + k + h, i].sum()

                    if use_coronal:
                        patch_1.append(patch_1_coronal)
                        label_1 += mask_np[start_idx: start_idx + w, j + k + h // 2, i - h // 2: i + h // 2].sum()

                    if use_sagital:
                        patch_1.append(patch_1_sagital)
                        label_1 += mask_np[start_idx + w // 2, j + k: j + k + h, i - w // 2: i + w // 2].sum()

                    # patch_1 = np.concatenate(tuple(patch_1))
                    patch_1 = patch_1_axial

                    patch_2_axial = np.stack((
                        target_np[-start_idx - w: -start_idx, j + k: j + k + h, i],
                        target_np[start_idx + w - 1: start_idx - 1: -1, j + k: j + k + h, i],
                    ))

                    patch_2_coronal = np.stack((
                        target_np[-start_idx - w: -start_idx, j + k + h // 2, i - h // 2: i + h // 2],
                        target_np[start_idx + w - 1: start_idx - 1: -1, j + k + h // 2, i - h // 2: i + h // 2],
                    ))

                    patch_2_sagital = np.stack((
                        target_np[-start_idx - w // 2, j + k: j + k + h, i - w // 2: i + w // 2],
                        target_np[start_idx + w // 2, j + k: j + k + h, i - w // 2: i + w // 2],
                    ))
                    patch_2_sagital = np.transpose(patch_2_sagital, axes=(0, 2, 1))

                    patch_2 = [patch_2_axial]
                    label_2 = mask_np[-start_idx - w: -start_idx, j + k: j + k + h, i].sum()

                    if use_coronal:
                        patch_2.append(patch_2_coronal)
                        label_2 += mask_np[-start_idx - w: -start_idx, j + k + h // 2, i - h // 2: i + h // 2].sum()

                    if use_sagital:
                        patch_2.append(patch_2_sagital)
                        label_2 += mask_np[-start_idx - w // 2, j + k: j + k + h, i - w // 2: i + w // 2].sum()

                    # patch_2 = np.concatenate(tuple(patch_2))
                    patch_2 = patch_2_axial

                    if k == 0 or label_1:
                        all_patches[counter] = patch_1
                        all_labels[counter] = label_1
                        if record_results and k == 0:
                            side_mask_np[start_idx: start_idx + w, j + k: j + k + h, i] = pred_labels[counter]
                        counter += 1

                    if k == 0 or label_2:
                        all_patches[counter] = patch_2
                        all_labels[counter] = label_2
                        if record_results and k == 0:

                            side_mask_np[-start_idx - w: -start_idx, j + k: j + k + h, i] = pred_labels[counter]
                        counter += 1

                # middle patches
                # if not (i <= 44 and j + k >= 118):

                patch_3_axial = np.stack((
                    target_np[mid_idx: mid_idx + w, j + k: j + k + h, i],
                    target_np[mid_idx + 2 * w - 1: mid_idx + w - 1: -1, j + k: j + k + h, i]
                ))

                patch_3_coronal = np.stack((
                    target_np[mid_idx: mid_idx + w, j + k + h // 2, i - h // 2: i + h // 2],
                    target_np[mid_idx + 2 * w - 1: mid_idx + w - 1: -1, j + k + h // 2, i - h // 2: i + h // 2]
                ))

                patch_3_sagital = np.stack((
                    target_np[mid_idx + w // 2, j + k: j + k + h, i - w // 2: i + w // 2],
                    target_np[mid_idx + w + w // 2, j + k: j + k + h, i - w // 2: i + w // 2],
                ))
                patch_3_sagital = np.transpose(patch_3_sagital, axes=(0, 2, 1))

                patch_3 = [patch_3_axial]
                label_3 = mask_np[mid_idx: mid_idx + w, j + k: j + k + h, i].sum()

                if use_coronal:
                    patch_3.append(patch_3_coronal)
                    label_3 += mask_np[mid_idx: mid_idx + w, j + k + h // 2, i - h // 2: i + h // 2].sum()

                if use_sagital:
                    patch_3.append(patch_3_sagital)
                    label_3 += mask_np[mid_idx + w // 2, j + k: j + k + h, i - w // 2: i + w // 2].sum()

                # patch_3 = np.concatenate(tuple(patch_3))
                patch_3 = patch_3_axial

                patch_4_axial = np.stack((
                    target_np[mid_idx + w: mid_idx + 2 * w, j + k: j + k + h, i],
                    target_np[mid_idx + w - 1: mid_idx - 1: -1, j + k: j + k + h, i],
                ))

                patch_4_coronal = np.stack((
                    target_np[mid_idx + w: mid_idx + 2 * w, j + k + h // 2, i - h // 2: i + h // 2],
                    target_np[mid_idx + w - 1: mid_idx - 1: -1, j + k + h // 2, i - h // 2: i + h // 2]
                ))

                patch_4_sagital = np.stack((
                    target_np[mid_idx + w + w // 2, j + k: j + k + h, i - w // 2: i + w // 2],
                    target_np[mid_idx + w // 2, j + k: j + k + h, i - w // 2: i + w // 2],
                ))
                patch_4_sagital = np.transpose(patch_4_sagital, axes=(0, 2, 1))

                patch_4 = [patch_4_axial]
                label_4 = mask_np[mid_idx + w: mid_idx + 2 * w, j + k: j + k + h, i].sum()

                if use_coronal:
                    patch_4.append(patch_4_coronal)
                    label_4 += mask_np[mid_idx + w: mid_idx + 2 * w, j + k + h // 2, i - h // 2: i + h // 2].sum()

                if use_sagital:
                    patch_4.append(patch_4_sagital)
                    label_4 += mask_np[mid_idx + w + w // 2, j + k: j + k + h, i - w // 2: i + w // 2].sum()

                # patch_4 = np.concatenate(tuple(patch_4))
                patch_4 = patch_4_axial

                if k == 0 or label_3:
                    all_patches[counter] = patch_3
                    all_labels[counter] = label_3
                    if record_results and k == 0:
                        mid_mask_np[mid_idx: mid_idx + w, j + k: j + k + h, i] += pred_labels[counter]
                    counter += 1

                if k == 0 or label_4:
                    all_patches[counter] = patch_4
                    all_labels[counter] = label_4
                    if record_results and k == 0:
                        mid_mask_np[mid_idx + w: mid_idx + 2 * w, j + k: j + k + h, i] = pred_labels[counter]
                    counter += 1

    all_patches, all_labels = all_patches[:counter], all_labels[:counter]
    if all_labels.max() != 0.:
        all_labels = all_labels / all_labels.max()
        all_labels = all_labels ** coef

    return all_patches, all_labels, side_mask_np, mid_mask_np


def get_image_patches(input_img_name, mod_nb, gmpm=None, use_coronal=False,
                      use_sagital=False, input_mask_name=None, augment=True, h=16, w=32, coef=.2,
                      record_results=False, pred_labels=None):
    subject_dict = {
        'mri': torchio.Image(input_img_name, torchio.INTENSITY),
    }

    # torchio normalization
    t1_landmarks = Path(f'./data/t1_landmarks_{mod_nb}.npy')
    landmarks_dict = {'mri': t1_landmarks}
    histogram_transform = HistogramStandardization(landmarks_dict)
    znorm_transform = ZNormalization(masking_method=ZNormalization.mean)
    transform = torchio.transforms.Compose([histogram_transform, znorm_transform])
    subject = torchio.Subject(subject_dict)
    zimage = transform(subject)
    target_np = zimage['mri'].data[0].numpy()

    if input_mask_name is not None:
        mask = nib.load(input_mask_name)
        mask_np = (mask.get_fdata() > 0).astype('float')
    else:
        mask_np = np.zeros_like(target_np)

    all_patches, all_labels, side_mask_np, mid_mask_np = get_patches_and_labels(target_np, gmpm, mask_np,
                                                                                use_coronal=use_coronal,
                                                                                use_sagital=use_sagital, h=h, w=w,
                                                                                coef=coef, augment=augment,
                                                                                record_results=record_results,
                                                                                pred_labels=pred_labels)
    if not record_results:
        return all_patches, all_labels
    else:
        return side_mask_np, mid_mask_np


def get_patch_list(use_controls: bool, use_fcd: bool, use_coronal: bool, use_sagital: bool,
                   augment=True, hard_labeling=False, h=16, w=32, coef=.2, mods=1, batch_size=512):
    gray_matter_template = nib.load('./data/MNI152_T1_0.5mm_brain_gray.nii.gz')
    gmpm = gray_matter_template.get_fdata()
    gmpm = (gmpm > 0).astype('float')

    # list_of_tensors = []
    # list_of_labels = []

    # fcd brains
    for i in range(NB_OF_FCD_SUBJECTS):
        print('Patch extraction: fcd', i)

        number = str(i).zfill(2)
        # if os.path.exists(f'data/saved_x_fcd_{number}.npy'):
        #     list_of_patches_per_modality = np.load(f'data/saved_x_fcd_{number}.npy')
        #     y = np.load(f'data/saved_y_fcd_{number}.npy')
        #     list_of_tensors.append(list_of_patches_per_modality)
        #     list_of_labels.append(y)
        #     continue
        if os.path.exists(f'data/saved_patches/fcd_{i}_patches'):
            continue

        input_mask_name = f'mask_fcd_{number}.1.nii.gz'
        list_of_patches_per_modality = []
        for m in range(1, mods + 1):
            X, y = get_image_patches(input_img_name=os.path.join(FCD_FOLDER, f'fcd_{number}.{m}.nii.gz'),
                                     mod_nb=m,
                                     input_mask_name=os.path.join(MASK_FOLDER, input_mask_name),
                                     gmpm=gmpm, h=h, w=w, augment=augment, coef=coef,
                                     use_coronal=use_coronal, use_sagital=use_sagital)
            list_of_patches_per_modality += [X]
            # y is the same for all modalities
        if hard_labeling:
            y = y > 0.
        list_of_patches_per_modality = np.concatenate(list_of_patches_per_modality, axis=1)
        # np.save(f'data/saved_x_fcd_{number}.npy', list_of_patches_per_modality)
        # np.save(f'data/saved_y_fcd_{number}.npy', y)
        os.makedirs(f'data/saved_patches/fcd_{i}_patches', exist_ok=True)
        for k in range(len(list_of_patches_per_modality)//batch_size):
            current_pair = np.concatenate([
                list_of_patches_per_modality[k*batch_size: (k+1)*batch_size].reshape(-1),
                y[k*batch_size: (k+1)*batch_size]
            ])
            np.save(f'data/saved_patches/fcd_{i}_patches/patch_{k}.npy', current_pair)
        # os.mknod(f'data/saved_patches/fcd_{i}_patches/.ready')
        # list_of_tensors.append(list_of_patches_per_modality)
        # list_of_labels.append(y)

    # nofcd brains
    if use_fcd:
        for i in range(NB_OF_NOFCD_SUBJECTS):
            print('Patch extraction: nofcd', i)

            number = str(i).zfill(2)
            # if os.path.exists(f'data/saved_x_nofcd_{number}.npy'):
            #     list_of_patches_per_modality = np.load(f'data/saved_x_nofcd_{number}.npy')
            #     y = np.load(f'data/saved_y_nofcd_{number}.npy')
            #     list_of_tensors.append(list_of_patches_per_modality)
            #     list_of_labels.append(y)
            #     continue
            if os.path.exists(f'data/saved_patches/nofcd_{i}_patches'):
                continue
            list_of_patches_per_modality = []
            for m in range(1, mods + 1):
                X, y = get_image_patches(input_img_name=os.path.join(FCD_FOLDER, f'nofcd_{number}.{m}.nii.gz'),
                                         mod_nb=m,
                                         input_mask_name=None,
                                         gmpm=gmpm, h=h, w=w, augment=augment, coef=coef,
                                         use_coronal=use_coronal, use_sagital=use_sagital)
                list_of_patches_per_modality += [X]
                # y is the same for all modalities
            if hard_labeling:
                y = y > 0.

            list_of_patches_per_modality = np.concatenate(list_of_patches_per_modality, axis=1)
            print(list_of_patches_per_modality.shape)
            # np.save(f'data/saved_x_nofcd_{number}.npy', list_of_patches_per_modality)
            # np.save(f'data/saved_y_nofcd_{number}.npy', y)
            os.makedirs(f'data/saved_patches/nofcd_{i}_patches', exist_ok=True)
            for k in range(len(list_of_patches_per_modality)//batch_size):
                current_pair = np.concatenate([
                    list_of_patches_per_modality[k * batch_size: (k + 1) * batch_size].reshape(-1),
                    y[k * batch_size: (k + 1) * batch_size]
                ])
                np.save(f'data/saved_patches/nofcd_{i}_patches/patch_{k}.npy', current_pair)
            # list_of_tensors.append(list_of_patches_per_modality)
            # list_of_labels.append(y)

    # don't use it, mate
    # if use_controls:
    #     for i in range(NB_OF_CONTROL_SUBJECTS):
    #         print('Patch extraction: controls', i)
    #
    #         number = str(i).zfill(2)
    #         list_of_patches_per_modality = []
    #         for m in range(1, mods + 1):
    #             X, y = get_image_patches(input_img_name=os.path.join(CONTROL_FOLDER, f'fcd_{number}.{m}.nii.gz'),
    #                                      mod_nb=m,
    #                                      input_mask_name=None,
    #                                      gmpm=gmpm, h=h, w=w, augment=augment, coef=coef,
    #                                      use_coronal=use_coronal, use_sagital=use_sagital)
    #             list_of_patches_per_modality += [X]
    #             # y is the same for all modalities
    #         if hard_labeling:
    #             y = y > 0.
    #
    #         # list_of_tensors.append(np.concatenate(list_of_patches_per_modality, axis=1))
    #         # list_of_labels.append(y)

    return None, None

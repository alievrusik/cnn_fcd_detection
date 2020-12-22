import nibabel as nib
import numpy as np
import os

from numba import njit

from pathlib import Path
import torchio
from torchio.transforms import HistogramStandardization
from torchio.transforms import ZNormalization


#@njit
def get_patches_and_labels(target_np: np.array, gmpm: np.array, mask_np: np.array, use_coronal=False, use_sagital=False, h=16, w=32, coef=.2, max_counter=10000, augment=True):
    nb_of_dims = 2 + 2*int(use_coronal) + 2*int(use_sagital)
    all_patches = np.zeros((max_counter, nb_of_dims, w, h))
    all_labels = np.zeros((max_counter))
    counter = 0
    for i in range(gmpm.shape[2]):
        if i - w // 2 < 0: #condition so coronal slices will fit
            continue

        if gmpm[:, :, i].sum() == 0.:
            continue

        for j in range(0, gmpm.shape[1], h):
            if (16 <= i <= 30 and j >= 118):
                continue
            if j + h > gmpm.shape[1]:
                continue

            if gmpm[:, j: j + h, i].sum() == 0.:   #just black stride is useless
                continue

            rodon = gmpm[:, j: j + h, i].sum(1) > 0
            start_idx = rodon.argmax()
            mid_idx = gmpm.shape[0] // 2 - w - 2

            assert start_idx != 0

            #side patches
            if start_idx < mid_idx:
                patch_1_axial = np.stack((
                    target_np[start_idx: start_idx + w, j: j + h, i], #axial  slice
                    target_np[-start_idx - 1: -start_idx - w - 1: -1, j: j + h, i], #mirrored axial slice
                ))

                patch_1_coronal = np.stack((
                    target_np[start_idx: start_idx + w, j + h//2, i - h//2: i + h//2], #coronal slice
                    target_np[-start_idx - 1: -start_idx - w - 1: -1, j + h//2, i - h//2: i + h//2], #mirrored coronal slice
                ))

                patch_1_sagital = np.stack((
                    target_np[start_idx + w//2, j: j + h, i - w//2: i + w//2], #sagital  slice
                    target_np[-start_idx - w//2, j: j + h, i - w//2: i + w//2], #mirrored sagital slice
                ))
                patch_1_sagital = np.transpose(patch_1_sagital, axes=(0, 2, 1))

                patch_1 = [patch_1_axial]
                label_1 = mask_np[start_idx: start_idx + w, j: j + h, i].sum()

                if use_coronal:
                    patch_1.append(patch_1_coronal)
                    label_1 += mask_np[start_idx: start_idx + w, j + h//2, i - h//2: i + h//2].sum()

                if use_sagital:
                    patch_1.append(patch_1_sagital)
                    label_1 += mask_np[start_idx + w//2, j: j + h, i - w//2: i + w//2].sum()

                patch_1 = np.concatenate(tuple(patch_1))


                patch_2_axial = np.stack((
                    target_np[-start_idx - w : -start_idx, j: j + h, i],
                    target_np[start_idx + w - 1: start_idx - 1: -1, j: j + h, i],
                ))

                patch_2_coronal = np.stack((
                    target_np[-start_idx - w : -start_idx, j + h//2, i - h//2: i + h//2],
                    target_np[start_idx + w - 1: start_idx - 1: -1, j + h//2, i - h//2: i + h//2],
                ))

                patch_2_sagital = np.stack((
                    target_np[-start_idx - w//2, j: j + h, i - w//2: i + w//2],
                    target_np[start_idx + w//2, j: j + h, i - w//2: i + w//2],
                ))
                patch_2_sagital = np.transpose(patch_2_sagital, axes=(0, 2, 1))


                patch_2 = [patch_2_axial]
                label_2 = mask_np[-start_idx - w : -start_idx, j: j + h, i].sum()

                if use_coronal:
                    patch_2.append(patch_2_coronal)
                    label_2 += mask_np[-start_idx - w : -start_idx, j + h//2, i - h//2: i + h//2].sum()

                if use_sagital:
                    patch_2.append(patch_2_sagital)
                    label_2 += mask_np[-start_idx - w//2, j: j + h, i - w//2: i + w//2].sum()

                patch_2 = np.concatenate(tuple(patch_2))

                all_patches[counter] = patch_1
                all_labels[counter] = label_1
                counter += 1

                all_patches[counter] = patch_2
                all_labels[counter] = label_2
                counter += 1

            if not (i <= 44 and j >= 118):
                #middle patches

                patch_3_axial = np.stack((
                    target_np[mid_idx: mid_idx + w, j: j + h, i],
                    target_np[mid_idx + 2*w - 1: mid_idx + w - 1: -1, j: j + h, i]
                ))

                patch_3_coronal = np.stack((
                    target_np[mid_idx: mid_idx + w, j + h//2, i - h//2: i + h//2],
                    target_np[mid_idx + 2*w - 1: mid_idx + w - 1: -1, j + h//2, i - h//2: i + h//2]
                ))

                patch_3_sagital = np.stack((
                    target_np[mid_idx + w//2, j: j + h, i - w//2: i + w//2],
                    target_np[mid_idx + w + w//2, j: j + h, i - w//2: i + w//2],
                ))
                patch_3_sagital = np.transpose(patch_3_sagital, axes=(0, 2, 1))



                patch_3 = [patch_3_axial]
                label_3 = mask_np[mid_idx: mid_idx + w, j: j + h, i].sum()

                if use_coronal:
                    patch_3.append(patch_3_coronal)
                    label_3 += mask_np[mid_idx: mid_idx + w, j + h//2, i - h//2: i + h//2].sum()

                if use_sagital:
                    patch_3.append(patch_3_sagital)
                    label_3 += mask_np[mid_idx + w//2, j: j + h, i - w//2: i + w//2].sum()

                #print(patch_3_axial.shape)
                #print(patch_3_coronal.shape)
                #print(patch_3_sagital.shape)
                patch_3 = np.concatenate(tuple(patch_3))

                ######### 4

                patch_4_axial = np.stack((
                    target_np[mid_idx + w: mid_idx + 2*w, j: j + h, i],
                    target_np[mid_idx + w - 1: mid_idx - 1 : -1, j: j + h, i],
                ))

                patch_4_coronal = np.stack((
                    target_np[mid_idx + w: mid_idx + 2*w,  j + h//2, i - h//2: i + h//2],
                    target_np[mid_idx + w - 1: mid_idx - 1 : -1, j + h//2, i - h//2: i + h//2]
                ))

                patch_4_sagital = np.stack((
                    target_np[mid_idx + w + w//2, j: j + h, i - w//2: i + w//2],
                    target_np[mid_idx + w//2, j: j + h, i - w//2: i + w//2],
                ))
                patch_4_sagital = np.transpose(patch_4_sagital, axes=(0, 2, 1))



                patch_4 = [patch_4_axial]
                label_4 = mask_np[mid_idx + w: mid_idx + 2*w, j: j + h, i].sum()

                if use_coronal:
                    patch_4.append(patch_4_coronal)
                    label_4 += mask_np[mid_idx + w: mid_idx + 2*w,  j + h//2, i - h//2: i + h//2].sum()

                if use_sagital:
                    patch_4.append(patch_4_sagital)
                    label_4 += mask_np[mid_idx + w + w//2, j: j + h, i - w//2: i + w//2].sum()

                patch_4 = np.concatenate(tuple(patch_4))

                all_patches[counter] = patch_3
                all_labels[counter] = label_3
                counter += 1

                all_patches[counter] = patch_4
                all_labels[counter] = label_4
                counter += 1

    #oversampling
    if augment:
        for k in range(1, h):
            for i in range(gmpm.shape[2]):
                if i - w // 2 <= 0: #condition so sagital slices will fit
                    continue

                if gmpm[:,:,i].sum() == 0.:
                    continue

                for j in range(0, gmpm.shape[1], h):
                    if j + k + h > gmpm.shape[1]:
                        continue
                    if (16 <= i <= 30 and j + k >= 118):
                        continue

                    if gmpm[:, j + k: j + k + h, i].sum() == 0.:   #just black stride is useless
                        continue

                    rodon = gmpm[:, j + k: j + k + h, i].sum(1) > 0
                    start_idx = rodon.argmax()
                    mid_idx = gmpm.shape[0] // 2 - w - 2

                    assert start_idx != 0

                    #side patches
                    if start_idx < mid_idx:
                        patch_1_axial = np.stack((
                            target_np[start_idx: start_idx + w, j + k: j + k + h, i], #axial  slice
                            target_np[-start_idx - 1: -start_idx - w - 1: -1, j + k: j + k + h, i], #mirrored axial slice
                        ))

                        patch_1_coronal = np.stack((
                            target_np[start_idx: start_idx + w, j + k + h//2, i - h//2: i + h//2], #coronal slice
                            target_np[-start_idx - 1: -start_idx - w - 1: -1, j + k + h//2, i - h//2: i + h//2], #mirrored coronal
                        ))

                        patch_1_sagital = np.stack((
                            target_np[start_idx + w//2, j + k: j + k + h, i - w//2: i + w//2], #sagital  slice
                            target_np[-start_idx - w//2, j + k: j + k + h, i - w//2: i + w//2], #mirrored sagital slice
                        ))
                        patch_1_sagital = np.transpose(patch_1_sagital, axes=(0, 2, 1))


                        patch_1 = [patch_1_axial]
                        label_1 = mask_np[start_idx: start_idx + w, j + k: j + k + h, i].sum()

                        if use_coronal:
                            patch_1.append(patch_1_coronal)
                            label_1 += mask_np[start_idx: start_idx + w, j + k + h//2, i - h//2: i + h//2].sum()

                        if use_sagital:
                            patch_1.append(patch_1_sagital)
                            label_1 += mask_np[start_idx + w//2, j + k: j + k + h, i - w//2: i + w//2].sum()

                        patch_1 = np.concatenate(tuple(patch_1))


                        patch_2_axial = np.stack((
                            target_np[-start_idx - w : -start_idx, j + k: j + k + h, i],
                            target_np[start_idx + w - 1: start_idx - 1: -1, j + k: j + k + h, i],
                        ))

                        patch_2_coronal = np.stack((
                            target_np[-start_idx - w : -start_idx, j + k + h//2, i - h//2: i + h//2],
                            target_np[start_idx + w - 1: start_idx - 1: -1, j + k + h//2, i - h//2: i + h//2],
                        ))

                        patch_2_sagital = np.stack((
                            target_np[-start_idx - w//2, j + k: j + k + h, i - w//2: i + w//2],
                            target_np[start_idx + w//2, j + k: j + k + h, i - w//2: i + w//2],
                        ))
                        patch_2_sagital = np.transpose(patch_2_sagital, axes=(0, 2, 1))



                        patch_2 = [patch_2_axial]
                        label_2 = mask_np[-start_idx - w : -start_idx, j + k: j + k + h, i].sum()

                        if use_coronal:
                            patch_2.append(patch_2_coronal)
                            label_2 += mask_np[-start_idx - w : -start_idx, j + k + h//2, i - h//2: i + h//2].sum()

                        if use_sagital:
                            patch_2.append(patch_2_sagital)
                            label_2 += mask_np[-start_idx - w//2, j + k: j + k + h, i - w//2: i + w//2].sum()

                        patch_2 = np.concatenate(tuple(patch_2))

                        if label_1:
                            all_patches[counter] = patch_1
                            all_labels[counter] = label_1
                            counter += 1

                        if label_2:
                            all_patches[counter] = patch_2
                            all_labels[counter] = label_2
                            counter += 1

                    if not (i <= 44 and j + k >= 118):
                        #middle patches

                        patch_3_axial = np.stack((
                            target_np[mid_idx: mid_idx + w, j + k: j + k + h, i],
                            target_np[mid_idx + 2*w - 1: mid_idx + w - 1: -1, j + k: j + k + h, i]
                        ))

                        patch_3_coronal = np.stack((
                            target_np[mid_idx: mid_idx + w, j + k + h//2, i - h//2: i + h//2],
                            target_np[mid_idx + 2*w - 1: mid_idx + w - 1: -1, j + k + h//2, i - h//2: i + h//2]
                        ))

                        patch_3_sagital = np.stack((
                            target_np[mid_idx + w//2, j + k: j + k + h, i - w//2: i + w//2],
                            target_np[mid_idx + w + w//2, j + k: j + k + h, i - w//2: i + w//2],
                        ))
                        patch_3_sagital = np.transpose(patch_3_sagital, axes=(0, 2, 1))


                        patch_3 = [patch_3_axial]
                        label_3 = mask_np[mid_idx: mid_idx + w, j + k: j + k + h, i].sum()

                        if use_coronal:
                            patch_3.append(patch_3_coronal)
                            label_3 += mask_np[mid_idx: mid_idx + w, j + k + h//2, i - h//2: i + h//2].sum()

                        if use_sagital:
                            patch_3.append(patch_3_sagital)
                            label_3 += mask_np[mid_idx + w//2, j + k: j + k + h, i - w//2: i + w//2].sum()
                        #print(i, j, k)
                        #print(patch_3_axial.shape)
                        #print(patch_3_coronal.shape)
                        #print(patch_3_sagital.shape)
                        patch_3 = np.concatenate(tuple(patch_3))

                        ######### 4

                        patch_4_axial = np.stack((
                            target_np[mid_idx + w: mid_idx + 2*w, j + k: j + k + h, i],
                            target_np[mid_idx + w - 1: mid_idx - 1 : -1, j + k: j + k + h, i],
                        ))

                        patch_4_coronal = np.stack((
                            target_np[mid_idx + w: mid_idx + 2*w,  j + k + h//2, i - h//2: i + h//2],
                            target_np[mid_idx + w - 1: mid_idx - 1 : -1, j + k + h//2, i - h//2: i + h//2]
                        ))

                        patch_4_sagital = np.stack((
                            target_np[mid_idx + w + w//2, j + k: j + k + h, i - w//2: i + w//2],
                            target_np[mid_idx + w//2, j + k: j + k + h, i - w//2: i + w//2],
                        ))
                        patch_4_sagital = np.transpose(patch_4_sagital, axes=(0, 2, 1))



                        patch_4 = [patch_4_axial]
                        label_4 = mask_np[mid_idx + w: mid_idx + 2*w, j + k: j + k + h, i].sum()

                        if use_coronal:
                            patch_4.append(patch_4_coronal)
                            label_4 += mask_np[mid_idx + w: mid_idx + 2*w,  j + k + h//2, i - h//2: i + h//2].sum()

                        if use_sagital:
                            patch_4.append(patch_4_sagital)
                            label_4 += mask_np[mid_idx + w + w//2, j + k: j + k + h, i - w//2: i + w//2].sum()

                        patch_4 = np.concatenate(tuple(patch_4))

                        if label_3:
                            all_patches[counter] = patch_3
                            all_labels[counter] = label_3
                            counter += 1

                        if label_4:
                            all_patches[counter] = patch_4
                            all_labels[counter] = label_4
                            counter += 1


    all_patches, all_labels = all_patches[:counter], all_labels[:counter]
    if all_labels.max() != 0.:
        all_labels = all_labels/all_labels.max()
        all_labels = all_labels**coef
    return all_patches, all_labels


def get_image_patches(input_img_name, gmpm=None, use_coronal=False,
                      use_sagital=False, input_mask_name=None, augment=True, h=16, w=32, coef=.2):
    target_img = nib.load(input_img_name)
    subject_dict = {
                'mri': torchio.Image(input_img_name, torchio.INTENSITY),
            }

    #torchio normalizaion
    t1_landmarks = Path('./data/t1_landmarks.npy')
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

    all_patches, all_labels = get_patches_and_labels(target_np, gmpm, mask_np, use_coronal=use_coronal, use_sagital=use_sagital, h=h, w=w, coef=coef)

    return all_patches, all_labels

def get_patch_list(use_controls: bool, use_fcd: bool, use_coronal: bool, use_sagital: bool, augment=True, hard_labeling=False, h=16, w=32, coef=.2):
    FCD_FOLDER = './data/fcd_brains/'
    CONROL_FOLDER = './data/control_brains/'
    MASK_FOLDER = './data/masks/'

    gray_matter_template = nib.load('./data/MNI152_T1_1mm_brain_gray.nii.gz')
    gmpm = gray_matter_template.get_fdata()
    gmpm = (gmpm > 0).astype('float')

    list_of_tensors = []
    list_of_labels = []
    #fcd brains
    for i, input_img_name in enumerate(sorted(os.listdir(FCD_FOLDER))):

        if not input_img_name.endswith('.nii.gz'):
            continue

        if 'nofcd' in input_img_name:
            continue
        print('Patch extraction:', input_img_name)
        number = input_img_name.split('_')[1][:2]
        input_mask_name = f'mask_{number}.nii.gz'

        X, y = get_image_patches(input_img_name=os.path.join(FCD_FOLDER, input_img_name),
                                 input_mask_name=os.path.join(MASK_FOLDER, input_mask_name),
                                 gmpm=gmpm, h=h, w=w, augment=augment, coef=coef,
                                 use_coronal=use_coronal, use_sagital=use_sagital)
        if hard_labeling:
            y = y > 0.
        list_of_tensors.append(X)
        list_of_labels.append(y)

    if use_fcd:
        for i, input_img_name in enumerate(sorted(os.listdir(FCD_FOLDER))):

            if not input_img_name.endswith('.nii.gz'):
                continue

            if 'nofcd' not in input_img_name:
                continue
            print('Patch extraction:', input_img_name)


            X, y = get_image_patches(input_img_name=os.path.join(FCD_FOLDER, input_img_name),
                                     input_mask_name=None,
                                     gmpm=gmpm, h=h, w=w, augment=augment, coef=coef,
                                     use_coronal=use_coronal, use_sagital=use_sagital)
            list_of_tensors.append(X)
            list_of_labels.append(y)

    if use_controls:
        for i, input_img_name in enumerate(sorted(os.listdir(CONROL_FOLDER))):
            if not input_img_name.endswith('.nii.gz'):
                continue
            print('Patch extraction:', input_img_name)
            X, y = get_image_patches(input_img_name=os.path.join(CONROL_FOLDER, input_img_name),
                                     input_mask_name=None,
                                     gmpm=gmpm, h=h, w=w, augment=augment, coef=coef,
                                     use_coronal=use_coronal, use_sagital=use_sagital)
            list_of_tensors.append(X)
            list_of_labels.append(y)
    return list_of_tensors, list_of_labels

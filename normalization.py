import os
import numpy as np
from pathlib import Path
from torchio.transforms import HistogramStandardization
import glob

FCD_FOLDER = './data/fcd_brains/'
CONTROL_FOLDER = './data/control_brains/'
NB_OF_FCD_SUBJECTS = 26
NB_OF_NOFCD_SUBJECTS = 15
NB_OF_CONTROL_SUBJECTS = 100500


def create_normalization_file(use_controls, use_nofcd, mods):
    """
        Creates t1_landmark.npy file using torchio library for brain normalizations.
    """

    for j in range(1, mods + 1):
        fcd_paths = sorted(glob.glob(FCD_FOLDER + f'fcd_*.{j}.nii.gz'))
        nofcd_paths = sorted(glob.glob(FCD_FOLDER + f'nofcd_*.{j}.nii.gz'))
        control_paths = sorted(glob.glob(CONTROL_FOLDER + f'control_*.{j}.nii.gz'))

        mri_paths = fcd_paths
        if use_nofcd:
            mri_paths += nofcd_paths
        if use_controls:
            mri_paths += control_paths

        t1_landmarks_path = Path(f'./data/t1_landmarks_{j}.npy')

        if t1_landmarks_path.is_file():
            continue
            # os.remove(f'./data/t1_landmarks_{j}.npy')

        t1_landmarks = (
            t1_landmarks_path
            if t1_landmarks_path.is_file()
            else HistogramStandardization.train(mri_paths)
        )

        np.save(t1_landmarks_path, t1_landmarks, allow_pickle=True)

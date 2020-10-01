import os
import numpy as np

from pathlib import Path
from torchio.transforms import HistogramStandardization
from torchio.transforms import ZNormalization

def create_normalization_file(use_controls, use_nofcd):
    """
        Creates t1_landmark.npy file using torchio library for brain normalizations.
    """
    FCD_FOLDER = './data/fcd_brains/'
    fcd_paths = sorted(list(filter(lambda x: 'nofcd' not in x, os.listdir(FCD_FOLDER))))
    fcd_paths = list(map(lambda x: FCD_FOLDER + x, fcd_paths))
    nofcd_paths = sorted(list(filter(lambda x: 'nofcd' in x, os.listdir(FCD_FOLDER))))
    nofcd_paths = list(map(lambda x: FCD_FOLDER + x, nofcd_paths))

    CONTROL_FOLDER = './data/control_brains/'
    control_paths = sorted(os.listdir(CONTROL_FOLDER))
    control_paths = list(map(lambda x: CONTROL_FOLDER + x, control_paths))

    mri_paths = fcd_paths
    if use_nofcd: 
        mri_paths += fcd_paths
    if use_controls:
        mri_paths += control_paths

    t1_landmarks_path = Path('./data/t1_landmarks.npy')

    if t1_landmarks_path.is_file():
        os.remove('./data/t1_landmarks.npy')
    
    t1_landmarks = (
        t1_landmarks_path
        if t1_landmarks_path.is_file()
        else HistogramStandardization.train(mri_paths)
    )

    np.save(t1_landmarks_path, t1_landmarks, allow_pickle=True)
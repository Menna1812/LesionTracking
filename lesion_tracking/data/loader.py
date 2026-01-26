import nibabel as nib
import numpy as np

def load_nifti(path):
    """
    Loads a nifti file and returns the data array.
    """
    img = nib.load(path)
    return img.get_fdata().astype(int)

from scipy.ndimage import label
import numpy as np

def label_mask(mask, structure=None):
    """
    Labels connected components in a binary mask.
    
    Args:
        mask (np.ndarray): Binary mask.
        structure (np.ndarray, optional): Structuring element for connectivity. 
                                          Defaults to 3x3x3 ones.
    
    Returns:
        tuple: (labeled_array, num_features)
    """
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=int)
    
    labeled_array, num_features = label(mask, structure=structure)
    return labeled_array, num_features

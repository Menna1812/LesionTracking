# MS Lesion Registration, Segmentation and Tracking 

This project provides tools for registering, segmenting, and tracking multiple sclerosis (MS) lesions from FLAIR MRI images. It includes:
- **Registration**: Longitudinal image registration using FSL FLIRT
- **Segmentation**: MS lesion segmentation using the FLAMeS model and nnU-Net v2
- **Lesion Tracking**: Track and analyze lesion changes over time


## Overview

This project provides three main functionalities:

1. **Longitudinal Registration** (`registeration.py`): Registers time-point 2 (T2) MRI images to time-point 1 (T1) images using FSL FLIRT with rigid transformation (6 DOF).
2. **Lesion Segmentation** (`run_ms_segmentation.py`): Automatically segments MS lesions from FLAIR images using the pretrained FLAMeS model with nnU-Net v2.
3. **Lesion Tracking**: Compares segmentation masks across time points to identify new, absent ,splited, and merged lesions.



### For Registration:
- Python 3.10 or 3.11 (64-bit)
- FSL (FLAIR Software Library) installed and accessible in system PATH
- Python libraries: `nibabel`, `nilearn`, `matplotlib`

### For Segmentation:
- Python 3.10 or 3.11 (64-bit)
- PyTorch (CPU or CUDA-enabled)
- Python libraries: `nnunetv2`, `nibabel`, `numpy`, `SimpleITK`, `pyyaml`
- NVIDIA GPU with CUDA support (recommended for segmentation)


### Longitudinal Image Registration with FSL FLIRT

Register a time-point 2 (T2) image to time-point 1 (T1) reference using rigid transformation.

### Usage

#### Command Line:
```bash
python registeration.py <moving_image> <reference_image> <output_image>
```

#### Parameters:
- `<moving_image>`: Path to the T2 image to register (NIfTI format)
- `<reference_image>`: Path to the T1 reference image (NIfTI format)
- `<output_image>`: Path where the registered image will be saved

#### Example:
```bash
python registeration.py patient_T2.nii.gz patient_T1.nii.gz patient_T2_registered.nii.gz
```

### Registration Details

- **Algorithm**: FLIRT (FMRIB's Linear Image Registration Tool)
- **Transformation**: Rigid (6 degrees of freedom - 3 rotation + 3 translation)
- **Cost Function**: Normalized mutual information (normmi)
- **Interpolation**: Spline
- **Output**: NIfTI format registered image

### Requirements for Registration

- FSL must be installed and the `flirt` command must be available in your system's PATH
- Input images must be in NIfTI format (.nii or .nii.gz)
- Both images should have same dimensions or FSL will handle the preprocessing

---

## Segmentation

### MS Lesion Segmentation with nnU-Net v2 and FLAMeS

Automatically segment MS lesions from FLAIR MRI images using a pretrained neural network model.

### Usage

#### Command Line:
```bash
python run_ms_segmentation.py <input_flair> <output_mask>
```

#### Parameters:
- `<input_flair>`: Path to the input FLAIR image (NIfTI format)
- `<output_mask>`: Path where the segmentation mask will be saved

#### Example:
```bash
python run_ms_segmentation.py P1_T1_FLAIR.nii.gz lesion_mask.nii.gz
```

### Segmentation Details

- **Model**: FLAMeS (pretrained)
- **Framework**: nnU-Net v2 (nnUNetv2)
- **Architecture**: 3D Full Resolution Network
- **Input**: FLAIR MRI image
- **Output**: Binary lesion mask (NIfTI format)
- **Training Epochs**: 8000
- **Cross-validation**: 5-fold



### Configuration

The segmentation process is configured via `config.json`:

```json
{
  "nnunet": {
    "dataset_name": "Dataset004_WML",
    "trainer": "nnUNetTrainer_8000epochs__nnUNetPlans__3d_fullres",
    "configuration": "3d_fullres"
  },
  "paths": {
    "runtime_dir": "nnunet_runtime",
    "model_dir": "FLAMeS_MODEL",
    "output_dir": "output"
  }
}
```

### Performance Notes

- Segmentation requires a CUDA-compatible GPU for reasonable performance
- CPU inference is possible but significantly slower
- Processing time: ~2-5 minutes per image (GPU-dependent)
- Output is a binary mask in NIfTI format (.nii.gz)

---

## Lesion Tracking


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

### Algorithm Description (implementation details)

lesion-tracking implementation is contained in `lesion_tracking_full.py`. The tracker operates on labeled binary lesion masks for a baseline time-point (T1) and a follow-up time-point (T2). The core steps are:

- Load baseline and follow-up masks as NIfTI volumes (using `nibabel`) and read voxel spacing from the NIfTI header to compute voxel volume (mm^3).
- Label connected components (26-connectivity) in each mask using `scipy.ndimage.label` (wrapped as `label_mask`). The labeled volumes use integer IDs for each lesion.
- Optionally filter out very small lesions using `filter_small_lesions()` which removes components smaller than `min_voxels` (configured in voxels or mm^3 using voxel volume) and re-labels the volume.
- Compute per-lesion sizes (voxel counts) with `compute_lesion_sizes()` and centers-of-mass with `compute_center_of_mass()`.
- Build an overlap map between baseline and follow-up labels with `compute_overlap_map()`. Each pair (A, B) stores the number of intersecting voxels between baseline lesion A and follow-up lesion B.
- Detect splits (one baseline → multiple follow-up) and merges (multiple baseline → one follow-up) by comparing intersection scores to a threshold `tau` (used differently for split detection and for general overlap handling). The default split-detection threshold is `0.2` and the overlap acceptance threshold is `0.1`.
- Classify each lesion using `classify_and_update()` which implements rules for splits, merges, present, new, enlarged, shrinking, and absent lesions. The function returns the follow-up label map with propagated/tracked IDs, a per-lesion `status` dictionary, and a `volumes` dictionary with final reported volumes used to build CSV summary tables.

Key implementation details you can find in the code:

- Splits: detected when a baseline lesion A overlaps with two or more follow-up lesions B whose intersection score with A is >= `split_tau`. The implementation assigns all split fragments the parent's ID (so they share the same label in the tracked follow-up map) and reports the lesion's follow-up volume as the sum of the split fragments' volumes.
- Merges: detected when a follow-up lesion B overlaps with two or more baseline lesions A with intersection score >= `split_tau`. The follow-up label B is assigned the ID of the largest baseline parent lesion. The reported volume for that surviving parent is set to the sum of the baseline parents' volumes; other parent lesions are marked as `merged` with zero follow-up volume.
- Normal one-to-one matches: if overlap score >= `overlap_tau` (default 0.1) and the follow-up lesion B is not part of a merge, the baseline lesion A inherits B's measured follow-up volume and is classified as `present`, `enlarged`, or `shrinking` depending on the volume ratio.
- New lesions: follow-up lesions with no significant overlap are assigned new unique IDs (IDs start from max(baseline IDs)+1) and status `new`.

### Numerical thresholds and status rules

- Overlap thresholds:
  - `split_tau` (default 0.2): used to detect splits/merges (requires stronger evidence).
  - `overlap_tau` (default 0.1): used to accept a baseline↔follow-up pairing as a tracked correspondence.
- Enlargement / Shrinking thresholds: a lesion is considered `enlarged` if follow-up volume >= 1.25 * baseline volume (≥25% increase) and `shrinking` if follow-up volume <= 0.75 * baseline volume (≥25% decrease). Otherwise the lesion is `present` if it remains matched.

### Lesion statuses (reported in the CSV summary)

Each tracked lesion is assigned one of the following statuses in the follow-up summary:

- Present: default for baseline lesions that persist with significant overlap and without a ≥25% volume change.
- Absent: baseline lesions that do not appear in the follow-up report (no matching overlap and not represented in the final labeled follow-up map). In the follow-up CSV these are assigned when there is no final volume recorded for that lesion.
- Merged: multiple baseline lesions merged into a single follow-up lesion. Both (or all) merged baseline lesions are labeled `merged` in the follow-up table. The largest parent's row will carry the summed volume of all parents; other merged parents will have zero follow-up volume recorded.
- Split: a baseline lesion that divides into multiple follow-up fragments. All resulting fragments are assigned the original baseline ID in the tracked follow-up map. In the summary table these fragments are reported as a single row (the parent's ID) with follow-up volume equal to the sum of the split fragments' volumes.
- Enlarged: lesion volume increased by 25% or more at follow-up (volume ratio ≥ 1.25).
- Shrinking: lesion volume decreased by 25% or more at follow-up (volume ratio ≤ 0.75).

Note: `new` lesions (follow-up-only) are also reported in the follow-up summary table and receive a new unique `Lesion ID` and `Status` = `new`.

### Outputs

- For each processed patient the tracker saves:
  - `{PID}_Baseline_summary_table.csv` — table with baseline lesions (ID, baseline volume, COM, initial `Present` status)
  - `{PID}_Followup_summary_table.csv` — table with tracked lesions, final volumes, baseline volumes, COM, `Status`, and percentage change
  - `{PID}_Followup_labeled.nii.gz` — a NIfTI image where follow-up voxels are labeled with tracked lesion IDs (baseline IDs preserved when possible; new lesions get new IDs)

CSV fields and conversions:
- Volumes in CSV are reported in mm^3. Voxel volume is computed from the first three elements of the NIfTI header `get_zooms()` and multiplied by voxel counts.
- Percentage change is reported as a floored integer percent computed from `(followup - baseline) / baseline * 100` in the code.

### Corner cases and implementation notes (important to review)

- Overlapping splits and merges: a follow-up lesion might be involved in both merge and split patterns depending on thresholds. The current pipeline first resolves splits (assigns parent IDs to all split fragments), then handles normal overlaps, and finally handles merges. This ordering biases results toward honoring splits before merges and can be tuned by changing `split_tau` and `overlap_tau` in `lesion_tracking_full.py`.

- Tie-breaking in merges: when selecting the parent ID to keep for a merged lesion, the code selects the largest baseline parent by voxel count. If multiple parents have equal voxel count, selection depends on Python's ordering (the `max` call will return the first max encountered), which can be deterministic for identical inputs but should be considered when interpreting results.

- Volume reporting for merges/splits: the code follows the requested rules: merged lesions carry the summed volume on the largest parent's row while the other parents are given zero follow-up volume; split fragments all share the parent's ID and the parent's follow-up volume is the sum of fragments.

- Small lesion filtering: very small components are removed before analysis (if `min_voxels > 0`) and the volumes/tables use the relabeled volumes after filtering. This affects split/merge detection when a fragment is filtered out.

- Center-of-mass computation: COM uses `scipy.ndimage.center_of_mass` and the code floors the coordinates to integers. The COM shown in the CSV is based on the COM of the final labeled follow-up map for follow-up rows and on the baseline labeled map for baseline rows.

- Edge/empty inputs: the code raises on mixed directory/file inputs; if no matching patient IDs are found in baseline/follow-up directories the function prints a message and continues. Be careful to ensure baseline and follow-up inputs are either both files or both directories.

### Where to tune behavior

- `split_tau` and `overlap_tau` in `run_lesion_tracking()` control how conservative the algorithm is when declaring splits/merges and general overlaps.
- `min_voxels` controls the minimum lesion size retained for analysis (in voxel counts, optionally mapped to mm^3 using voxel volume).
- `enlarge_thr` and `shrink_thr` are set to 1.25 and 0.75 in `classify_and_update()` to match the 25% thresholds; modify these constants if you want different percent-change cutoffs.

### Usage examples

Below are quick, copy-pasteable examples showing how to run the two primary `lesion_tracking_full.py` subcommands: label_baseline and run_tracking.

Label baseline (produce a labeled baseline NIfTI):
```bash
python lesion_tracking_full.py label_baseline path/to/baseline_mask.nii.gz output_dir --min_voxels 10
# saves: output_dir/labeled_baseline.nii.gz
```

Run tracking (single patient files). `baseline_input_labeled` should be the labeled baseline NIfTI produced above:
```bash
python lesion_tracking_full.py run_tracking output_dir/labeled_baseline.nii.gz path/to/followup_mask.nii.gz output_dir --min_voxels 10
# saves: {PID}_Baseline_summary_table.csv, {PID}_Followup_summary_table.csv, {PID}_Followup_labeled.nii.gz in output_dir
```

Run tracking (directories of baseline and follow-up masks; matches files by prefix before the underscore):
```bash
python lesion_tracking_full.py run_tracking /path/to/baseline_dir /path/to/followup_dir output_dir --min_voxels 10
```

Notes:
- Adjust `--min_voxels` to filter very small components before analysis (10 by default).





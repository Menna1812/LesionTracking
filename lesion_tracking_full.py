import os
import sys
import glob
import argparse
import math
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label, center_of_mass
from collections import defaultdict


def load_nifti(path):
    """
    Loads a nifti file and returns the data array and the affine matrix.
    
    Args:
        path (str): The absolute or relative path to the .nii or .nii.gz file.
        
    Returns:
        tuple: (data_array (np.ndarray), affine (np.ndarray), header)
        - data_array: The 3D numpy array containing the voxel data (casted to int).
        - affine: The 4x4 affine matrix defining the image orientation/spacing.
        - header: The original NIfTI header.
    """
    img = nib.load(path)
    data = img.get_fdata().astype(int)
    return data, img.affine, img.header

def save_nifti(data, affine, header, output_path):
    """
    Saves a numpy array as a NIfTI file.
    
    Args:
        data (np.ndarray): The 3D data array.
        affine (np.ndarray): The 4x4 affine matrix.
        header: The NIfTI header.
        output_path (str): The path to save the file to.
    """
    new_img = nib.Nifti1Image(data, affine, header)
    nib.save(new_img, output_path)


def label_mask(mask, structure=None):
    """
    Labels connected components in a binary mask.
    
    Args:
        mask (np.ndarray): Binary mask (input volume).
        structure (np.ndarray, optional): Structuring element for connectivity. 
                                          Defaults to 3x3x3 ones (26-connectivity).
    
    Returns:
        tuple: (labeled_array, num_features)
        - labeled_array: Volume where each connected component has a unique integer ID.
        - num_features: Total number of connected components found.
    """
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=int)
    
    labeled_array, num_features = label(mask, structure=structure)
    return labeled_array, num_features

def compute_lesion_sizes(labeled_volume):
    """
    Computes the size (voxel count) of each lesion in the labeled volume.
    
    Args:
        labeled_volume (np.ndarray): Labeled volume where each lesion has a unique integer ID.
        
    Returns:
        dict: {lesion_id: voxel_count}
              A dictionary mapping each lesion ID (int) to its size in voxels (int).
    """
    unique_labels, counts = np.unique(labeled_volume, return_counts=True)
    size_dict = dict(zip(unique_labels, counts))
    if 0 in size_dict:
        del size_dict[0] # remove background
    return size_dict


def compute_center_of_mass(labeled_volume):
    """
    Computes the center of mass (x, y, z) for each lesion in the labeled volume.
    
    Args:
        labeled_volume (np.ndarray): Labeled volume where each lesion has a unique integer ID.
        
    Returns:
        dict: {lesion_id: (x, y, z)}
              A dictionary mapping each lesion ID (int) to its center of mass tuple.
    """
    unique_labels = np.unique(labeled_volume)
    unique_labels = unique_labels[unique_labels != 0]
    
    coms = {}
    for label_id in unique_labels:
        # Calculate COM for each label individually to ensure consistent tuple output
        com = tuple(math.floor(c) for c in center_of_mass(labeled_volume, labeled_volume, index=label_id))
        coms[label_id] = com
        
    return coms

def compute_overlap_map(labels_A, labels_B):

    """
    Computes intersection voxel counts between lesions in volume A and volume B.
    
    Args:
        labels_A (np.ndarray): Baseline labeled volume.
        labels_B (np.ndarray): Follow-up labeled volume.
        
    Returns:
        dict: {(A_id, B_id): intersection_voxel_count}
              A dictionary mapping pairs of intersecting lesion IDs to the number of overlapping voxels.
    """
    overlap = {}
    
    A_ids = np.unique(labels_A)
    A_ids = A_ids[A_ids != 0]
    
    for A in A_ids:
        mask_A = labels_A == A
        overlapping_B = labels_B[mask_A]
        
        B_ids, counts = np.unique(overlapping_B, return_counts=True)
        for B, cnt in zip(B_ids, counts):
            if B == 0:
                continue
            overlap[(A, B)] = cnt
            
    return overlap

def detect_splits(overlap, size_A, size_B, tau=0.2):
    """
    Detects lesions in A that have split into multiple lesions in B.
    
    Args:
        overlap (dict): Overlap map {(A, B): count} obtained from compute_overlap_map.
        size_A (dict): Sizes of lesions in A (Baseline).
        size_B (dict): Sizes of lesions in B (Follow-up).
        tau (float): Overlap threshold score = intersection / min(size_A, size_B).
        
    Returns:
        dict: {A_id: [(B_id, intersection), ...]} 
              A dictionary mapping a parent lesion ID in A to a list of children lesion IDs in B 
              for true splits (>=2 children).
    """
    splits = defaultdict(list)
    
    for (A, B), inter in overlap.items():
        score = inter / min(size_A[A], size_B[B])
        if score >= tau:
            splits[A].append((B, inter))
            
    # keep only true splits
    return {
        A: Bs for A, Bs in splits.items()
        if len(Bs) >= 2
    }

def assign_split_ids(A_id, children):
    """
    Assigns IDs to split children. 
    In this implementation, all split fragments retain the parent's ID to maintain lineage tracking.
    
    Args:
        A_id (int): ID of the parent lesion in baseline.
        children (list): List of (B_id, intersection_count) tuples for children in follow-up.
        
    Returns:
        dict: {B_id: final_id} mapping.
              Different parts of the split in B get assigned the original ID from A.
    """
    # sort by intersection size
    children = sorted(children, key=lambda x: x[1], reverse=True)
    
    mapping = {}
    for (B, _) in children:
        mapping[B] = A_id
        
    return mapping

def merge_handling(overlap, size_A, size_B, tau=0.2):
    """
    Detects lesions in B that are merged from multiple lesions in A.
    
    Args:
        overlap (dict): Overlap map.
        size_A (dict): Sizes in A.
        size_B (dict): Sizes in B.
        tau (float): Threshold for overlap.
        
    Returns:
        dict: {B_id: [(A_id, intersection), ...]} 
              for true merges (>=2 parents).
    """
    merges = defaultdict(list)
    
    for (A, B), inter in overlap.items():
        score = inter / min(size_A[A], size_B[B])
        if score >= tau:
            merges[B].append((A, inter))
            
    return {
        B: As for B, As in merges.items()
        if len(As) >= 2
    }

def classify_and_update(labeled_T1, labeled_T2, size_1, size_2, overlap, splits, merges, tau=0.1, enlarge_thr=1.25, shrink_thr=0.75):
    """
    Main logic to classify lesions and assign final IDs for follow-up.
    
    Args:
        labeled_T1, labeled_T2: Labeled volumes.
        size_1, size_2: Dictionaries of lesion sizes.
        overlap: Overlap map.
        splits: Detected splits.
        merges: Detected merges.
        tau (float): Overlap threshold.
        enlarge_thr (float): Ratio threshold for enlargement (default 1.25).
        shrink_thr (float): Ratio threshold for shrinking (default 0.75).
        
    Returns:
        tuple: (volumes, status, labels_B_final)
        - volumes: Dict of final volumes keyed by Lesion ID.
        - status: Dict of classification status keyed by Lesion ID.
        - labels_B_final: The new labeled volume for follow-up with propagated IDs.
    
    Classification categories:
    - Splits: One baseline lesion splits into multiple follow-up lesions.
    - Merges: Multiple baseline lesions merge into one follow-up lesion.
    - Enlarged: Follow-up volume >= 1.25 * Baseline volume.
    - Shrinking: Follow-up volume <= 0.75 * Baseline volume.
    - Present: Significant overlap but stable volume.
    - New: No significant overlap with any baseline lesion.
    """
    labels_B_final = np.zeros_like(labeled_T2, dtype=np.int32)
    
    processed_B = set()
    volumes = {}
    status = {}
    ratios = {}
    
    # STEP 1: Handle SPLITS
    for A, children in splits.items():
        # Sum volumes of all split fragments in T2 for the total new volume
        volume = sum(size_2[B] for B, _ in children)
        
        mapping = assign_split_ids(A, children)
        
        for B, final_id in mapping.items():
            labels_B_final[labeled_T2 == B] = final_id
            processed_B.add(B)
            
        volumes[A] = volume
        ratios[A] = (volumes[A] - size_1[A]) / size_1[A]
        status[A] = 'split'
        
    # STEP 2: Handle NORMAL overlaps (non-splits)
    sorted_overlap = sorted(overlap.items())
    
    for (A, B), inter in sorted_overlap:
        if B in processed_B:
            continue
            
        score = inter / min(size_1[A], size_2[B])
        
        if score >= tau:
            # If B is part of a merge, it will be handled in STEP 3
            if B in merges:
                continue 
            
            volumes[A] = size_2[B]

            ratio = volumes[A] / size_1[A]
            ratios[A] = (volumes[A] - size_1[A]) / size_1[A]
            
            if ratio >= enlarge_thr:
                curr_status = 'enlarged'
            elif ratio <= shrink_thr:
                curr_status = 'shrinking'
            else:
                curr_status = 'present'
                
            status[A] = curr_status
            labels_B_final[labeled_T2 == B] = A
            processed_B.add(B)
            
    # STEP 3: Handle MERGES
    for B, parents in merges.items():
        if B in processed_B:
            continue
            
        # Assign ID of largest parent to the merged lesion
        largest_parent_A = max(parents, key=lambda x: size_1[x[0]])[0]
        
        labels_B_final[labeled_T2 == B] = largest_parent_A
        
        # In this logic, volume of merged lesion is sum of parents' baseline volumes
        volumes[largest_parent_A] = sum(size_1[p[0]] for p in parents)
        ratios[largest_parent_A] = (volumes[largest_parent_A] - size_1[largest_parent_A]) / size_1[largest_parent_A]
        status[largest_parent_A] = 'merged'
        
        for p in parents:
            p_id = p[0]
            if p_id != largest_parent_A:
                volumes[p_id] = 0 # Merged source lesions lose their individual volume entry
                status[p_id] = 'merged'
                
        processed_B.add(B)
        
    # STEP 4: Handle NEW lesions (no significant overlap)
    if labeled_T1.size > 0:
        max_t1 = np.max(labeled_T1)
        next_new_id = int(max_t1) + 1
    else:
        next_new_id = 1
        
    unique_B = np.unique(labeled_T2)
    unique_B = unique_B[unique_B != 0]
    
    for B in unique_B:
        if B not in processed_B:
            # Assign new unique ID
            labels_B_final[labeled_T2 == B] = next_new_id
            status[next_new_id] = 'new'
            volumes[next_new_id] = size_2[B]
            ratios[next_new_id] = 1
            next_new_id += 1
            
    return volumes, status, labels_B_final, ratios

def build_lesion_summary_tables(size_1, final_volumes, ratios, overlap, splits, merges, labeled_T1, labeled_T2, status, com_map_1, com_map_2, voxel_vol_1=1.0, voxel_vol_2=1.0):
    """
    Builds the summary tables for baseline and follow-up.
    
    Args:
        size_1: Baseline lesion sizes.
        final_volumes: Final calculated volumes.
        overlap: Overlap map.
        splits: Split info.
        merges: Merge info.
        labeled_T1: Baseline label map.
        labeled_T2: Followup label map.
        status: Classification status.
        voxel_vol_1: Volume of a single voxel in baseline image (mm^3).
        voxel_vol_2: Volume of a single voxel in follow-up image (mm^3).
        
    Returns:
        tuple: (baseline_df, followup_df)
    """
    # Baseline table
    A_ids = set(np.unique(labeled_T1))
    A_ids = A_ids - {0}
    
    baseline_rows = []
    for A in A_ids:
        baseline_rows.append({
            "Lesion ID": A,
            "Baseline Volume (mm3)": size_1.get(A, 0) * voxel_vol_1,
            "Center of Mass": str(com_map_1.get(A, "N/A")),
            "Status": "Present"
        })
    baseline_table = pd.DataFrame(baseline_rows)
    
    # Follow-up table
    # Include all tracked IDs: original baseline IDs (if persisted/merged/split) and new IDs
    reportable_ids = set(A_ids)
    reportable_ids.update(final_volumes.keys())
    
    followup_rows = []
    for lesion_id in reportable_ids:
        vol = final_volumes.get(lesion_id, 0)
        stat = status.get(lesion_id, "Absent")
        percentage_change = ratios.get(lesion_id, -1) * 100
        
        followup_rows.append({
            "Lesion ID": lesion_id,
            "Follow-up Volume (mm3)": vol * voxel_vol_2,
            "Baseline Volume (mm3)": size_1.get(lesion_id, 0) * voxel_vol_1,
            "Center of Mass": str(com_map_2.get(lesion_id, "N/A")),
            "Status": stat,
            "Percentage Change": f"{math.floor(percentage_change)}%"
        })
        
    followup_table = pd.DataFrame(followup_rows)
    return baseline_table, followup_table

def extract_patient_id(path):
    """
    Extracts patient ID from the filename.
    Assumes filename format starts with PatientID followed by underscore like 'P9_T1_MASK.nii.gz'.
    
    Args:
        path (str): File path.
        
    Returns:
        str: Extracted Patient ID.
    """
    filename = os.path.basename(path)
    if '_' in filename:
        return filename.split('_')[0]
    return "Patient"

def filter_small_lesions(labeled_volume, min_voxels, voxel_vol):
    """
    Zeroes out lesions with fewer than min_voxels.
    """
    if min_voxels <= 0:
        return labeled_volume
        
    unique, counts = np.unique(labeled_volume, return_counts=True)
    
    # labels to remove (exclude background 0)
    small_labels = unique[((counts * voxel_vol) < min_voxels) & (unique != 0)]
    
    if len(small_labels) > 0:
        # Set voxels of small lesions to 0 (background)
        mask = np.isin(labeled_volume, small_labels)
        labeled_volume[mask] = 0
        
    # Re-label to ensure consecutive IDs and no gaps
    labeled_volume, _ = label_mask(labeled_volume > 0)
    return labeled_volume

def run_lesion_tracking( followup_input, labeled_T1, output_dir, min_voxels=10, baseline_input = None):
    """
    Wraps all functionality to process lesion tracking for baseline and follow-up.
    Can accept either direct file paths or directories containing matching files.
    
    Args:
        baseline_input (str): Path to the baseline file OR directory containing baseline files.
        followup_input (str): Path to the follow-up file OR directory containing follow-up files.
        output_dir (str): Directory where the output files (images and tables) will be saved.
        
    Returns:
        tuple: (baseline_images_map, followup_images_map, baseline_tables_map, followup_tables_map)
        
        The returns are dictionaries where keys are PatientIDs (or filenames) and values are:
        - baseline_images_map: {pid: labeled_T1_nib_object}
        - followup_images_map: {pid: labeled_T2_final_nib_object}
        - baseline_tables_map: {pid: baseline_dataframe}
        - followup_tables_map: {pid: followup_dataframe}
    """
    
    # Determine if inputs are files or directories
    baseline_is_dir = os.path.isdir(labeled_T1)
    followup_is_dir = os.path.isdir(followup_input)
    
    # Prepare list of pairs to process: (baseline_path, followup_path, patient_id)
    pairs = []
    
    if baseline_is_dir and followup_is_dir:
        # Scan directories (assuming .nii or .nii.gz)
        b_files = sorted(glob.glob(os.path.join(labeled_T1, "*.nii*")))
        f_files = sorted(glob.glob(os.path.join(followup_input, "*.nii*")))
        
        # Convert to dict for easy lookup
        b_dict = {extract_patient_id(f): f for f in b_files}
        f_dict = {extract_patient_id(f): f for f in f_files}
        
        common_ids = set(b_dict.keys()) & set(f_dict.keys())
        
        if not common_ids:
            print("No matching patient IDs found between baseline and follow-up directories.")
        else:
            print(f"Found {len(common_ids)} matching pairs.")
            
        for pid in common_ids:
            pairs.append((b_dict[pid], f_dict[pid], pid))
            
    elif not baseline_is_dir and not followup_is_dir:
        # Both are files
        pid = extract_patient_id(labeled_T1)
        pairs.append((labeled_T1, followup_input, pid))
    else:
        # Mixed file/dir input is ambiguous
        raise ValueError("Both inputs must be either files or directories.")
        
    # Results containers
    baseline_imgs = {}
    followup_imgs = {}
    baseline_tbls = {}
    followup_tbls = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    for b_path, f_path, pid in pairs:
        print(f"Processing Patient: {pid}")
        print(f"  Baseline: {b_path}")
        print(f"  Follow-up: {f_path}")
        
        # Load
        # Note: We need affine/header to save the outputs properly
        mask_1, affine_1, header_1 = load_nifti(b_path)
        mask_2, affine_2, header_2 = load_nifti(f_path)
        
        # Extract voxel volumes (mm^3)
        # header.get_zooms() returns (dx, dy, dz, dt...), we take top 3
        zooms_1 = header_1.get_zooms()
        voxel_vol_1 = np.prod(zooms_1[:3])
        
        zooms_2 = header_2.get_zooms()
        voxel_vol_2 = np.prod(zooms_2[:3])
        
        # Label
        labeled_T1 = mask_1
        labeled_T2, num_lesions_2 = label_mask(mask_2)
        
        # Filter small lesions
        if min_voxels > 0:
            labeled_T1 = filter_small_lesions(labeled_T1, min_voxels, voxel_vol_1)
            labeled_T2 = filter_small_lesions(labeled_T2, min_voxels, voxel_vol_2)
        
        # Analyze
        size_1 = compute_lesion_sizes(labeled_T1)
        size_2 = compute_lesion_sizes(labeled_T2)
        
        overlap = compute_overlap_map(labeled_T1, labeled_T2)
        
        # thresholds for split and overlap handling
        split_tau = 0.2
        overlap_tau = 0.1
        
        splits = detect_splits(overlap, size_1, size_2, tau=split_tau)
        merges = merge_handling(overlap, size_1, size_2, tau=split_tau)
        
        volumes, status, labels_B_final, ratios = classify_and_update(
            labeled_T1, labeled_T2, size_1, size_2, overlap, splits, merges, tau=overlap_tau
        )
        
        # Compute Center of Mass
        com_map_1 = compute_center_of_mass(labeled_T1)
        com_map_2 = compute_center_of_mass(labels_B_final)
        
        baseline_df, followup_df = build_lesion_summary_tables(
            size_1, volumes, ratios, overlap, splits, merges, labeled_T1, labeled_T2, status,
            com_map_1, com_map_2,
            voxel_vol_1=voxel_vol_1, voxel_vol_2=voxel_vol_2
        )
        
        # Save Tables
        b_csv_name = f"{pid}_Baseline_summary_table.csv"
        f_csv_name = f"{pid}_Followup_summary_table.csv"
        baseline_df.to_csv(os.path.join(output_dir, b_csv_name), index=False)
        followup_df.to_csv(os.path.join(output_dir, f_csv_name), index=False)
        
        
        # Followup labeled with tracked IDs (labels_B_final)
        img_f_labeled = nib.Nifti1Image(labels_B_final, affine_2, header_2)
        f_nii_name = f"{pid}_Followup_labeled.nii.gz"
        nib.save(img_f_labeled, os.path.join(output_dir, f_nii_name))
        
        # Store in dicts
        baseline_imgs[pid] = labeled_T1
        followup_imgs[pid] = img_f_labeled
        baseline_tbls[pid] = baseline_df
        followup_tbls[pid] = followup_df
        
        print(f"  Saved outputs to {output_dir}")
        
    return baseline_imgs, followup_imgs, baseline_tbls, followup_tbls

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lesion Tracking Analysis")
    parser.add_argument("baseline_input", help="Path to labeled baseline file or directory")
    parser.add_argument("followup_input", help="Path to follow-up file or directory")
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument("--min_voxels", type=int, default=10, help="Minimum voxel count for a lesion to be included (default: 10)")
    parser.add_argument("--baseline_input", help="Path to baseline file or directory")
    
    args = parser.parse_args()
    
    run_lesion_tracking(labeled_T1=args.baseline_input, followup_input=args.followup_input, output_dir=args.output_dir, min_voxels=args.min_voxels, baseline_input=args.baseline_input)
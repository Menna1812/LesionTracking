import os
import sys
from .core.labeling import label_mask
from .core.analysis import (
    compute_lesion_sizes,
    compute_overlap_map,
    detect_splits,
    merge_handling,
    classify_and_update,
    build_lesion_summary_tables
)
from .data.loader import load_nifti

def extract_patient_id(path):
    """
    Extracts patient ID from the filename.
    Assumes filename format starts with PatientID followed by underscore like 'P9_T1_MASK.nii.gz'.
    """
    filename = os.path.basename(path)
    if '_' in filename:
        return filename.split('_')[0]
    return "Patient"

def process_lesion_masks(baseline_path, followup_path, output_dir=None, overlap_tau=0.1, split_tau=0.2):
    """
    Main pipeline to process lesion masks.
    
    Args:
        baseline_path (str): Path to baseline nifti mask.
        followup_path (str): Path to follow-up nifti mask.
        output_dir (str, optional): Ignored. Output is always saved to 'output' folder.
        overlap_tau (float): Threshold for non-split overlap detection.
        split_tau (float): Threshold for split detection.
        
    Returns:
        tuple: (baseline_df, followup_df)
    """
    
    # Enforce output directory as 'output'
    target_output_dir = 'output'
        
    print(f"Loading {baseline_path} and {followup_path}...")
    mask_1 = load_nifti(baseline_path)
    mask_2 = load_nifti(followup_path)
    
    print("Labeling connected components...")
    labeled_T1, num_lesions_1 = label_mask(mask_1)
    labeled_T2, num_lesions_2 = label_mask(mask_2)
    
    print(f"Found {num_lesions_1} lesions in baseline, {num_lesions_2} in follow-up.")
    
    print("Computing sizes and overlaps...")
    size_1 = compute_lesion_sizes(labeled_T1)
    size_2 = compute_lesion_sizes(labeled_T2)
    
    overlap = compute_overlap_map(labeled_T1, labeled_T2)
    
    splits = detect_splits(overlap, size_1, size_2, tau=split_tau)
    if splits:
        print(f"Detected {len(splits)} splits.")
        
    merges = merge_handling(overlap, size_1, size_2, tau=split_tau)
    if merges:
        print(f"Detected {len(merges)} merges.")
    
    print("Classifying lesions...")
    volumes, status, labels_B_final = classify_and_update(
        labeled_T1, labeled_T2, size_1, size_2, overlap, splits, merges, tau=overlap_tau
    )
    
    print("Building summary tables...")
    baseline_df, followup_df = build_lesion_summary_tables(
        size_1, volumes, overlap, splits, merges, labeled_T1, labeled_T2, status
    )
    
    # Extract Patient ID
    patient_id = extract_patient_id(baseline_path)
    
    os.makedirs(target_output_dir, exist_ok=True)
    
    # Naming format: "{PatientID} Baseline summary table.csv"
    b_filename = f"{patient_id} Baseline summary table.csv"
    f_filename = f"{patient_id} Follow up summary table.csv"
    
    b_out = os.path.join(target_output_dir, b_filename)
    f_out = os.path.join(target_output_dir, f_filename)
    
    baseline_df.to_csv(b_out, index=False)
    followup_df.to_csv(f_out, index=False)
    print(f"Saved tables to {target_output_dir} as '{b_filename}' and '{f_filename}'")
        
    return baseline_df, followup_df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m lesion_tracking.main <baseline.nii.gz> <followup.nii.gz>")
        sys.exit(1)
        
    base = sys.argv[1]
    follow = sys.argv[2]
    
    process_lesion_masks(base, follow)

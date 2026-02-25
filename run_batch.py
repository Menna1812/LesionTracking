
import os
import glob
import sys
from lesion_tracking_full import run_lesion_tracking

def batch_process(dataset_dir, output_dir):
    """
    Runs lesion tracking on all patients in the dataset directory.
    Assumes structure: dataset_dir/{PatientID}/T1/...FLAIR... and dataset_dir/{PatientID}/T2/...FLAIR...
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    patient_dirs = sorted(glob.glob(os.path.join(dataset_dir, "P*")))
    
    print(f"Found {len(patient_dirs)} patient directories.")
    
    for p_dir in patient_dirs:
        pid = os.path.basename(p_dir)
        
        # Construct paths to T1 and T2 FLAIR images
        # The user said "using T1 as baseline and T2 as followup"
        # And "each folder has T1 and T2 folder each having FLAIR image"
        
        # Pattern matching for flexibility in case filename varies slightly
        t1_flair_pattern = os.path.join(p_dir, "T1", "*_FLAIR.nii.gz")
        t2_flair_pattern = os.path.join(p_dir, "T2", "*_FLAIR.nii.gz")
        
        t1_files = glob.glob(t1_flair_pattern)
        t2_files = glob.glob(t2_flair_pattern)
        
        if not t1_files or not t2_files:
            print(f"Skipping {pid}: Could not find FLAIR images in T1/T2 subfolders.")
            continue
            
        t1_path = t1_files[0]
        t2_path = t2_files[0]
        
        print(f"Processing {pid}...")
        print(f"  Baseline: {t1_path}")
        print(f"  Followup: {t2_path}")
        
        try:
            # We call run_lesion_tracking with file paths directly.
            # It expects (baseline_input, followup_input, output_dir)
            # Since we are running in a loop, we can pass the same output_dir 
            # and it will save files prefixed with the Patient ID extracted from the file name.
            # Assuming run_lesion_tracking handles single file inputs correctly as per its docstring.
            
            run_lesion_tracking(t1_path, t2_path, output_dir)
            
        except Exception as e:
            print(f"Error processing {pid}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    dataset_path = "/home/mannon/LesionTracking/MSLesSeg_Dataset/train"
    output_path = "/home/mannon/LesionTracking/output_batch"
    
    batch_process(dataset_path, output_path)

import numpy as np
import nibabel as nib
import os
import subprocess
import pandas as pd
import shutil

def create_synthetic_data(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create empty volume 50x50x50
    shape = (50, 50, 50)
    affine = np.eye(4)
    
    # Baseline: 
    # Lesion 1: Large (20 voxels) at (10, 10, 10)
    # Lesion 2: Small (5 voxels) at (30, 30, 30)
    data_b = np.zeros(shape, dtype=np.int32)
    
    # Large lesion
    data_b[10:12, 10:15, 10:12] = 1 # 2*5*2 = 20 voxels
    
    # Small lesion
    data_b[30:35, 30, 30] = 1 # 5 voxels
    
    # Followup: Same
    data_f = data_b.copy()
    
    # Save
    img_b = nib.Nifti1Image(data_b, affine)
    img_f = nib.Nifti1Image(data_f, affine)
    
    nib.save(img_b, os.path.join(output_dir, "Patient1_baseline.nii.gz"))
    nib.save(img_f, os.path.join(output_dir, "Patient1_followup.nii.gz"))
    
    return os.path.join(output_dir, "Patient1_baseline.nii.gz"), os.path.join(output_dir, "Patient1_followup.nii.gz")

def test_run():
    test_dir = "test_threshold_repro"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    b_path, f_path = create_synthetic_data(test_dir)
    output_dir = os.path.join(test_dir, "output")
    
    print("Running lesion tracking (default settings)...")
    # This should detect BOTH lesions currently
    cmd = ["python3", "../lesion_tracking_full.py", b_path, f_path, output_dir]
    subprocess.run(cmd, check=True)
    
    # Check output
    csv_path = os.path.join(output_dir, "Patient1_Baseline_summary_table.csv")
    df = pd.read_csv(csv_path)
    
    print("\nDetected Lesions in Baseline:")
    print(df)
    
    lesion_sizes = df["Baseline Volume (mm3)"].tolist()
    has_small = any(s <= 5 for s in lesion_sizes)
    has_large = any(s >= 20 for s in lesion_sizes)
    
    if has_small and has_large:
        print("\nSUCCESS: Reproduction confirmed. Both small (<=5) and large (>=20) lesions detected.")
    else:
        print("\nFAILURE: Reproduction failed. Unexpected lesions detected.")

if __name__ == "__main__":
    test_run()

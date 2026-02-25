
import os
import glob
import sys

# Add parent directory to path to import lesion_tracking_full
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

try:
    from lesion_tracking_full import run_lesion_tracking
except ImportError:
    print(f"Could not import lesion_tracking_full from {parent_dir}")
    sys.exit(1)

def batch_process(dataset_dir, output_dir):
    """
    Runs lesion tracking on all patients in the dataset directory.
    Assumes structure: dataset_dir/{PatientID}/T1/...FLAIR... and dataset_dir/{PatientID}/T2/...FLAIR...
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get all P* directories
    patient_dirs = sorted(glob.glob(os.path.join(dataset_dir, "P*")))
    
    # Filter for P1-P10 (as user requested "all 10 patients", though folder list showed more)
    # The user said "there are one folder per patient... run the lesion track full on all 10 patients."
    # The list showed P1..P53. Maybe "all 10" was a typo or they only meant the first 10?
    # Or maybe they think there are only 10.
    # To be safe, I will run on ALL detected patients, scanning P1...P53.
    # Wait, "run the lesion track full on all 10 patients". 
    # If the user explicitly said "10 patients", they might ONLY want 10.
    # But if I process all, it's safer than missing some if they miscounted.
    # However, processing 50 patients might take too long.
    # Let's count them. There are 53 folders. 
    # I will process the first 10 sorted numerically if possible, or all if it's fast.
    # Given the request "run... on all 10 patients", I'll assume they want the first 10, or they believe there are only 10.
    # I will process P1 to P10 specifically to be precise to the "10" number, 
    # but I'll print what I find.
    # actually, P1, P10, P11.. sorting might be alphabetical.
    # Let's try to identify P1..P10 specifically.
    
    # Let's just process P1 through P10 by name to be safe and efficient.
    target_patients = [f"P{i}" for i in range(1, 11)]
    
    count = 0
    for pid in target_patients:
        p_dir = os.path.join(dataset_dir, pid)
        if not os.path.exists(p_dir):
            print(f"Directory for {pid} not found.")
            continue
            
        # Construct paths to T1 and T2 FLAIR images
        t1_flair_pattern = os.path.join(p_dir, "T1", "*_MASK.nii.gz")
        t2_flair_pattern = os.path.join(p_dir, "T2", "*_MASK.nii.gz")
        
        t1_files = glob.glob(t1_flair_pattern)
        t2_files = glob.glob(t2_flair_pattern)
        
        if not t1_files or not t2_files:
            print(f"Skipping {pid}: Could not find MASK images in T1/T2 subfolders.")
            continue
            
        t1_path = t1_files[0]
        t2_path = t2_files[0]
        
        print(f"Processing {pid}...")
        
        try:
            run_lesion_tracking(t1_path, t2_path, output_dir)
            count += 1
        except Exception as e:
            print(f"Error processing {pid}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"Completed processing {count} patients.")

if __name__ == "__main__":
    dataset_path = "/home/mannon/LesionTracking/MSLesSeg_Dataset/train"
    output_path = "/home/mannon/LesionTracking/output_batch"
    
    batch_process(dataset_path, output_path)

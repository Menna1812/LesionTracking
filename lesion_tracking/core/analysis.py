import numpy as np
import pandas as pd
from collections import defaultdict

def compute_lesion_sizes(labeled_volume):
    """
    Computes the size (voxel count) of each lesion.
    
    Args:
        labeled_volume (np.ndarray): Labeled volume where each lesion has a unique integer ID.
        
    Returns:
        dict: {lesion_id: voxel_count}
    """
    unique_labels, counts = np.unique(labeled_volume, return_counts=True)
    size_dict = dict(zip(unique_labels, counts))
    if 0 in size_dict:
        del size_dict[0] # remove background
    return size_dict

def compute_overlap_map(labels_A, labels_B):
    """
    Computes intersection voxel counts between lesions in A and B.
    
    Args:
        labels_A (np.ndarray): Baseline labeled volume.
        labels_B (np.ndarray): Follow-up labeled volume.
        
    Returns:
        dict: {(A_id, B_id): intersection_voxel_count}
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
        overlap (dict): Overlap map {(A, B): count}
        size_A (dict): Sizes of lesions in A
        size_B (dict): Sizes of lesions in B
        tau (float): Overlap threshold score = intersection / min(size_A, size_B)
        
    Returns:
        dict: {A_id: [(B_id, intersection), ...]} for true splits (>=2 children)
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
        overlap (dict): Overlap map
        size_A (dict): Sizes in A
        size_B (dict): Sizes in B
        tau (float): Threshold
        
    Returns:
        dict: {B_id: [(A_id, intersection), ...]} for true merges (>=2 parents)
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
    
    # STEP 1: Handle SPLITS
    for A, children in splits.items():
        # Sum volumes of all split fragments in T2 for the total new volume
        volume = sum(size_2[B] for B, _ in children)
        
        mapping = assign_split_ids(A, children)
        
        for B, final_id in mapping.items():
            labels_B_final[labeled_T2 == B] = final_id
            processed_B.add(B)
            
        volumes[A] = volume
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
            next_new_id += 1
            
    return volumes, status, labels_B_final

def build_lesion_summary_tables(size_1, final_volumes, overlap, splits, merges, labeled_T1, labeled_T2, status):
    """
    Builds the summary tables for baseline and follow-up.
    
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
            "Baseline Volume (voxels)": size_1.get(A, 0),
            "Status": status.get(A, "Absent")
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
        
        followup_rows.append({
            "Lesion ID": lesion_id,
            "Follow-up Volume (voxels)": vol,
            "Status": stat
        })
        
    followup_table = pd.DataFrame(followup_rows)
    return baseline_table, followup_table

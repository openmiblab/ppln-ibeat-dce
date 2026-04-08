import os
import sys
import numpy as np
import napari

# =================================================================================
# STAGE 3: AIF CALCULATION (INTERACTIVE)
# =================================================================================

def calculate_aif(data_3d, mask_2d):
    """Calculates the Arterial Input Function (AIF) curve from the selected mask."""
    aif_curve = []
    for t in range(data_3d.shape[-1]):
        frame = data_3d[:, :, t]
        aif_curve.append(np.mean(frame[mask_2d == 1]))
    return np.array(aif_curve)

def process_stage_3(stage_2_dir, stage_3_dir):
    print(f"\n[STAGE 3] Starting Interactive AIF Extraction...")
    
    patient_folders = [f for f in os.listdir(stage_2_dir) if os.path.isdir(os.path.join(stage_2_dir, f))]
    
    if not patient_folders:
        print(f"[!] No patient folders found in {stage_2_dir}")
        return

    for patient in patient_folders:
        print(f"\n==================================================")
        print(f"Loading Patient: {patient}")
        
        patient_in_dir = os.path.join(stage_2_dir, patient)
        
        # --- DEEP SEARCH ---
        valid_data_folders = []
        
        print("  -> Searching deep inside all subfolders for data...")
        for root, dirs, files in os.walk(patient_in_dir):
            if 'average_map_slice_0.npy' in files and 'raw_data_slice_0.npy' in files:
                valid_data_folders.append(root)
        
        if not valid_data_folders:
            print(f"  [-] Missing data for {patient}. Checked all subfolders. Skipping...")
            continue
            
        print(f"  -> Found {len(valid_data_folders)} scan(s) for {patient}!")

        for idx, data_folder in enumerate(valid_data_folders):
            if len(valid_data_folders) > 1:
                print(f"\n  --- Processing Scan {idx + 1} of {len(valid_data_folders)} for {patient} ---")
                
            avg_map_path = os.path.join(data_folder, 'average_map_slice_0.npy')
            raw_data_path = os.path.join(data_folder, 'raw_data_slice_0.npy')
            
            print(f"  -> Loading from: {data_folder}")
            average_map = np.load(avg_map_path)
            raw_data = np.load(raw_data_path)

            print("  >>> ACTION REQUIRED <<<")
            print("  1. Napari is opening...")
            print("  2. Draw your AIF mask using the paint tool.")
            print("  3. CLOSE the Napari window when finished to save and continue.")
            
            # Open Napari
            viewer = napari.Viewer()
            viewer.add_image(average_map, name="Average Map")
            labels = viewer.add_labels(np.zeros_like(average_map, dtype=np.uint8), name="AIF_MASK")
            labels.mode = "paint"
            labels.brush_size = 5
            napari.run() 

            mask = (labels.data > 0).astype(np.uint8)
            
            if np.sum(mask) == 0:
                print("  [!] Warning: No mask was drawn! Skipping calculation for this scan.")
                continue
                
            print("  -> Calculating AIF curve...")
            aif_values = calculate_aif(raw_data, mask)

            # --- THE FOLDER MIRRORING UPGRADE ---
            # 1. Figure out exactly where this data folder sits relative to the main Stage 2 directory
            relative_subpath = os.path.relpath(data_folder, stage_2_dir)
            
            # 2. Recreate that exact same structural path inside the Stage 3 directory
            specific_out_dir = os.path.join(stage_3_dir, relative_subpath)
            os.makedirs(specific_out_dir, exist_ok=True)
            
            # 3. Save it perfectly with a standard name
            out_path = os.path.join(specific_out_dir, 'aif_values.npy')
            np.save(out_path, aif_values)
            
            print(f"  -> SUCCESS! Saved to: {out_path}")

    print("\n[STAGE 3] All available patients processed!")

if __name__ == "__main__":
    LOCAL_STAGE_2_FOLDER = r"X:\abdominal_imaging\Shared\ibeat_dce\results\stage_2"
    LOCAL_STAGE_3_FOLDER = r"X:\abdominal_imaging\Shared\ibeat_dce\results\stage_3"
    
    process_stage_3(LOCAL_STAGE_2_FOLDER, LOCAL_STAGE_3_FOLDER)
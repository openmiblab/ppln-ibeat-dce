import os
import pydicom
import numpy as np

# =================================================================================
# STAGE 2: 4D RECONSTRUCTION & DESCRIPTIVE MAPS (SHEFFIELD FIX)
# =================================================================================

def process_sheffield_dicoms(dicom_paths):
    """
    Takes a massive list of scrambled DICOM paths (from 140 different folders), 
    reads them, sorts them by Slice Location and Time, and builds the 4D array.
    """
    print("  -> Reading DICOM headers for sorting...")
    slices_dict = {}
    
    for path in dicom_paths:
        try:
            # Read just the header first to save RAM while sorting
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            
            # Group by physical slice location (rounded to avoid floating point errors)
            slice_loc = round(float(ds.SliceLocation), 2)
            
            # We need a temporal marker to sort the frames. 
            # AcquisitionTime or InstanceNumber usually works best for DCE.
            time_marker = ds.AcquisitionTime if 'AcquisitionTime' in ds else ds.InstanceNumber
            
            if slice_loc not in slices_dict:
                slices_dict[slice_loc] = []
                
            slices_dict[slice_loc].append({'path': path, 'time': float(time_marker)})
            
        except Exception as e:
            # Skip non-DICOM files or corrupted headers
            continue

    if not slices_dict:
        print("  [!] Error: No valid DICOMs found or missing SliceLocation data.")
        return None, None

    # Sort the physical slices from bottom to top
    sorted_slice_locs = sorted(slices_dict.keys())
    
    # Pre-allocate the 4D array (x, y, slices, time) to save memory
    # We grab the first file to get the image dimensions
    sample_ds = pydicom.dcmread(slices_dict[sorted_slice_locs[0]][0]['path'])
    rows, cols = sample_ds.Rows, sample_ds.Columns
    num_slices = len(sorted_slice_locs)
    num_frames = len(slices_dict[sorted_slice_locs[0]]) # Assuming all slices have same frames

    print(f"  -> Rebuilding 4D Array: {rows}x{cols} | {num_slices} Slices | {num_frames} Frames")
    data_4d = np.zeros((rows, cols, num_slices, num_frames), dtype=np.float32)

    # Fill the array
    for z, loc in enumerate(sorted_slice_locs):
        # Sort the frames for this specific slice chronologically
        sorted_frames = sorted(slices_dict[loc], key=lambda k: k['time'])
        
        for t, frame_data in enumerate(sorted_frames):
            ds = pydicom.dcmread(frame_data['path'])
            data_4d[:, :, z, t] = ds.pixel_array

    return data_4d, sorted_slice_locs

def process_stage_2(stage_1_dir, stage_2_dir):
    print(f"\n[STAGE 2] Starting Sheffield Targeted Reconstruction...")
        
    patient_folders = [f for f in os.listdir(stage_1_dir) if os.path.isdir(os.path.join(stage_1_dir, f))]
    
    for patient in patient_folders:
        
        # --- TARGET ONLY SHEFFIELD ---
        if "sheffield" not in patient.lower():
            continue

        print(f"\n==================================================")
        print(f"Processing Sheffield Patient: {patient}")
        
        patient_in_dir = os.path.join(stage_1_dir, patient)
        
        # --- 1. FLATTEN THE FOLDER STRUCTURE ---
        all_dicom_paths = []
        for root, dirs, files in os.walk(patient_in_dir):
            for file in files:
                # Assuming DICOMs either have .dcm, .IMA, or no extension
                # Adjust this if your Sheffield files have a specific naming convention
                if file.endswith('.dcm') or file.endswith('.IMA') or '.' not in file:
                    all_dicom_paths.append(os.path.join(root, file))
                    
        print(f"  -> Found {len(all_dicom_paths)} total files across all frames.")
        
        if len(all_dicom_paths) == 0:
            print("  [!] No files found, skipping...")
            continue

        # --- 2. REBUILD AND SORT THE 4D ARRAY ---
        data_4d, slice_locations = process_sheffield_dicoms(all_dicom_paths)
        
        if data_4d is None:
            continue
            
        # --- 3. CALCULATE DESCRIPTIVE MAPS & SAVE BY SLICE ---
        # Your Stage 3 expects folders with raw_data_slice_0.npy and average_map_slice_0.npy
        
        patient_out_dir = os.path.join(stage_2_dir, patient, 'processed_scan')
        os.makedirs(patient_out_dir, exist_ok=True)
        
        # Loop through each slice to save it out for Stage 3
        for z in range(data_4d.shape[2]):
            
            # Get the raw time-series for just this one slice (Shape: X, Y, Time)
            raw_data_slice = data_4d[:, :, z, :]
            
            # Calculate the descriptive average map for this slice across all time frames
            average_map_slice = np.mean(raw_data_slice, axis=-1)
            
            # You can add other descriptive maps here if needed (like standard deviation, max enhancement, etc.)
            
            # Save them out
            raw_out_path = os.path.join(patient_out_dir, f'raw_data_slice_{z}.npy')
            avg_out_path = os.path.join(patient_out_dir, f'average_map_slice_{z}.npy')
            
            np.save(raw_out_path, raw_data_slice)
            np.save(avg_out_path, average_map_slice)
            
        print(f"  -> SUCCESS! Saved {data_4d.shape[2]} slices to: {patient_out_dir}")

    print("\n[STAGE 2] All Sheffield patients processed!")

if __name__ == "__main__":
    # --- UPDATE THESE PATHS BEFORE RUNNING ON THE HPC ---
    HPC_STAGE_1_FOLDER = "/mnt/parscratch/users/eia21frd/data/ibeat_dce/stage_1_download/BEAt-DKD-WP4-Sheffield"
    HPC_STAGE_2_FOLDER = "/mnt/parscratch/users/eia21frd/build/stage_2_compute_descriptivemaps"
    
    process_stage_2(HPC_STAGE_1_FOLDER, HPC_STAGE_2_FOLDER)
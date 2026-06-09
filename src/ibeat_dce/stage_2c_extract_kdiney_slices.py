import os
import numpy as np
import pydicom
from collections import Counter

def load_single_slice_time_series(folder_path, slice_idx):
    dicom_headers = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'AcquisitionTime'):
                    t_str = str(ds.AcquisitionTime).strip()
                    if '.' not in t_str: t_str += '.0'
                    t_str = t_str.zfill(8) 
                    hh, mm, ss = float(t_str[:2])*3600.0, float(t_str[2:4])*60.0, float(t_str[4:])
                    dicom_headers.append({
                        'time_stamp': hh + mm + ss,
                        'z': round(float(ds.ImagePositionPatient[2]), 2),
                        'path': filepath
                    })
            except Exception:
                continue

    if not dicom_headers: return None
    unique_z = sorted(list(set(d['z'] for d in dicom_headers)))
    if slice_idx < 0 or slice_idx >= len(unique_z): return None
        
    target_z = unique_z[slice_idx]
    target_files = sorted([d for d in dicom_headers if d['z'] == target_z], key=lambda x: x['time_stamp']) 
    
    ref_ds = pydicom.dcmread(target_files[0]['path'])
    rows, cols = int(ref_ds.Rows), int(ref_ds.Columns)
    
    pixel_array = np.zeros((rows, cols, len(target_files)), dtype=np.float64)
    for i, item in enumerate(target_files):
        ds = pydicom.dcmread(item['path'])
        img_data = ds.pixel_array.astype(np.float64)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img_data = (img_data * ds.RescaleSlope) + ds.RescaleIntercept
        pixel_array[:, :, i] = img_data

    return pixel_array

def get_true_dimensions(folder_path):
    z_positions = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            try:
                ds = pydicom.dcmread(os.path.join(root, filename), stop_before_pixels=True)
                if hasattr(ds, 'ImagePositionPatient'):
                    z_positions.append(round(float(ds.ImagePositionPatient[2]), 2))
            except: continue
    if not z_positions: return 0, 0
    z_counts = Counter(z_positions)
    return len(z_counts), max(z_counts.values())

if __name__ == "__main__":
    stage_1_dir = "/mnt/parscratch/users/eia21frd/data/ibeat_dce/stage_1_download"
    stage_2_dir = "/mnt/parscratch/users/eia21frd/build/stage_2_compute_descriptivemaps"

    print("Scanning for DICOM series...")
    dicom_folders = sorted([root for root, _, files in os.walk(stage_1_dir) if any(f.lower().endswith('.dcm') for f in files)])
    
    for dicom_folder in dicom_folders:
        relative_path = os.path.relpath(dicom_folder, stage_1_dir)
        results_folder = os.path.join(stage_2_dir, relative_path)
        
        # FORCE CREATE the directory so it never skips due to missing folders
        os.makedirs(results_folder, exist_ok=True)

        print(f"\n[+] EXTRACTING DATA FOR: {relative_path}")
        total_slices, true_max_times = get_true_dimensions(dicom_folder)
        
        if total_slices == 0:
            continue
            
        print(f"    -> Found {total_slices} total slices. Generating .npy files...")
        
        # Extract ALL slices from 0 to the maximum
        for z in range(total_slices): 
            pixel_array_z = load_single_slice_time_series(dicom_folder, slice_idx=z)
            if pixel_array_z is None: continue
            
            t_len = min(true_max_times, pixel_array_z.shape[2])
            final_array = pixel_array_z[:, :, :t_len]
            
            avg_map = np.mean(final_array, axis=-1)
            
            np.save(os.path.join(results_folder, f'average_map_slice_{z}.npy'), avg_map)
            np.save(os.path.join(results_folder, f'raw_data_slice_{z}.npy'), final_array)
            print(f"       Saved slice {z}")
            
    print("\nData extraction complete! You can now run your Stage 4 job.")
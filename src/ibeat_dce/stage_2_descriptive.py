import os
import math
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio_ffmpeg
import matplotlib as mpl
from collections import Counter

# Point matplotlib to the embedded ffmpeg for video creation
mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# =================================================================================
# 1. THE ROBUST LOADER 
# =================================================================================

def load_single_slice_time_series(folder_path, slice_idx):
    """Exact metadata extraction. Pre-allocates with np.zeros."""
    dicom_headers = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'AcquisitionTime'):
                    t_str = str(ds.AcquisitionTime).strip()
                    if '.' not in t_str:
                        t_str += '.0'
                    t_str = t_str.zfill(8) 
                    
                    hh = float(t_str[:2]) * 3600.0
                    mm = float(t_str[2:4]) * 60.0
                    ss = float(t_str[4:])
                    t_val = hh + mm + ss

                    dicom_headers.append({
                        'time_stamp': t_val,
                        'z': round(float(ds.ImagePositionPatient[2]), 2),
                        'path': filepath
                    })
            except Exception:
                continue

    if not dicom_headers:
        raise ValueError(f"No valid DICOM files found in {folder_path}")

    unique_z_positions = sorted(list(set(d['z'] for d in dicom_headers)))
    if slice_idx < 0 or slice_idx >= len(unique_z_positions):
        raise IndexError(f"Slice index {slice_idx} is out of bounds.")
        
    target_z = unique_z_positions[slice_idx]
    target_files = [d for d in dicom_headers if d['z'] == target_z]
    target_files.sort(key=lambda x: x['time_stamp']) 

    start_time = target_files[0]['time_stamp']
    time_points = np.array([d['time_stamp'] - start_time for d in target_files])
    
    num_times = len(target_files)
    ref_ds = pydicom.dcmread(target_files[0]['path'])
    rows, cols = int(ref_ds.Rows), int(ref_ds.Columns)
    
    # Pre-allocate with np.zeros. No stacking!
    pixel_array = np.zeros((rows, cols, num_times), dtype=np.float64)
    
    for i, item in enumerate(target_files):
        ds = pydicom.dcmread(item['path'])
        img_data = ds.pixel_array.astype(np.float64)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img_data = (img_data * ds.RescaleSlope) + ds.RescaleIntercept
        pixel_array[:, :, i] = img_data

    return pixel_array, time_points, len(unique_z_positions)


def describe(pixel_array):
    return np.mean(pixel_array, axis=-1)

# =================================================================================
# 2. VIDEO GENERATION FUNCTIONS
# =================================================================================

def save_uncorrected_mp4(data, filename, title='Uncorrected Data', fps=3):
    # Use 99th percentile to ignore extreme bright artifact pixels
    vmax = np.percentile(data, 99) 
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    im = ax.imshow(data[:, :, 0], cmap='gray', vmin=0, vmax=vmax)
    
    def update(frame_idx):
        im.set_array(data[:, :, frame_idx])
        return [im]
        
    anim = animation.FuncAnimation(fig, update, frames=data.shape[-1], blit=True)
    anim.save(filename, writer='ffmpeg', fps=fps)
    plt.close(fig)

def save_uncorrected_grid_mp4(data_4d, filename, title='Uncorrected DCE - All Slices', fps=3):
    num_slices = data_4d.shape[2]
    num_times = data_4d.shape[3]
    
    grid_cols = math.ceil(math.sqrt(num_slices))
    grid_rows = math.ceil(num_slices / grid_cols)
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
    fig.suptitle(title, fontsize=16)
    
    if num_slices == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
        
    ims = []
    
    # Use 99th percentile of the ENTIRE 4D volume for consistent contrast
    vmax = np.percentile(data_4d, 99)
    
    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        ax.axis('off') 
        
        if i < num_slices:
            ax.set_title(f"Slice {i}")
            im = ax.imshow(data_4d[:, :, i, 0], cmap='gray', vmin=0, vmax=vmax)
            ims.append(im)
            
    plt.tight_layout()
    
    def update(frame_idx):
        for i, im in enumerate(ims):
            im.set_array(data_4d[:, :, i, frame_idx])
        return ims
        
    anim = animation.FuncAnimation(fig, update, frames=num_times, blit=True)
    anim.save(filename, writer='ffmpeg', fps=fps)
    plt.close(fig)

# =================================================================================
# 3. CORE COMPUTE FUNCTION
# =================================================================================

def get_true_dimensions(folder_path):
    """Safely checks all DICOMs to find the absolute maximum number of frames any slice has."""
    z_positions = []
    
    for root, _, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                if hasattr(ds, 'ImagePositionPatient'):
                    z_positions.append(round(float(ds.ImagePositionPatient[2]), 2))
            except Exception:
                continue
                
    if not z_positions:
        return 0, 0
        
    # Count how many frames exist for each Z slice
    z_counts = Counter(z_positions)
    total_slices = len(z_counts)
    true_max_times = max(z_counts.values()) # Finds the longest slice!
    
    return total_slices, true_max_times

def compute_maps_for_patient(dicom_folder, results_folder):
    expected_grid_mp4 = os.path.join(results_folder, 'uncorrected_motion_grid.mp4')
    
    if os.path.exists(expected_grid_mp4):
        print(f"[-] SKIPPING: Already processed.")
        return True 

    os.makedirs(results_folder, exist_ok=True)

    try:
        print(f"    -> Scanning DICOMs to find true dimensions...")
        total_slices, true_max_times = get_true_dimensions(dicom_folder)
        
        if total_slices == 0:
            print(f"    [!] WARNING: Found 0 valid DICOM slices.")
            return False

        # Load Slice 0 just to get the X and Y pixel dimensions (rows, cols)
        slice_0_data, _, _ = load_single_slice_time_series(dicom_folder, slice_idx=0)
        rows, cols = slice_0_data.shape[0], slice_0_data.shape[1]
        
        print(f"    -> Found {total_slices} slices. True Max Frames: {true_max_times}")
        print(f"    -> Expected 4D shape: ({rows}, {cols}, {total_slices}, {true_max_times})")
        
        # Build the 4D array using the TRUE maximum time length
        data_4d = np.zeros((rows, cols, total_slices, true_max_times), dtype=np.float64)
        
        # Fill the array slice by slice
        for z in range(total_slices):
            pixel_array_z, _, _ = load_single_slice_time_series(dicom_folder, slice_idx=z)
            
            # Protect against dropped frames: only fill up to whatever time points exist
            t_len = min(true_max_times, pixel_array_z.shape[2])
            data_4d[:, :, z, :t_len] = pixel_array_z[:, :, :t_len]

        print("    -> Extracting Slice 0...")
        average_map_0 = describe(data_4d[:, :, 0, :])
        np.save(os.path.join(results_folder, 'average_map_slice_0.npy'), average_map_0)
        np.save(os.path.join(results_folder, 'raw_data_slice_0.npy'), data_4d[:, :, 0, :])

        print("    -> Generating Full Z-Stack Animation Grid...")
        save_uncorrected_grid_mp4(data_4d, expected_grid_mp4, title='Uncorrected DCE Slices')
        
        print("    -> Generating individual MP4 files for each slice...")
        for z in range(total_slices):
            slice_data = data_4d[:, :, z, :]  
            slice_mp4_path = os.path.join(results_folder, f'uncorrected_motion_slice_{z}.mp4')
            save_uncorrected_mp4(slice_data, slice_mp4_path, title=f'Uncorrected Slice {z}')
        
        return True
        
    except Exception as e:
        print(f"    [!] ERROR processing: {e}")
        return False

# =================================================================================
# 4. MAIN RUNNER 
# =================================================================================
if __name__ == "__main__":
    stage_1_dir = "/mnt/parscratch/users/eia21frd/data/ibeat_dce/stage_1_download"
    stage_2_dir = "/mnt/parscratch/users/eia21frd/build/stage_2_compute_descriptivemaps"
    
    n_max = 10      
    processed_count = 0

    print(f"\nScanning {stage_1_dir} for DICOM series...")
    dicom_series_folders = []
    
    # Updated logic: Search for any folder that actually contains .dcm files
    for root, dirs, files in os.walk(stage_1_dir):
        if any(f.lower().endswith('.dcm') for f in files):
            dicom_series_folders.append(root)
            
    dicom_series_folders.sort()
    print(f"Found {len(dicom_series_folders)} potential patient series to process.")

    for dicom_folder in dicom_series_folders:
        if n_max is not None and processed_count >= n_max:
            print(f"\n[STOPPING] Reached n_max limit of {n_max} successful cases.")
            break

        # This will preserve the deep folder structure in your results folder
        relative_path = os.path.relpath(dicom_folder, stage_1_dir)
        results_folder = os.path.join(stage_2_dir, relative_path)
        
        print(f"\n[+] PROCESSING: {relative_path}")
        
        success = compute_maps_for_patient(dicom_folder, results_folder)
        
        if success:
            processed_count += 1
            print(f"    -> Patient processing complete. ({processed_count}/{n_max})")

    print(f"\nStage 2 compute complete! Processed {processed_count} series.")
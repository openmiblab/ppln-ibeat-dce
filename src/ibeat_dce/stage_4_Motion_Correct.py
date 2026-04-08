import os
import time
import math
import numpy as np
import pydicom
import mdreg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio_ffmpeg
import matplotlib as mpl

# Set the path to ffmpeg for the MP4 exporter
mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def get_total_slices(folder_path):
    """Scans the DICOM folder to find out exactly how many unique Z-slices exist."""
    unique_z = set()
    for root, _, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                if hasattr(ds, 'ImagePositionPatient'):
                    unique_z.add(float(ds.ImagePositionPatient[2]))
            except Exception:
                pass
    return len(unique_z)

def load_single_slice_time_series(folder_path, slice_idx):
    dicom_headers = []

    # 1. EXACT METADATA EXTRACTION
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
                        'z': float(ds.ImagePositionPatient[2]),
                        'path': filepath
                    })
            except Exception:
                continue

    if not dicom_headers:
        raise ValueError(f"No valid DICOM files found in {folder_path}")

    # 2. IDENTIFY TARGET Z
    unique_z_positions = sorted(list(set(d['z'] for d in dicom_headers)))
    if slice_idx < 0 or slice_idx >= len(unique_z_positions):
        raise IndexError(f"Slice index {slice_idx} is out of bounds.")
        
    target_z = unique_z_positions[slice_idx]
    target_files = [d for d in dicom_headers if d['z'] == target_z]
    target_files.sort(key=lambda x: x['time_stamp']) 

    # 3. CALCULATE RELATIVE SECONDS
    start_time = target_files[0]['time_stamp']
    time_points = np.array([d['time_stamp'] - start_time for d in target_files])
    
    # 4. BUILD PIXEL ARRAY & EXTRACT PIXEL SPACING
    num_times = len(target_files)
    ref_ds = pydicom.dcmread(target_files[0]['path'])
    rows, cols = int(ref_ds.Rows), int(ref_ds.Columns)
    pixel_array = np.zeros((rows, cols, num_times), dtype=np.float64)
    
    if hasattr(ref_ds, 'PixelSpacing'):
        pixel_spacing = [float(x) for x in ref_ds.PixelSpacing]
    else:
        pixel_spacing = [1.0, 1.0]

    for i, item in enumerate(target_files):
        ds = pydicom.dcmread(item['path'])
        img_data = ds.pixel_array.astype(np.float64)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img_data = (img_data * ds.RescaleSlope) + ds.RescaleIntercept
        pixel_array[:, :, i] = img_data

    return pixel_array, time_points, pixel_spacing

def run_mdr_motion_correction(pixel_array, acq_times, aif_values, baseline_frames=18, results_dir='MDR_Temp_Results', coreg_options=None):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    t = time.time()
    coreg, fit, transfo, pars = mdreg.fit(
        pixel_array,
        fit_image={
            'func': mdreg.fit_2cm_lin,
            'time': acq_times,
            'aif': aif_values,
            'baseline': baseline_frames,
        },
        fit_coreg=coreg_options, 
        maxit=3,
        path=results_dir,
        verbose=1, 
    )
    print(f"MDR Computation time: {round(time.time() - t)} seconds.")
    return coreg, fit

def save_grid_mp4(dict_coreg_data, dict_uncorrected_data, filename, title='Motion Corrected Slices', fps=10):
    """
    Takes a dictionary of corrected slices and outputs a single grid MP4.
    """
    slices = sorted(list(dict_coreg_data.keys()))
    num_slices = len(slices)
    
    if num_slices == 0:
        print("  [!] No valid slices to animate.")
        return

    # 1. Calculate Grid Dimensions dynamically
    cols = math.ceil(math.sqrt(num_slices))
    rows = math.ceil(num_slices / cols)

    # 2. Calculate a global contrast (vmax) so all slices look uniform
    max_vals = [np.max(data[..., 0]) for data in dict_uncorrected_data.values()]
    global_vmax = 0.9 * max(max_vals)

    # 3. Setup the Matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(title, fontsize=16)
    
    # Flatten axes array for easy looping (and handle if there's only 1 slice)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    images = []
    
    # Squeeze arrays to 3D and gather the first frame for the grid
    for s in slices:
        if dict_coreg_data[s].ndim == 4:
            dict_coreg_data[s] = np.squeeze(dict_coreg_data[s])

    num_frames = dict_coreg_data[slices[0]].shape[-1]

    # Populate the initial grid
    for idx, ax in enumerate(axes):
        ax.axis('off') # Hide axes ticks/borders
        if idx < num_slices:
            s = slices[idx]
            data = dict_coreg_data[s]
            im = ax.imshow(data[:, :, 0], cmap='gray', vmin=0, vmax=global_vmax)
            ax.set_title(f"Slice {s}", color='white', backgroundcolor='black')
            images.append((data, im))

    plt.tight_layout()

    # 4. Define the animation update function
    def update(frame_idx):
        ims = []
        for data, im in images:
            im.set_array(data[:, :, frame_idx])
            ims.append(im)
        return ims

    # 5. Render and save
    anim = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)
    anim.save(filename, writer='ffmpeg', fps=fps)
    plt.close(fig)

# =================================================================================
# MAIN STAGE 4 PIPELINE
# =================================================================================

def process_stage_4(stage_1_base, stage_3_base, stage_4_base):
    print("\n[STAGE 4] Starting Motion Correction (mdreg)...")
    
    scans_to_process = []
    for root, dirs, files in os.walk(stage_3_base):
        if 'aif_values.npy' in files:
            scans_to_process.append(root)
            
    if not scans_to_process:
        print(f"[!] No AIF files found in {stage_3_base}. Please run Stage 3 first.")
        return

    print(f"-> Found {len(scans_to_process)} scan(s) ready for motion correction.")

    for idx, aif_folder in enumerate(scans_to_process):
        print(f"\n==================================================")
        print(f"Processing Scan {idx + 1} of {len(scans_to_process)}")
        
        # 1. Mirror paths
        relative_path = os.path.relpath(aif_folder, stage_3_base)
        dicom_folder = os.path.join(stage_1_base, relative_path)
        results_folder = os.path.join(stage_4_base, relative_path)
        
        if not os.path.exists(dicom_folder):
            print(f"  [-] Warning: Found AIF, but missing Stage 1 DICOMs at {dicom_folder}. Skipping...")
            continue
            
        os.makedirs(results_folder, exist_ok=True)
        print(f"  -> Input DICOMs: {dicom_folder}")
        print(f"  -> Output Folder: {results_folder}")

        # 2. Load the AIF
        aif_path = os.path.join(aif_folder, 'aif_values.npy')
        print("  -> Loading AIF from Stage 3...")
        aif_values = np.load(aif_path)
        
        # 3. Detect slices and Run MDR
        total_slices = get_total_slices(dicom_folder)
        print(f"  -> Total slices detected: {total_slices}. Processing slices 1 to {total_slices - 1}...")

        # Dictionaries to hold data for the grid video
        scan_corrected_data = {}
        scan_uncorrected_data = {}

        for current_slice in range(1, total_slices):
            print(f"\n  --- PROCESSING SLICE {current_slice} ---")
            pixel_array_s, acq_times_s, pixel_spacing_s = load_single_slice_time_series(dicom_folder, slice_idx=current_slice)

            # ---> THE SAFETY NET: Check for time mismatch <---
            if len(acq_times_s) != len(aif_values):
                print(f"  [!] SKIPPING SLICE {current_slice}: Frame mismatch! Slice has {len(acq_times_s)} frames, but AIF has {len(aif_values)}.")
                continue

            my_coreg_options = {
                'package': 'elastix', 
                'spacing': pixel_spacing_s, 
                'FinalGridSpacingInPhysicalUnits': 25.0
            }

            temp_mdr_dir = os.path.join(results_folder, f'MDR_Temp_Slice_{current_slice}')
            
            corrected_s, fit_s = run_mdr_motion_correction(
                pixel_array=pixel_array_s, 
                acq_times=acq_times_s, 
                aif_values=aif_values, 
                baseline_frames=10,
                results_dir=temp_mdr_dir,
                coreg_options=my_coreg_options 
            )

            # Save the individual raw Numpy arrays (always good for data integrity)
            np.save(os.path.join(results_folder, f'moco_slice_{current_slice}.npz'), corrected_s)
            
            # Store data in memory for our grid video later
            scan_corrected_data[current_slice] = corrected_s
            scan_uncorrected_data[current_slice] = pixel_array_s

        # 4. Generate the final Grid Video once all slices are done
        if scan_corrected_data:
            print("\n  -> Generating unified Grid Video...")
            grid_mp4_path = os.path.join(results_folder, 'all_slices_moco_grid.mp4')
            save_grid_mp4(
                dict_coreg_data=scan_corrected_data, 
                dict_uncorrected_data=scan_uncorrected_data, 
                filename=grid_mp4_path, 
                title=f'Motion Corrected Slices'
            )
            print(f"  -> SUCCESS: Saved Grid Video to {grid_mp4_path}")

        print(f"  -> Scan Complete! Results saved perfectly.")

    print("\n[STAGE 4] All available scans processed successfully!")


if __name__ == "__main__":
    # Point these to your cluster paths!
    STAGE_1_FOLDER = r"X:\abdominal_imaging\Shared\ibeat_dce\data\stage_1_download"
    STAGE_3_FOLDER = r"X:\abdominal_imaging\Shared\ibeat_dce\results\stage_3"
    STAGE_4_FOLDER = r"X:\abdominal_imaging\Shared\ibeat_dce\results\stage_4"
    
    process_stage_4(STAGE_1_FOLDER, STAGE_3_FOLDER, STAGE_4_FOLDER)
import os
import time
import math
import glob
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

def get_pixel_spacing(dicom_folder):
    """Grabs pixel spacing from just the first valid DICOM to avoid parsing thousands of files."""
    for root, _, files in os.walk(dicom_folder):
        for filename in files:
            try:
                ds = pydicom.dcmread(os.path.join(root, filename), stop_before_pixels=True)
                if hasattr(ds, 'PixelSpacing'):
                    return [float(x) for x in ds.PixelSpacing]
            except Exception:
                continue
    return [1.0, 1.0] # Fallback if none found

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
    print(f"    MDR Computation time: {round(time.time() - t)} seconds.")
    return coreg, fit

def save_grid_mp4(dict_coreg_data, dict_uncorrected_data, filename, title='Motion Corrected Slices', fps=10):
    slices = sorted(list(dict_coreg_data.keys()))
    num_slices = len(slices)
    
    if num_slices == 0:
        print("  [!] No valid slices to animate.")
        return

    cols = math.ceil(math.sqrt(num_slices))
    rows = math.ceil(num_slices / cols)

    max_vals = [np.max(data[..., 0]) for data in dict_uncorrected_data.values()]
    global_vmax = 0.9 * max(max_vals)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(title, fontsize=16)
    
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    images = []
    
    for s in slices:
        if dict_coreg_data[s].ndim == 4:
            dict_coreg_data[s] = np.squeeze(dict_coreg_data[s])

    num_frames = dict_coreg_data[slices[0]].shape[-1]

    for idx, ax in enumerate(axes):
        ax.axis('off') 
        if idx < num_slices:
            s = slices[idx]
            data = dict_coreg_data[s]
            im = ax.imshow(data[:, :, 0], cmap='gray', vmin=0, vmax=global_vmax)
            ax.set_title(f"Slice {s}", color='white', backgroundcolor='black')
            images.append((data, im))

    plt.tight_layout()

    def update(frame_idx):
        ims = []
        for data, im in images:
            im.set_array(data[:, :, frame_idx])
            ims.append(im)
        return ims

    anim = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)
    anim.save(filename, writer='ffmpeg', fps=fps)
    plt.close(fig)

# =================================================================================
# MAIN STAGE 4 PIPELINE
# =================================================================================

def process_stage_4(stage_1_base, stage_2_base, stage_3_base, stage_4_base):
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
        
        # 1. Mirror paths across all stages
        relative_path = os.path.relpath(aif_folder, stage_3_base)
        dicom_folder = os.path.join(stage_1_base, relative_path)
        stage2_folder = os.path.join(stage_2_base, relative_path)
        results_folder = os.path.join(stage_4_base, relative_path)
        
        if not os.path.exists(stage2_folder):
            print(f"  [-] Warning: Missing Stage 2 data at {stage2_folder}. Skipping...")
            continue
            
        os.makedirs(results_folder, exist_ok=True)

        # 2. Load the AIF (Stage 3)
        aif_path = os.path.join(aif_folder, 'aif_values.npy')
        aif_values = np.load(aif_path)
        frame_count_aif = len(aif_values)
        
        # 3. Load & Fix Time Array (Stage 2)
        time_path = os.path.join(stage2_folder, 'acq_times.npy')
        if not os.path.exists(time_path):
            print(f"  [!] Missing acq_times.npy. Skipping...")
            continue
            
        acq_times = np.load(time_path)
        acq_times = np.sort(acq_times)
        
        if len(acq_times) > frame_count_aif * 2:
            # Compress slice times into volume times
            acq_times = np.array([np.mean(chunk) for chunk in np.array_split(acq_times, frame_count_aif)])

        # 4. Get Stage 1 Metadata (Just one file for spacing)
        pixel_spacing_s = get_pixel_spacing(dicom_folder)

        # 5. Find all Pre-computed Stage 2 Slices
        slice_files = sorted(glob.glob(os.path.join(stage2_folder, "raw_data_slice_*.npy")))
        print(f"  -> Total pre-computed slices found: {len(slice_files)}")

        scan_corrected_data = {}
        scan_uncorrected_data = {}

        for slice_path in slice_files:
            # Extract slice number from filename (e.g., raw_data_slice_0.npy -> 0)
            current_slice = int(os.path.basename(slice_path).split('_')[-1].split('.')[0])
            print(f"\n  --- PROCESSING SLICE {current_slice} ---")
            
            # Load Stage 2 pixel data
            pixel_array_s = np.load(slice_path)
            frame_count_slice = pixel_array_s.shape[-1]

            # ---> THE SAFETY NET: Ultimate Trim <---
            # Find the lowest frame count between AIF, Time, and Pixel Array
            min_len = min(len(acq_times), len(aif_values), frame_count_slice)
            
            if len(acq_times) != min_len or len(aif_values) != min_len or frame_count_slice != min_len:
                print(f"  [*] Frame mismatch detected! Trimming Time, AIF, and Array down to {min_len} frames to perfectly align.")
            
            # Trim everything to exactly match
            acq_times_trim = acq_times[:min_len]
            aif_values_trim = aif_values[:min_len]
            pixel_array_trim = pixel_array_s[:, :, :min_len]

            # Normalize time to start at 0 seconds
            acq_times_trim = acq_times_trim - acq_times_trim[0]

            my_coreg_options = {
                'package': 'elastix', 
                'spacing': pixel_spacing_s, 
                'FinalGridSpacingInPhysicalUnits': 25.0
            }

            temp_mdr_dir = os.path.join(results_folder, f'MDR_Temp_Slice_{current_slice}')
            
            corrected_s, fit_s = run_mdr_motion_correction(
                pixel_array=pixel_array_trim, 
                acq_times=acq_times_trim, 
                aif_values=aif_values_trim, 
                baseline_frames=10,
                results_dir=temp_mdr_dir,
                coreg_options=my_coreg_options 
            )

            np.save(os.path.join(results_folder, f'moco_slice_{current_slice}.npz'), corrected_s)
            scan_corrected_data[current_slice] = corrected_s
            scan_uncorrected_data[current_slice] = pixel_array_trim

        # Generate Grid Video
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

    print("\n[STAGE 4] All available scans processed successfully!")

if __name__ == "__main__":
    # Point these to your cluster paths! Make sure STAGE_2_FOLDER is mapped correctly.
    STAGE_1_FOLDER = "/mnt/parscratch/users/eia21frd/data/ibeat_dce/stage_1_download/BEAt-DKD-WP4-Exeter"
    STAGE_2_FOLDER = "/mnt/parscratch/users/eia21frd/build/stage_2_compute_descriptivemaps/BEAt-DKD-WP4-Exeter" 
    STAGE_3_FOLDER = "/mnt/parscratch/users/eia21frd/build/stage_3/BEAt-DKD-WP4-Exeter"
    STAGE_4_FOLDER = "/mnt/parscratch/users/eia21frd/build/stage_4_motioncorrected"
    
    process_stage_4(STAGE_1_FOLDER, STAGE_2_FOLDER, STAGE_3_FOLDER, STAGE_4_FOLDER)
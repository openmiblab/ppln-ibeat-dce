# =================================================================================
# STAGE 2 vs STAGE 4 KYMOGRAPH GENERATOR (ULTRA-ROBUST VERSION)
# =================================================================================
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# =================================================================================
# 1. CONFIGURATION
# =================================================================================
stage2_folder = r"X:\abdominal_imaging\Shared\ibeat_dce\results\stage_2\BEAt-DKD-WP4-Bari\Bari_Patients\iBE-1128-017\iBE-1128-017\scans\1701-DCE_kidneys_cor_oblique_fb_wet_pulse\resources\DICOM\files"
stage4_folder = r"X:\abdominal_imaging\Shared\ibeat_dce\results\stage_4\BEAt-DKD-WP4-Bari\Bari_Patients\iBE-1128-017\iBE-1128-017\scans\1701-DCE_kidneys_cor_oblique_fb_wet_pulse\resources\DICOM\files"

slice_axis = 'column' 
target_idx = 125       

# =================================================================================
# 2. SMART LOADING HELPER
# =================================================================================
def find_and_load(folder, keywords):
    """Searches a folder for files containing specific keywords and loads them."""
    all_files = os.listdir(folder)
    # Filter files that contain all keywords (case-insensitive)
    matches = [f for f in all_files if all(k.lower() in f.lower() for k in keywords)]
    
    if not matches:
        print(f"\n[ERROR] No files found in {folder} matching: {keywords}")
        print(f"[DEBUG] Available files (first 5): {all_files[:5]}")
        return None
    
    target_path = os.path.join(folder, matches[0])
    print(f"Found and loading: {matches[0]}")
    
    data = np.load(target_path, allow_pickle=True)
    # Handle .npz vs .npy content
    if not isinstance(data, np.ndarray) and hasattr(data, 'files'):
        data = data[data.files[0]]
    return data

# =================================================================================
# 3. PLOTTING FUNCTIONS
# =================================================================================
def preview_target_line(preview_img, axis, idx):
    plt.figure(figsize=(8, 8))
    plt.imshow(preview_img, cmap='gray', vmax=np.percentile(preview_img, 95))
    plt.title(f"Target Line Preview: {axis.title()} {idx}", fontsize=16, fontweight='bold')
    if axis.lower() == 'column':
        plt.axvline(x=idx, color='red', linewidth=2, linestyle='--')
    else:
        plt.axhline(y=idx, color='red', linewidth=2, linestyle='--')
    plt.axis('off')
    plt.show()  

def generate_kymographs(raw_array, corr_array, acq_times, axis, idx, save_dir):
    if axis.lower() == 'column':
        kymo_raw, kymo_corr = raw_array[:, idx, :], corr_array[:, idx, :]
        y_label = "Superior-Inferior Position"
    else:
        kymo_raw, kymo_corr = raw_array[idx, :, :], corr_array[idx, :, :]
        y_label = "Left-Right Position"

    extent = [acq_times[0], acq_times[-1], kymo_raw.shape[0], 0] if acq_times is not None else None
    display_vmax = np.percentile(kymo_raw, 90)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.imshow(kymo_raw, cmap='gray', aspect='auto', vmax=display_vmax, extent=extent)
    ax1.set_title("Stage 2: Raw Data", fontweight='bold', fontsize=16)
    
    ax2.imshow(kymo_corr, cmap='gray', aspect='auto', vmax=display_vmax, extent=extent)
    ax2.set_title("Stage 4: Motion Corrected", fontweight='bold', fontsize=16)
    ax2.set_ylabel(y_label, fontsize=14)
    ax2.set_xlabel("Time (Seconds)" if acq_times is not None else "Frames", fontsize=14)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"Kymograph_Comparison_{axis}_{idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved result to: {save_path}")

# =================================================================================
# 4. MAIN EXECUTION
# =================================================================================
if __name__ == "__main__":
    print("Scanning share drive folders...")
    
    # Use keywords to find files instead of hardcoded names
    raw_data = find_and_load(stage2_folder, ["raw", "slice_4"])
    corr_data = find_and_load(stage4_folder, ["moco", "slice_4"])
    
    if raw_data is not None and corr_data is not None:
        # Try to find aux files, but don't crash if they are missing
        avg_img = find_and_load(stage2_folder, ["average", "slice_4"])
        if avg_img is None: avg_img = np.mean(raw_data, axis=-1)
        
        acq_times = find_and_load(stage2_folder, ["acq_times"])
        
        # Run Visualization
        preview_target_line(avg_img, slice_axis, target_idx)
        generate_kymographs(raw_data, corr_data, acq_times, slice_axis, target_idx, stage4_folder)
    else:
        print("\n[CRITICAL] Missing main data arrays. Check the folder scan above.")
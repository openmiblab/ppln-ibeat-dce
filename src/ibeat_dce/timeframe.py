# =================================================================================
# 1. IMPORTS
# =================================================================================
import os
import time
import numpy as np
import napari
import pydicom
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mdreg

# =================================================================================
# 2. DATA LOADING FUNCTIONS
# =================================================================================
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
                        'z': float(ds.ImagePositionPatient[2]),
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
    pixel_array = np.zeros((rows, cols, num_times), dtype=np.float64)
    
    for i, item in enumerate(target_files):
        ds = pydicom.dcmread(item['path'])
        img_data = ds.pixel_array.astype(np.float64)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img_data = (img_data * ds.RescaleSlope) + ds.RescaleIntercept
        pixel_array[:, :, i] = img_data

    return pixel_array, time_points
    
def load_entire_time_series(folder_path):
    pixel_array_list = []
    acq_times = None 
    for z in [0]:
        pixel_array_z, acq_times = load_single_slice_time_series(folder_path, z)
        pixel_array_list.append(pixel_array_z)
    pixel_array = np.stack(pixel_array_list)
    return pixel_array, acq_times

def describe(pixel_array):
    return np.mean(pixel_array, axis=-1)

# =================================================================================
# 3. MASKING & INTENSITY FUNCTIONS
# =================================================================================
def draw_arterial_input(average_pixel_array):
    viewer = napari.Viewer()
    viewer.add_image(average_pixel_array, name="SEmax")
    labels = viewer.add_labels(np.zeros_like(average_pixel_array, dtype=np.uint8), name="AIF_MASK")
    labels.mode = "paint"
    labels.brush_size = 5
    print("\n[Action Required] Please draw the AIF mask in Napari and close the window to continue.")
    napari.run()
    return (labels.data > 0).astype(np.uint8)

def calculate_aif(full_4d_data, mask):
    aif_curve = []
    data_3d = np.squeeze(full_4d_data) 
    mask_2d = np.squeeze(mask)
    for t in range(data_3d.shape[-1]):
        frame = data_3d[:, :, t]
        aif_curve.append(np.mean(frame[mask_2d == 1]))
    return np.array(aif_curve)

def create_slice_4_average(pixel_array_s4):
    return np.mean(pixel_array_s4, axis=-1)

def draw_kidney_masks_slice_4(avg_map_s4):
    viewer = napari.Viewer(title="SLICE 4: Label 1=Left Kidney, Label 2=Right Kidney")
    viewer.add_image(avg_map_s4, name="S4_Average_Anatomy")
    labels = viewer.add_labels(np.zeros_like(avg_map_s4, dtype=np.uint8), name="Kidney_Masks")
    
    print("\n--- DRAWING INSTRUCTIONS ---")
    print("Label 1: Paint the LEFT Kidney")
    print("Label 2: Paint the RIGHT Kidney")
    print("[Action Required] Close Napari window when finished.")
    napari.run()
    
    left_m = (labels.data == 1).astype(np.uint8)
    right_m = (labels.data == 2).astype(np.uint8)
    return left_m, right_m

def extract_kidney_intensities(data_3d, left_m, right_m):
    n_times = data_3d.shape[-1]
    left_intensity = np.zeros(n_times)
    right_intensity = np.zeros(n_times)
    for t in range(n_times):
        frame = data_3d[:, :, t]
        if np.any(left_m): left_intensity[t] = np.mean(frame[left_m == 1])
        if np.any(right_m): right_intensity[t] = np.mean(frame[right_m == 1])
    return left_intensity, right_intensity

# =================================================================================
# 4. MOTION CORRECTION & VISUALIZATION FUNCTIONS
# =================================================================================
def run_mdr_motion_correction(pixel_array, acq_times, aif_values, baseline_frames=18, results_dir='MDR_Temp_Results', coreg_options=None):
    print(f"\n--- STARTING MODEL-DRIVEN REGISTRATION ---")
    os.makedirs(results_dir, exist_ok=True)

    t = time.time()
    coreg, fit, transfo, pars = mdreg.fit(
        pixel_array,
        fit_image={'func': mdreg.fit_2cm_lin, 'time': acq_times, 'aif': aif_values, 'baseline': baseline_frames},
        fit_coreg=coreg_options,
        maxit=3,
        path=results_dir,
        verbose=2,
    )
    print(f"MDR Computation time: {round(time.time() - t)} seconds.")
    return coreg, fit

def save_gif(data, filename, title='Motion Corrected Data', fps=5):
    print(f"Saving GIF animation to {filename}...")
    vmax = np.percentile(data, 99) 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    im = ax.imshow(data[:, :, 0], cmap='gray', vmin=0, vmax=vmax)
    
    def update(frame_idx):
        im.set_array(data[:, :, frame_idx])
        return [im]
        
    anim = animation.FuncAnimation(fig, update, frames=data.shape[-1], blit=True)
    anim.save(filename, writer='pillow', fps=fps)
    plt.close(fig)

def generate_comparison_kymograph(raw_array, corrected_array, test_name, save_path, col_idx=125):
    """Generates vertical space-time plots to justify motion correction choice."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # AMENDED: Slicing vertically along a column to track Superior-Inferior motion
    kymo_raw = raw_array[:, col_idx, :]
    kymo_corr = corrected_array[:, col_idx, :]

    ax1.imshow(kymo_raw, cmap='gray', aspect='auto')
    ax1.set_title(f"Raw Vertical Kymograph (Column {col_idx})")
    ax1.set_ylabel("Superior-Inferior Position")
    
    ax2.imshow(kymo_corr, cmap='gray', aspect='auto')
    ax2.set_title(f"Corrected: {test_name} (Column {col_idx})")
    ax2.set_ylabel("Superior-Inferior Position")
    ax2.set_xlabel("Time (Frames)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

# =================================================================================
# 5. MAIN EXECUTION SCRIPT
# =================================================================================
if __name__ == "__main__":
    results_folder = r"C:\Users\eia21frd\Documents\RESULTS\hyperparameters\followup\time"
    folder_path = r"C:\Users\eia21frd\Documents\DATA\hyperparametre\iBE-3128-136_followup\series_43\iBE-3128-136_followup\scans\43-DCE_kidneys_cor_oblique_fb\resources\DICOM\files"
    os.makedirs(results_folder, exist_ok=True)

    # --- AIF CALCULATION (SLICE 0) ---
    print("\n--- CALCULATING AIF ON SLICE 0 ---")
    pixel_array, acq_times = load_entire_time_series(folder_path)
    average_pixel_array = describe(pixel_array) 
    mask = draw_arterial_input(average_pixel_array) 
    
    aif_values_raw = calculate_aif(pixel_array, mask)
    np.save(os.path.join(results_folder, 'aif_values_raw.npy'), aif_values_raw)

    # --- SLICE 4 SETUP & MASKING ---
    print("\n--- LOADING SLICE 4 ---")
    pixel_array_s4, acq_times_s4 = load_single_slice_time_series(folder_path, slice_idx=4)
    np.save(os.path.join(results_folder, 'raw_slice_4.npy'), pixel_array_s4)
    
    avg_map_s4 = create_slice_4_average(pixel_array_s4)
    aif_values_s4 = np.interp(acq_times_s4, acq_times, aif_values_raw)

    l_mask, r_mask = draw_kidney_masks_slice_4(avg_map_s4)

    # --- HYPERPARAMETER TESTING LOOP ---
    test_configs = [
        {'name': 'skimage', 'options': {'progress_bar': True, 'attachment': 30}},
        {'name': 'elastix_space_25', 'options': {'package': 'elastix', 'spacing': [1.0416666269302, 1.0416666269302], 'FinalGridSpacingInPhysicalUnits': 25.0}},
        {'name': 'elastix_space_50', 'options': {'package': 'elastix', 'spacing': [1.0416666269302, 1.0416666269302], 'FinalGridSpacingInPhysicalUnits': 50.0}},
        {'name': 'elastix_space_75', 'options': {'package': 'elastix', 'spacing': [1.0416666269302, 1.0416666269302], 'FinalGridSpacingInPhysicalUnits': 75.0}},
        {'name': 'ants', 'options': {'package': 'ants', 'type_of_transform': 'SyNOnly'}}
    ]

    for test in test_configs:
        test_name = test['name']
        options = test['options']
        
        print(f"\n--- STARTING TEST: {test_name.upper()} ---")
        test_results_dir = os.path.join(results_folder, f"MDR_{test_name}_Results")
        os.makedirs(test_results_dir, exist_ok=True)

        corrected_s4, fit_s4 = run_mdr_motion_correction(
            pixel_array=pixel_array_s4, acq_times=acq_times_s4, 
            aif_values=aif_values_s4, baseline_frames=10,
            results_dir=test_results_dir, coreg_options=options 
        )

        # Save corrected full arrays and GIFs
        np.save(os.path.join(test_results_dir, f'motion_corrected_{test_name}.npy'), corrected_s4)
        save_gif(corrected_s4, os.path.join(test_results_dir, f'Corrected_Slice_4_{test_name}.gif'))

        # --- GENERATE KYMOGRAPH JUSTIFICATION ---
        # AMENDED: Now using Column 125 for the justification plot
        kymo_path = os.path.join(test_results_dir, f'Justification_Kymograph_{test_name}.png')
        generate_comparison_kymograph(pixel_array_s4, corrected_s4, test_name, kymo_path, col_idx=125)

    print("\nPipeline Complete! Vertical Kymographs and GIFs generated.")
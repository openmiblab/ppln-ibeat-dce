# =================================================================================
# STANDALONE AIF PLOTTER (SLICE 0)
# =================================================================================
import os
import numpy as np
import napari
import pydicom
import matplotlib.pyplot as plt

# =================================================================================
# 1. DATA LOADING FUNCTIONS
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
# 2. MASKING & INTENSITY FUNCTIONS
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

# =================================================================================
# 3. MAIN EXECUTION SCRIPT
# =================================================================================
if __name__ == "__main__":
    results_folder = r"C:\Users\eia21frd\Documents\RESULTS\hyperparameters\baseline"
    folder_path = r"C:\Users\eia21frd\Documents\DATA\hyperparametre\iBE-3128-136\series_43\iBE-3128-136\scans\43-DCE_kidneys_cor_oblique_fb\resources\DICOM\files"
    os.makedirs(results_folder, exist_ok=True)

    print("\n--- CALCULATING AIF ON SLICE 0 ---")
    pixel_array, acq_times = load_entire_time_series(folder_path)
    average_pixel_array = describe(pixel_array) 
    
    # 1. Draw Mask
    mask = draw_arterial_input(average_pixel_array) 
    
    # 2. Calculate Intensity
    aif_values_raw = calculate_aif(pixel_array, mask)
    np.save(os.path.join(results_folder, 'aif_values_raw.npy'), aif_values_raw)

    # 3. Generate Plot
    print("\nGenerating and saving AIF Plot for Slice 0...")
    plt.figure(figsize=(10, 5))
    
    # CHANGED: Removed markers and made the line thinner (linewidth=1)
    plt.plot(acq_times, aif_values_raw, color='darkred', linewidth=1)
    
    plt.title("Arterial Input Function (AIF) - Slice 0", fontsize=16, fontweight='bold')
    plt.xlabel("Time (Seconds)", fontsize=14)
    plt.ylabel("Mean Signal Intensity", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 4. Save and Show
    aif_plot_path = os.path.join(results_folder, 'AIF_Slice_0_Curve.png')
    plt.savefig(aif_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved AIF Plot to: {aif_plot_path}")
    
    # Display the plot on your screen
    plt.show()
    
    print("\nScript Complete!")
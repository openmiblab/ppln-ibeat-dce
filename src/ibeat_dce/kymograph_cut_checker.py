# =================================================================================
# PRE-RUN DIAGNOSTIC: VERTICAL KYMOGRAPH CUT CHECKER (AMENDED FOR COL 125)
# =================================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom

def load_single_slice_time_series(folder_path, slice_idx):
    """Loads Slice 4 across all time points to create the anatomy map."""
    dicom_headers = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'AcquisitionTime'):
                    dicom_headers.append({
                        'z': float(ds.ImagePositionPatient[2]),
                        'path': filepath,
                        'time': float(ds.AcquisitionTime)
                    })
            except: continue

    if not dicom_headers:
        raise ValueError(f"No valid DICOM files found in {folder_path}")

    # Identify Slice 4
    unique_zs = sorted(list(set(d['z'] for d in dicom_headers)))
    target_z = unique_zs[slice_idx]
    target_files = sorted([d for d in dicom_headers if d['z'] == target_z], key=lambda x: x['time'])
    
    # Load pixels
    ref = pydicom.dcmread(target_files[0]['path'])
    pixel_array = np.zeros((ref.Rows, ref.Columns, len(target_files)))
    for i, item in enumerate(target_files):
        ds = pydicom.dcmread(item['path'])
        pixel_array[:, :, i] = ds.pixel_array.astype(np.float64)
        
    return pixel_array

# --- CONFIGURATION ---
# 1. Update this to your local data path
data_path = r"C:\Users\eia21frd\Documents\DATA\hyperparametre\iBE-3128-136\series_43\iBE-3128-136\scans\43-DCE_kidneys_cor_oblique_fb\resources\DICOM\files"

# 2. SETTINGS
cut_mode = 'col'  # Vertical cut
cut_index = 125   # Centered on the kidney based on your previous check

# =================================================================================
# EXECUTION
# =================================================================================
print(f"--- Loading Slice 4 for Anatomy Check ---")
try:
    pixel_data = load_single_slice_time_series(data_path, slice_idx=4)
    avg_map = np.mean(pixel_data, axis=-1)

    plt.figure(figsize=(10, 10))
    # Using 'bone' or 'gray' cmap for better anatomical detail
    plt.imshow(avg_map, cmap='gray')
    
    # Draw the vertical line at Column 125
    plt.axvline(x=cut_index, color='red', linewidth=3, linestyle='--')
    
    plt.xlabel("X (Columns)")
    plt.ylabel("Y (Rows)")
    
    # Adding text to the plot to confirm coordinates
    plt.text(cut_index + 2, 20, f"Target: Col {cut_index}", color='red', fontweight='bold')

    plt.grid(alpha=0.3)
    print(f"Displaying preview... verify the red line bisects the kidney.")
    plt.show()

except Exception as e:
    print(f"ERROR: {e}")
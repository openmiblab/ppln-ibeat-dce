## =================================================================================
# PRE-RUN DIAGNOSTIC: HORIZONTAL KYMOGRAPH CUT CHECKER (REPORT READY)
# =================================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom

# --- REPORT FORMATTING (LARGER FONTS) ---
plt.rcParams.update({
    'font.size': 16,          # Global font size
    'axes.titlesize': 20,     # Title size
    'axes.labelsize': 18,     # X and Y axis label size
    'xtick.labelsize': 14,    # Tick marks on X axis
    'ytick.labelsize': 14,    # Tick marks on Y axis
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

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
cut_mode = 'row'  # AMENDED: Horizontal cut
cut_index = 130    # AMENDED: Set to 95 for the row cut

# =================================================================================
# EXECUTION
# =================================================================================
print(f"--- Loading Slice 4 for Anatomy Check ---")
try:
    pixel_data = load_single_slice_time_series(data_path, slice_idx=4)
    avg_map = np.mean(pixel_data, axis=-1)

    # --- THE BRIGHTNESS FIX ---
    # Calculate the 98th percentile to clip extreme bright spots and brighten tissue
    display_vmax = np.percentile(avg_map, 98)

    # Increased figure size slightly to accommodate larger fonts
    plt.figure(figsize=(12, 12))
    
    # Passing the vmax threshold to brighten the image
    plt.imshow(avg_map, cmap='gray', vmax=display_vmax)
    
    # AMENDED: Draw the horizontal line at Row 95
    plt.axhline(y=cut_index, color='red', linewidth=4, linestyle='--')
    
    # AMENDED: Title and Axes updated for Horizontal orientation
    plt.title(f"PREVIEW: Horizontal Cut at ROW {cut_index}\n(Tracking Left-Right Kidney Cross-Section)", pad=20)
    plt.xlabel("X (Columns)")
    plt.ylabel("Y (Rows)")
    
    # AMENDED: Shifted text to the left side, sitting just above the red horizontal line
    plt.text(20, cut_index - 4, f"Target: Row {cut_index}", color='red', fontweight='bold', fontsize=18)

    plt.grid(alpha=0.3)
    print(f"Displaying preview... verify the red line passes through the kidney horizontally.")
    
    # Optional: Saves a high-res copy for your report methodology section
    plt.savefig("Methodology_Horizontal_Cut_Preview.png", dpi=300, bbox_inches='tight')
    
    plt.show()

except Exception as e:
    print(f"ERROR: {e}")
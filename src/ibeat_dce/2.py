# =================================================================================
# CONSOLIDATED REPORT FIGURE GENERATOR (BRIGHTNESS ADJUSTED)
# =================================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
results_folder = r"C:\Users\eia21frd\Documents\RESULTS\hyperparameters\followup"
col_idx = 125 # The target column we validated earlier
test_names = ['skimage', 'elastix_space_25', 'elastix_space_50', 'elastix_space_75', 'ants']

# --- 2. REPORT FORMATTING ---
plt.rcParams.update({
    'font.size': 14,          
    'axes.titlesize': 18,     
    'axes.labelsize': 14,     
    'xtick.labelsize': 12,    
    'ytick.labelsize': 12,    
    'font.family': 'sans-serif'
})

# --- 3. DATA LOADING & BRIGHTNESS CALCULATION ---
print(f"Loading raw data from: {results_folder}")
try:
    raw_array = np.load(os.path.join(results_folder, 'raw_slice_4.npy'))
    kymo_raw = raw_array[:, col_idx, :]
    
    # --- THE BRIGHTNESS FIX ---
    # Calculate the 98th percentile of the raw data. 
    # Any pixel brighter than this will just be white, making the rest of the tissue visible.
    display_vmax = np.percentile(kymo_raw, 98)
    
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'raw_slice_4.npy'. Ensure the main script finished successfully.")

# --- 4. PLOTTING THE 3x2 GRID ---
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

# Panel 1: Raw Data
# Notice the addition of vmax=display_vmax
axes[0].imshow(kymo_raw, cmap='gray', aspect='auto', vmax=display_vmax)
axes[0].set_title("Raw Data (No Correction)", fontweight='bold')
axes[0].set_ylabel("Superior-Inferior Position")

# Panels 2-6: Corrected Data
for i, test in enumerate(test_names):
    ax = axes[i + 1]
    
    file_path = os.path.join(results_folder, f"MDR_{test}_Results", f"motion_corrected_{test}.npy")
    try:
        corr_array = np.load(file_path)
        kymo_corr = corr_array[:, col_idx, :]
        
        # Apply the SAME brightness threshold to ensure a fair comparison
        ax.imshow(kymo_corr, cmap='gray', aspect='auto', vmax=display_vmax)
        
        clean_title = test.replace('_', ' ').title()
        if 'Elastix' in clean_title:
            clean_title = clean_title.replace('Space ', 'Grid Spacing ')
        elif 'Ants' in clean_title:
            clean_title = "ANTs (SyNOnly)"
            
        ax.set_title(f"Corrected: {clean_title}", fontweight='bold')
        
    except FileNotFoundError:
        ax.set_title(f"Data Missing: {test}", color='red')
        ax.axis('off')

    if (i + 1) % 2 == 0:
        ax.set_ylabel("Superior-Inferior Position")
        
    if (i + 1) >= 4:
        ax.set_xlabel("Time (Frames)")

# --- 5. FINAL SAVING ---
plt.tight_layout(pad=2.0)
save_path = os.path.join(results_folder, "Consolidated_Kymograph_Report_Figure.png")

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nSuccess! High-resolution, brightness-adjusted report figure saved to:\n{save_path}")

plt.show()
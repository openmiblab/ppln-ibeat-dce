# =================================================================================
# CONSOLIDATED REPORT FIGURE GENERATOR (BRIGHTNESS BOOSTED)
# =================================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
results_folder = r"C:\Users\eia21frd\Documents\RESULTS\hyperparameters\followup\time2"
col_idx = 125 # The target column we validated earlier
test_names = ['skimage', 'elastix_space_25', 'elastix_space_50', 'elastix_space_75', 'ants']

# --- 2. REPORT FORMATTING (LARGER FONTS) ---
# This updates matplotlib's global settings to make text larger and clearer
plt.rcParams.update({
    'font.size': 14,          # Global font size
    'axes.titlesize': 18,     # Title of each subplot
    'axes.labelsize': 14,     # X and Y axis labels
    'xtick.labelsize': 12,    # Tick marks on X axis
    'ytick.labelsize': 12,    # Tick marks on Y axis
    'font.family': 'sans-serif'
})

# --- 3. DATA LOADING & BRIGHTNESS SETTINGS ---
print(f"Loading raw data from: {results_folder}")
try:
    raw_array = np.load(os.path.join(results_folder, 'raw_slice_4.npy'))
    kymo_raw = raw_array[:, col_idx, :]
    
    # ADDED: Calculate the 95th percentile to massively boost overall brightness
    display_vmax = np.percentile(kymo_raw, 95)
    
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'raw_slice_4.npy'. Ensure the main script finished successfully.")

# --- 4. PLOTTING THE 3x2 GRID ---
# figsize=(16, 12) is roughly a 4:3 aspect ratio, great for standard report pages
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten() # Flatten to loop through easily

# Panel 1: Raw Data
# ADDED: vmax=display_vmax to brighten the image
axes[0].imshow(kymo_raw, cmap='gray', aspect='auto', vmax=display_vmax)
axes[0].set_title("Raw Data (No Correction)", fontweight='bold')
axes[0].set_ylabel("Superior-Inferior Position")

# Panels 2-6: Corrected Data
for i, test in enumerate(test_names):
    ax = axes[i + 1]
    
    # Load the specific test data
    file_path = os.path.join(results_folder, f"MDR_{test}_Results", f"motion_corrected_{test}.npy")
    try:
        corr_array = np.load(file_path)
        kymo_corr = corr_array[:, col_idx, :]
        
        # ADDED: vmax=display_vmax to keep brightness standard across all plots
        ax.imshow(kymo_corr, cmap='gray', aspect='auto', vmax=display_vmax)
        
        # Clean up the test name for a professional report title
        clean_title = test.replace('_', ' ').title()
        if 'Elastix' in clean_title:
            clean_title = clean_title.replace('Space ', 'Grid Spacing ')
        elif 'Ants' in clean_title:
            clean_title = "ANTs (SyNOnly)"
            
        ax.set_title(f"Corrected: {clean_title}", fontweight='bold')
        
    except FileNotFoundError:
        ax.set_title(f"Data Missing: {test}", color='red')
        ax.axis('off')

    # Formatting: Only show Y-labels on the left column to reduce clutter
    if (i + 1) % 2 == 0:
        ax.set_ylabel("Superior-Inferior Position")
        
    # Formatting: Only show X-labels on the bottom row
    if (i + 1) >= 4:
        ax.set_xlabel("Time (Frames)")

# --- 5. FINAL SAVING ---
plt.tight_layout(pad=2.0) # Adds padding between subplots so titles don't overlap
save_path = os.path.join(results_folder, "Consolidated_Kymograph_Report_Figure.png")

# Saving at 300 DPI ensures it stays crisp when printed or zoomed in a PDF
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nSuccess! High-resolution report figure saved to:\n{save_path}")

# Display it on screen
plt.show()
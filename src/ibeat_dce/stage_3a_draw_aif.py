import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_aif_intensities(stage3_dir, output_dir):
    """
    Reads Stage 3 AIF .npy files using os.walk, plots time-intensity curves, 
    extracts Min/Max, and saves the plots to a new QC directory.
    """
    # 1. Create the new output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Bulletproof deep search using os.walk
    aif_files = []
    for root, dirs, files in os.walk(stage3_dir):
        if "aif_values.npy" in files:
            aif_files.append(os.path.join(root, "aif_values.npy"))
    
    if not aif_files:
        print(f"[!] No AIF files found in {stage3_dir} or any of its subfolders.")
        print("Are you sure the X: drive is connected and the files were saved?")
        return

    summary_data = []
    print(f"Found {len(aif_files)} 'aif_values.npy' files. Generating QC plots...")

    for file_path in aif_files:
        # Create a unique ID based on the folder path so we don't overwrite plots
        relative_path = os.path.relpath(os.path.dirname(file_path), stage3_dir)
        subject_id = relative_path.replace(os.sep, "_")
        
        try:
            # 3. Read the AIF NumPy data
            intensities = np.load(file_path)
            
            # 4. Extract QC metrics
            min_val = np.min(intensities)
            max_val = np.max(intensities)
            frame_count = len(intensities)
            
            # Log for the master spreadsheet
            summary_data.append({
                "Subject_ID": subject_id,
                "Folder_Path": os.path.dirname(file_path),
                "Frame_Count": frame_count,
                "Min_Intensity": min_val,
                "Max_Intensity": max_val
            })

            # --- 5. Generate the Plot ---
            plt.figure(figsize=(10, 6))
            plt.plot(intensities, marker='o', linestyle='-', color='b', markersize=4)
            
            plt.title(f"AIF Time-Intensity Curve: {subject_id}", fontweight='bold')
            plt.xlabel("Frame / Timepoint")
            plt.ylabel("Signal Intensity")
            
            # Add a text box with the QC data for rapid viewing
            info_text = f"Frames: {frame_count}\nMin: {min_val:.2f}\nMax: {max_val:.2f}"
            plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, 
                     fontsize=12, verticalalignment='top', horizontalalignment='right', 
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 6. Save the plot as a PNG
            save_path = os.path.join(output_dir, f"{subject_id}_AIF_plot.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Close figure to free up memory
            plt.close() 
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    # 7. Save the master summary spreadsheet
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, "AIF_MinMax_Summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"--- SUCCESS ---")
    print(f"Plots and summary CSV successfully saved to:\n{output_dir}")

# ==========================================
# EXECUTION ZONE
# ==========================================
if __name__ == "__main__":
    # Updated with your exact local paths
    STAGE_3_INPUT_DIR = r"X:\abdominal_imaging\Shared\ibeat_dce\results\stage_3" 
    NEW_QC_OUTPUT_DIR = r"X:\abdominal_imaging\Shared\ibeat_dce\results\aif_plots"

    plot_aif_intensities(STAGE_3_INPUT_DIR, NEW_QC_OUTPUT_DIR)
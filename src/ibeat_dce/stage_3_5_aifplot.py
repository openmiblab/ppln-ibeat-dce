import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_aif_intensities(stage3_dir, stage2_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    aif_files = []
    for root, dirs, files in os.walk(stage3_dir):
        if "aif_values.npy" in files:
            aif_files.append(os.path.join(root, "aif_values.npy"))
    
    if not aif_files:
        print(f"[!] No AIF files found in {stage3_dir} or any of its subfolders.")
        return

    summary_data = []
    print(f"Found {len(aif_files)} 'aif_values.npy' files. Generating QC plots...")

    for file_path in aif_files:
        relative_path = os.path.relpath(os.path.dirname(file_path), stage3_dir)
        subject_id = relative_path.replace(os.sep, "_")
        
        time_file_path = os.path.join(stage2_dir, relative_path, "acq_times.npy")
        
        try:
            intensities = np.load(file_path)
            frame_count = len(intensities)
            
            if os.path.exists(time_file_path):
                acq_times = np.load(time_file_path)
                
                # 1. FIX JAGGEDNESS: Sort the times so they move strictly forward
                acq_times = np.sort(acq_times)
                time_count = len(acq_times)
                
                # 2. FIX MISMATCH: If we have slice-level times, average them into volume-level times
                if time_count > frame_count * 2:
                    print(f"[*] Compressing {subject_id}: Averaging {time_count} slice times into {frame_count} frames.")
                    # Split the array into 'frame_count' equal chunks and take the mean of each chunk
                    acq_times = np.array([np.mean(chunk) for chunk in np.array_split(acq_times, frame_count)])
                    time_count = len(acq_times)
                
                # Safety net for minor 1-2 frame mismatches (e.g. an extra localizer slice)
                if time_count != frame_count:
                    min_length = min(time_count, frame_count)
                    acq_times = acq_times[:min_length]
                    intensities = intensities[:min_length]
                    frame_count = min_length 

                # Normalize time so the plot starts at 0
                x_data = acq_times - acq_times[0] 
                x_label = "Acquisition Time (seconds)"
                
            else:
                print(f"[!] No time file found for {subject_id}. Using frame count.")
                x_data = np.arange(frame_count)
                x_label = "Frame / Timepoint"

            min_val = np.min(intensities)
            max_val = np.max(intensities)
            
            summary_data.append({
                "Subject_ID": subject_id,
                "Folder_Path": os.path.dirname(file_path),
                "Frame_Count": frame_count,
                "Min_Intensity": min_val,
                "Max_Intensity": max_val
            })

            plt.figure(figsize=(10, 6))
            plt.plot(x_data, intensities, marker='o', linestyle='-', color='b', markersize=4)
            plt.title(f"AIF Curve: {subject_id}", fontweight='bold')
            plt.xlabel(x_label)
            plt.ylabel("Signal Intensity")
            
            info_text = f"Frames: {frame_count}\nMin: {min_val:.2f}\nMax: {max_val:.2f}"
            plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, 
                     fontsize=12, verticalalignment='top', horizontalalignment='right', 
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            plt.grid(True, linestyle='--', alpha=0.7)
            
            save_path = os.path.join(output_dir, f"{subject_id}_AIF_plot.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close() 
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "AIF_MinMax_Summary.csv"), index=False)
    print(f"--- SUCCESS: Saved to {output_dir} ---")

if __name__ == "__main__":
    STAGE_3_INPUT_DIR = r"X:\abdominal_imaging\Shared\ibeat_dce\results\stage_3" 
    STAGE_2_INPUT_DIR = r"X:\abdominal_imaging\Shared\ibeat_dce\results\stage_2" 
    NEW_QC_OUTPUT_DIR = r"X:\abdominal_imaging\Shared\ibeat_dce\results\aif_plots3"
    
    plot_aif_intensities(STAGE_3_INPUT_DIR, STAGE_2_INPUT_DIR, NEW_QC_OUTPUT_DIR)
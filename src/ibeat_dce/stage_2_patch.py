import os
import pydicom
import numpy as np

def extract_times_from_dicoms(dicom_folder):
    """Reads DICOMs, sorts by Instance Number, and calculates time in seconds."""
    dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm') or f.endswith('.ima')]
    if not dicom_files:
        return None

    # Load only headers to save memory
    slices = [pydicom.dcmread(f, stop_before_pixels=True) for f in dicom_files]
    slices.sort(key=lambda x: int(x.InstanceNumber) if 'InstanceNumber' in x else 0)

    times = []
    for s in slices:
        if 'TriggerTime' in s:
            times.append(float(s.TriggerTime) / 1000.0)
        elif 'AcquisitionTime' in s:
            t_str = s.AcquisitionTime
            hours, minutes, seconds = float(t_str[0:2]), float(t_str[2:4]), float(t_str[4:6])
            times.append((hours * 3600) + (minutes * 60) + seconds)
        else:
            times.append(0.0)

    if 'TriggerTime' not in slices[0]:
        t0 = times[0]
        times = [t - t0 for t in times]

    return np.array(times)

def patch_stage2_times(raw_dir, stage2_dir):
    print("Starting retroactive time extraction on HPC...")
    
    for root, dirs, files in os.walk(stage2_dir):
        if 'raw_data_slice_0.npy' in files:
            relative_path = os.path.relpath(root, stage2_dir)
            matching_raw_folder = os.path.join(raw_dir, relative_path)
            save_path = os.path.join(root, 'acq_times.npy')
            
            if os.path.exists(save_path):
                continue
                
            if os.path.exists(matching_raw_folder):
                try:
                    acq_times = extract_times_from_dicoms(matching_raw_folder)
                    if acq_times is not None and len(acq_times) > 0:
                        np.save(save_path, acq_times)
                        print(f"[+] Patched: {relative_path}")
                    else:
                        print(f"[-] No DICOMs found in: {matching_raw_folder}")
                except Exception as e:
                    print(f"[!] Error processing {relative_path}: {e}")
            else:
                print(f"[!] Could not find matching raw folder for: {relative_path}")

if __name__ == "__main__":
    # UPDATE THESE TO YOUR ACTUAL HPC /parscratch PATHS
    RAW_DICOM_DIR = "/mnt/parscratch/users/eia21frd/data/ibeat_dce/stage_1_download" 
    STAGE_2_DIR = "/mnt/parscratch/users/eia21frd/build/stage_2_compute_descriptivemaps/"
    
    patch_stage2_times(RAW_DICOM_DIR, STAGE_2_DIR)
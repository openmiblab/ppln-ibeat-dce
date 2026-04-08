import os
import sys
import zipfile

# =================================================================================
# STAGE 1.5: DEDICATED UNZIPPER
# =================================================================================

def unzip_stage_1_data(stage_1_path):
    """Scans the Stage 1 folder and extracts any .zip files it finds, skipping already extracted ones."""
    print(f"\n[STAGE 1.5] Scanning directory for zipped patient data...")
    print(f"Target Directory: {stage_1_path}\n")
    
    zip_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(stage_1_path):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                
                # --- SMART SKIP LOGIC ---
                # Check if there are already extracted files OR folders in this exact directory
                extracted_files = [f for f in files if not f.endswith('.zip') and not f.startswith('.')]
                
                if len(extracted_files) > 0 or len(dirs) > 0:
                    skipped_count += 1
                    print(f"    [-] SKIPPING: {file} (Already extracted data/folder found)")
                    continue 
                # --------------------------------
                
                zip_count += 1
                print(f"    -> [{zip_count}] Unzipping: {file}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(root)
                    
                    # HPC PRO-TIP: Uncomment the line below to delete the .zip file 
                    # after it extracts. This saves massive amounts of storage space!
                    # os.remove(zip_path) 
                    
                except Exception as e:
                    print(f"    [!] Failed to unzip {file}: {e}")
                    
    print("\n==================================================")
    if zip_count == 0 and skipped_count == 0:
        print("RESULT: No zip files found anywhere.")
    else:
        print(f"RESULT: Extraction Complete!")
        print(f" -> Newly Extracted: {zip_count} archives")
        print(f" -> Skipped: {skipped_count} archives (already unzipped)")
    print("==================================================\n")

if __name__ == "__main__":
    # If the bash script passes a path, use it. Otherwise, use the Windows default.
    if len(sys.argv) > 1:
        stage_1_dir = sys.argv[1]
    else:
        stage_1_dir = r"X:\abdominal_imaging\Shared\ibeat_dce\data\stage_1_download"
    
    if not os.path.exists(stage_1_dir):
        print(f"[!] ERROR: Directory not found: {stage_1_dir}")
        sys.exit(1)
        
    unzip_stage_1_data(stage_1_dir)
"""
Create clean database
"""

import os
import zipfile
import shutil
import logging
import tempfile
import csv
import argparse

from tqdm import tqdm
import numpy as np
# import pydicom
import dbdicom as db
# import vreg

from pathlib import Path

from miblab import pipe


PIPELINE = 'ibeat_dce'
EXCLUDE = [
    # Sheffield IDs with series that requires pre-rpocessong
    "IBE7128002",
    "IBE7128005",
    "IBE7128006",
    "IBE7128007", 
    "IBE7128008",
    "IBE7128009",
    "IBE7128010",
    "IBE7128011",
    "IBE7128012",
    "IBE7128014",
    "IBE7128015",
    "IBE7128016",
    "IBE7128017",
]



# Going through the Sheffield patients
def sheffield_ibeat_patient_id(folder):
# sheffield IDs are formatted like IBE7128002, needs to be 7128_002
    id = folder[3:]
    id = id[:4] + '_' + id[4:]
    if id == '2178_157': # Data entry error
        id = '7128_157'
    return id


def sheffield(download_path, output_dir):
    sitedownloadpath = os.path.join(download_path, "ibeat_dce", "stage_2_download", "BEAt-DKD-WP4-Sheffield")
    sitedatapath = os.path.join(output_dir, "Patients", "Sheffield")
    os.makedirs(sitedatapath, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_folder:
        for patient_folder in os.listdir(sitedownloadpath):
            # patient_folder is e.g. IBE7128019; the experiment folder below it is ignored
            if patient_folder in EXCLUDE:
                logging.info(f"Patient {patient_folder} - skipped (in EXCLUDE list)")
                continue
            
            patient_path = Path(sitedownloadpath) / patient_folder
            zips = [p for p in patient_path.rglob("*") if p.suffix.lower() == ".zip"]

            for zip_path in zips:
                # Extract to a temporary folder (unique per patient and series)
                try:
                    extract_to = os.path.join(temp_folder, patient_folder, zip_path.stem)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                except Exception as e:
                    logging.error(f"Patient {patient_folder} - error extracting {zip_path.name}: {e}")
                    continue
                all_series_in_folder = db.series(extract_to)         # should return a list of 1
                source_series = all_series_in_folder[0]

                manufacturer = db.unique('Manufacturer', source_series)[0]

                if manufacturer=='GE MEDICAL SYSTEMS':
                    # Editing source series headers
                    # GE saves acquisition time in the TriggerTime field in msec, needs to be in AcquisitionTime in seconds
                    acq_time = db.values(source_series, 'TriggerTime', dims=['SliceLocation', 'TriggerTime'])
                    db.edit(source_series, {'AcquisitionTime': acq_time / 1000}, dims=['SliceLocation', 'TriggerTime'])

                # Copy edited source series ti final destination
                study = 'Baseline'
                patient_id = sheffield_ibeat_patient_id(patient_folder)
                series_description = 'DCE_kidneys_cor-oblique_fb'
                destination_series = [sitedatapath, patient_id, study, series_description]

                db.copy(source_series, destination_series)


        

    
def run(build, logfile):
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    sheffield(build, dir_output)



if __name__ == '__main__':

    build = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_stage(run, build, PIPELINE, __file__)

"""
Automatic download of DIXON data from XNAT.
"""

import logging

from miblab import pipe
from .utils.xnat_logging import download_series
# from miblab_data.xnat import download_series comented this because it uses custom miblab_data version

from tqdm import tqdm


PIPELINE = 'ibeat_dce'

SIEMENS = {
    "series_description": [
        "DCE_kidneys_cor-oblique_fb", 
    ]
} 
TURKU_PHILIPS = {
    "series_description": [
        "DCE-kidney-coronal-oblique-FB",
      
    ]
}  
TURKU_GE = {
    "series_description": [
        "3D_DISCO_Dyn_kidneys_cor-oblique_fb"
   ]
} 
SHEFFIELD_PATIENTS = {
    "series_description": [
        "DCE_kidneys_cor-oblique_fb",
        "3D_DISCO_Dyn_kidneys_cor-oblique_fb",
       
    ]
} 
BARI = {
    "series_description": [
        "DCE_kidneys_cor-oblique_fb_wet_pulse",
        "DCE_kidneys_cor-oblique_fb",
    ]
} 
LEEDS = {
    "parameters/sequence": ["*tfl2d1_16"]
} 



DOWNLOAD = {
    'leeds_patients':{
        'project_id': "BEAt-DKD-WP4-Leeds",
        'subject_label':"Leeds_Patients",
        'attr': LEEDS      
    },
    'bari_patients':{
        'project_id': "BEAt-DKD-WP4-Bari",
        'subject_label':"Bari_Patients",
        'attr': BARI      
    },
    'sheffield_patients':{
        'project_id': "BEAt-DKD-WP4-Sheffield",
        'attr': SHEFFIELD_PATIENTS      
    },
    'turku_ge_patients':{
        'project_id': "BEAt-DKD-WP4-Turku",
       'subject_label':"Turku_Patients_GE",
        'attr': TURKU_GE       
    },
    'turku_philips_patients':{
        'project_id': "BEAt-DKD-WP4-Turku",
        'subject_label': "Turku_Patients_Philips",
        'attr': TURKU_PHILIPS      
    },
    'bordeaux_patients_baseline':{
        'project_id': "BEAt-DKD-WP4-Bordeaux",
        'subject_label': "Bordeaux_Patients_Baseline",
        'attr': SIEMENS      
    },
    'bordeaux_patients_followup':{
        'project_id': "BEAt-DKD-WP4-Bordeaux",
        'subject_label': "Bordeaux_Patients_Followup",
        'attr': SIEMENS     
    },
    'exeter_patients_baseline':{
        'project_id': "BEAt-DKD-WP4-Exeter",
        'subject_label': "Exeter_Patients_Baseline",
        'attr': SIEMENS      
    },
    'exeter_patients_followup':{
        'project_id': "BEAt-DKD-WP4-Exeter",
        'subject_label': "Exeter_Patients_Followup",
        'attr': SIEMENS      
    },
}



def run(build, logfile):

    logging.info("Stage 1 --- Downloading data ---")

    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
 
    n_max= 1
    group_stats = {}   # dict of counts and failed files, recorded for each group
    failed_groups = [] # groups where download_series itself raised an error

    pbar = tqdm(DOWNLOAD.items())
    for group, props in pbar:
        pbar.set_description(f"Processing {group}")
        try:
            n_downloaded, failed_downloads, n_existed = download_series(
                xnat_url="https://qib.shef.ac.uk",
                output_dir=dir_output,
                log=True,
                n_max=n_max,
                **props
            )

            group_stats[group] = {
                "fetched": n_downloaded + n_existed,  # all matching series found
                "skipped": n_existed,                 # already exist on disk, not corrupted
                "attempted": n_downloaded,            # downloads actually tried
                "failed": failed_downloads,           # list of (filepath, error)
            }
            
            logging.info(f"Finished downloading {group}.") 
            
        except Exception:
            logging.exception(f"Error downloading {group}.")
            failed_groups.append(group) # add group to the list of failed groups

    _log_total_summary(group_stats, failed_groups, len(DOWNLOAD))


def _log_total_summary(group_stats, failed_groups, n_groups):
    "Log summary of the whole run, per group and in total"
    logging.info(f" \n ---------- DOWNLOAD SUMMARY ----------")
    logging.info(f"Total groups downloaded: {len(group_stats)} out of {n_groups}")
    if failed_groups:
        logging.info(f"Groups that had error during download: {', '.join(failed_groups)}")

    for group, s in group_stats.items():
        logging.info(
            f"\n Group: {group}"
            f"\n Files fetched (matched the series description): {s['fetched']}"
            f"\n Files skipped (already existed, not corrupted): {s['skipped']}"
            f"\n Files attempted to download: {s['attempted']}"
            f"\n Files failed due to error: {len(s['failed'])}"
        )
        for filepath, error in s["failed"]:
            logging.info(f"ERROR {error} : {filepath}")


if __name__ == '__main__':

    build = r"path\to\build"
    pipe.run_stage(run, build, PIPELINE, __file__)
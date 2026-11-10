import os
import logging
from tqdm import tqdm

from dataset_inspection_tools import inspect_series

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
        "3D_DISCO_Dyn_kidneys_cor-oblique_fb"
       
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

OUTPUT_DIR = "dataset_inspection"
os.makedirs(OUTPUT_DIR, exist_ok=True)


pbar = tqdm(DOWNLOAD.items())
for group, props in pbar:
    pbar.set_description(f"Processing {group}")

    try:
        df = inspect_series(
            xnat_url="https://qib.shef.ac.uk",
            **props
        )
        df.to_csv(f"{OUTPUT_DIR}/{group}_inspection.csv", index=False)
    except Exception as e:
        print(f"Error with {group}: {e}")
    
"""
Automatic download of DIXON data from XNAT.
"""
import logging

from miblab import pipe
from miblab_data.xnat import download_series
from tqdm import tqdm


PIPELINE = 'ibeat_dce'

SIEMENS = {
    "series_description": [
        "DCE_kidneys_cor-oblique_fb", 
    ]
} 
TURKU_PHILIPS = {
    "series_description": [
        'T1W-abdomen-Dixon-coronal-BH', 
        'T1W-abdomen-Dixon-post-coronal-BH'
    ]
}  
TURKU_GE = {
    "series_description": [
        "WATER: T1_abdomen_dixon_cor_bh", 
        "FAT: T1_abdomen_dixon_cor_bh",
        "InPhase: T1_abdomen_dixon_cor_bh",
        "OutPhase: T1_abdomen_dixon_cor_bh",
        "WATER: T1_abdomen_post_contrast_dixon_cor_bh",
        "FAT: T1_abdomen_post_contrast_dixon_cor_bh",
        "InPhase: T1_abdomen_post_contrast_dixon_cor_bh",
        "OutPhase: T1_abdomen_post_contrast_dixon_cor_bh"
    ]
} 
SHEFFIELD_PATIENTS = {
    "series_description": [
        # Philips data
        'T1w_abdomen_dixon_cor_bh', 
        'T1w_abdomen_post_contrast_dixon_cor_bh',
        # GE data
        'WATER: T1_abdomen_dixon_cor_bh',
        'FAT: T1_abdomen_dixon_cor_bh',
        'InPhase: T1_abdomen_dixon_cor_bh',
        'OutPhase: T1_abdomen_dixon_cor_bh',
        'WATER: T1_abdomen_post_contrast_dixon_cor_bh',
        'FAT: T1_abdomen_post_contrast_dixon_cor_bh',
        'InPhase: T1_abdomen_post_contrast_dixon_cor_bh',
        'OutPhase: T1_abdomen_post_contrast_dixon_cor_bh',
    ]
} 
BARI = {
    "series_description": [
        "T1w_abdomen_dixon_cor_bh", 
        "T1w_abdomen_post_contrast_dixon_cor_bh"
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



def run(build, dir_output):

    logging.info("Stage 1 --- Downloading data ---")
 
    n_max=None # downlaod all from each group
    # n_max=1 # download just 1 from each group

    for group, props in tqdm(DOWNLOAD.items(), desc='Downloading..'):
        try:
            download_series(
                xnat_url="https://qib.shef.ac.uk",
                output_dir=dir_output,
                log=True,
                n_max=n_max,
                **props
            )
            logging.info(f"Finished downloading {group}.")
        except:
            logging.exception(f"Error downloading {group}.")



if __name__ == '__main__':

    build = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_stage(run, build, PIPELINE, __file__)
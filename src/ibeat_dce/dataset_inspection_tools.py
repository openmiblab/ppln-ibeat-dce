"""
Tool for the dataset health analysis
"""

from tqdm import tqdm
import pandas as pd
import fnmatch
import logging

from xnat_logging import _xnat_session




# Define columns that should be added to the DataFrame
COLUMNS = [
    "project_id", "subject_label", "experiment_label", "scanner_type", "series_descr_match",
    "series_description", "scan_id", "n_frames",
]

def inspect_series(xnat_url, project_id, subject_label=None, experiment_label=None,  attr:dict=None, cred="user_XNAT.txt"):
    
    session = _xnat_session(cred)

    df_rows = [] # list of rows that should be added to the 

    subj_url = f"{xnat_url}/data/projects/{project_id}/subjects?format=json"
    r = session.get(subj_url)
    r.raise_for_status()
    subjects = r.json()['ResultSet']['Result']

    for subj in tqdm(subjects, desc='Scanning subjects..'):
        subject_id = subj['ID']

        # Continue if this subject was not requested
        if subject_label is not None:
            if subj['label'] != subject_label:
                continue
 
        # Loop over the subject's experiments
        exp_url = f"{xnat_url}/data/projects/{project_id}/subjects/{subject_id}/experiments?format=json"
        r = session.get(exp_url)
        r.raise_for_status()
        experiments = r.json()['ResultSet']['Result']       

        for exp in tqdm(experiments, desc=f'Scanning experiments of subj ID: {subject_id}'):
            exp_id = exp['ID']

            # Continue if this experiment was not requested
            if experiment_label is not None:
                if exp['label'] != experiment_label:
                    continue

            series_descr_match = False # flag signing that the series_description for experiment is found (by default it is False)

            # Loop over the scans in the experiment
            scans_url = f"{xnat_url}/data/experiments/{exp_id}/scans?format=json"
            r = session.get(scans_url)
            r.raise_for_status()
            scans = r.json()['ResultSet']['Result']

            for scan in scans:
                scan_id = scan['ID']

                # Retrieve scan attributes
                attr_url = f"{xnat_url}/data/experiments/{exp_id}/scans/{scan_id}?format=json"
                r = session.get(attr_url)
                r.raise_for_status()
                scan_attrs = r.json()['items'][0]['data_fields']

                # skip scan if there is no match with the series description
                if not _matches(scan_attrs, attr):
                    continue

                series_descr_match = True

                # dictionary in case if the series description was found
                df_rows.append({
                    "project_id": project_id,
                    "subject_label": subj['label'],
                    "experiment_label": exp['label'],
                    "scanner_type": _get_scanner_type(scan_attrs),
                    "series_descr_match": True,
                    "series_description": scan['series_description'],
                    "scan_id": scan['ID'],
                    "n_frames": scan_attrs.get("frames") # quick note: this value has dtype float for some of the series
                })

            if not series_descr_match:
                df_rows.append({
                    "project_id": project_id,
                    "subject_label": subj['label'],
                    "experiment_label": exp['label'],
                    "scanner_type": _get_scanner_type(scan_attrs),
                    "series_descr_match": False,
                }) # the rest of the rows will be recorded as None in the dataframe


    return pd.DataFrame(df_rows, columns=COLUMNS)



def _matches(scan_attrs, attr):
    """
    If series_description exists (all patients except leeds):
        True if the scan's series_description matches one of the requested patterns

    If there is parameters/sequence attribute (Leeds case):
        True if scan's parameter/sequence attribute matches one fthe requested patterns
    """
    try: 
        if "series_description" in attr:
            key = "series_description"

        elif "parameters/sequence" in attr: # Leeds case
            key = "parameters/sequence"

        else:
            logging.info("no match found")
            return False

        value = scan_attrs.get(key)
        if value is None:
            return False # return no match in case if there os

        return any(fnmatch.fnmatchcase(value, p) for p in attr[key])
    except Exception as e:
        logging.error(f"ERROR processing scan {scan_attrs.get('ID')} : {e}")



def _get_scanner_type(scan_attrs):
    """Extract the name of the scanner from attributes"""
    # collect the candidate parts
    parts = [scan_attrs.get("scanner/manufacturer"), scan_attrs.get("scanner/model")]

    # drop empty values for the candidate parts (otherwise there will be an error in the code)
    parts = [p for p in parts if p]
    return " ".join(parts) if parts else scan_attrs.get("scanner")
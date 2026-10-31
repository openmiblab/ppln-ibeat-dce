"""
Modified version of xnat download file with more detailed logging process
Original file: https://github.com/openmiblab/pckg-miblab-data/blob/main/src/miblab_data/xnat.py
"""


import os
import logging
import requests
from requests.auth import HTTPBasicAuth
import zipfile

from tqdm import tqdm


def download_series(
    xnat_url, output_dir, project_id,
    subject_label=None, experiment_label=None, attr:dict=None,
    n_max=None, log=False, cred="user_XNAT.txt"
):
    """
    Downloads all scan series with a given attribute value.

    Args:
        xnat_url (str): Base URL of the XNAT server.
        output_dir (str): Directory to store downloaded data.
        project_id (str): XNAT project ID.
        subject_label (str): XNAT subject label. If this is None, 
            all subjects are downloaded. Defaults to None.
        experiment_label (str): XNAT experiment label. If this is None, 
            all experiments are downloaded. Defaults to None.
        attr (dict, optional): Attribute(s) to filter by. Keys 
            must be valid XNAT attributes, and values must be a list 
            of required values. Series will only be returned if each 
            attribute has a value in the list.
        n_max (int, optional): maximum number of series to download
        log (bool, optional): If True each download is logged.
        cred (str, optional): path to the txt file holding the credentials

    Returns:

    None if log = False. 
    If log=True returns tuple (n_downloaded, failed_downloads, n_existed)

        - n_downloaded (int): number of series (zip-files) for which a download was
          attempted (The series that got skipped because they already exist are not 
          counted, this is the count that n_max limits).
        - failed_downloads (list): (filepath, error) tuples for each 
          downloaded file that is missing, empty or corrupted.
        - n_existed (int): number of series skipped because they already 
          exist in the target directory and are not corrupted.
    """
    n_downloaded = 0 # number of files for which a download was attempted (files that are getting skipped are excluded)
    n_existed = 0 # number of files that got skipped because they are already at the target repository and they are not corrupted
    failed_downloads = [] # (filepath, error) list of filepaths that failed to download, and the details of the error
    session = _xnat_session(cred)

    # Loop over all subjects in the project
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

        for exp in tqdm(experiments, desc='Scanning experiments..'):
            exp_id = exp['ID']

            # Continue if this experiment was not requested
            if experiment_label is not None:
                if exp['label'] != experiment_label:
                    continue

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

                # Continue if the scan has the right attributes
                download=True
                for key, val in attr.items():
                    if scan_attrs.get(key) not in val:
                        download=False
                if not download:
                    continue

                # Define download locations
                download_url = f"{xnat_url}/data/experiments/{exp_id}/scans/{scan_id}/resources/DICOM/files?format=zip"
                out_folder = os.path.join(output_dir, project_id, subj['label'], f"{exp['label']}")
                out_path = os.path.join(out_folder, f"series_{scan_id.zfill(2)}.zip")

                # Continue if the data have already been downloaded and the downloaded file is not corrupted
                if log:
                    logging.info(f"working with {out_path}")
                
                if os.path.exists(out_path) and _download_problem(out_path) is None:
                    n_existed += 1
                    if log:
                        logging.info(f"Skipping download because file {out_path} already exists and it is not corrupted")
                    continue

                # Download the scan
                os.makedirs(out_folder, exist_ok=True)
                r = session.get(download_url, stream=True)
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                if log:
                    # Define the error that occured during the file download process (None if there is no error)
                    file_donwload_problem = _download_problem(out_path) 

                    # If there downloaded file exists and not corrupted, declare successful download
                    if file_donwload_problem is None:
                        logging.info(f"Successfully downloaded {out_path}") 

                    # Record problem with the downloaded file if there is an error    
                    else:
                        logging.error(f"Error occured during download {out_path}: {file_donwload_problem}")
                        failed_downloads.append((out_path, file_donwload_problem))


                # Check exit condition
                n_downloaded += 1
                if n_max is not None:
                    if n_downloaded >= n_max:
                        # If logging is on, return the variables used for the logging summary
                        # This is needed to create the total summary of the downloaded files if multiple groups are being downloaded
                        if log: 
                            _log_summary(n_downloaded, failed_downloads, n_existed) 
                            return n_downloaded, failed_downloads, n_existed 
                        
                        return

    if log:
        _log_summary(n_downloaded, failed_downloads, n_existed)
        return n_downloaded, failed_downloads, n_existed
    return

                        



def _xnat_session(cred="user_XNAT.txt"):
    username, password = xnat_credentials(cred)
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)
    return session


def _create_user_file(cred="user_XNAT.txt"):
    # Ask the user for username and password
    username = input("Enter your username: ")
    password = input("Enter your password: ")

    # Create a text file and write the username and password to it
    with open(cred, "w") as file:
        file.write(f"Username: {username}\n")
        file.write(f"Password: {password}\n")


def _read_user_file(cred="user_XNAT.txt"):
    # Read the username and password from the text file
    with open(cred, "r") as file:
        lines = file.readlines()
        username = lines[0].split(":")[1].strip()
        password = lines[1].split(":")[1].strip()

    return username, password


def xnat_credentials(cred="user_XNAT.txt"):
    # Check if the file exists
    if os.path.exists(cred):
        # If the file exists, read username and password
        existing_username, existing_password = _read_user_file(cred)
    else:
        # If the file does not exist, create a new file and ask for username and password
        _create_user_file(cred)
        print("User file created successfully.")
        existing_username, existing_password = _read_user_file(cred)
    return existing_username, existing_password


# =================================================================================
# The following fucntions are added to introduce more detailed logging of 
# the downloaded files
# =================================================================================

def _download_problem(path):
    """
    Return the reason why the downloaded xnat zipfile at `path` is unusable, or None if it is fine
    Note: the code does not automatically check what file exactly is corrupted
    """
    try:
        with zipfile.ZipFile(path) as zf:
            if len(zf.namelist()) == 0:
                return "file found but empty (zip contains no files)"
    
    except zipfile.BadZipFile:
        return "file found but corrupted (unreadable zip contents)"
    
    except OSError as e:
        return f"file missing or unreadable ({e.strerror})"
    
    except Exception as e:
        return f"unexpected error occured ({e})"
    
    return None
    
def _log_summary(n_downloaded, failed_downloads, n_existed):
    "Record summary of all scanned files"
    n_failed = len(failed_downloads)
    logging.info(f"\n Total files iterated through: {n_downloaded + n_existed}"
                 f"\n Total files skipped because they already existed in target repository: {n_existed}"
                 f"\n Total files successfully downloaded: {n_downloaded - n_failed}"
                 f"\n Total files failed to download due to error: {n_failed}")
    if n_failed > 0:
        logging.info("List of files failed to download: ")
        for filepath, error in failed_downloads:
            logging.info(f"{filepath} : {error}")

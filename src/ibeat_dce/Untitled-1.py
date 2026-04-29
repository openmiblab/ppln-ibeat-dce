# =================================================================================
# 1. IMPORTS
# =================================================================================
import os
import time
import numpy as np
import napari
import pydicom
import mdreg

# =================================================================================
# 2. DATA LOADING FUNCTIONS
# =================================================================================
def load_single_slice_time_series(folder_path, slice_idx):
    dicom_headers = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'AcquisitionTime'):
                    t_str = str(ds.AcquisitionTime).strip()
                    if '.' not in t_str:
                        t_str += '.0'
                    t_str = t_str.zfill(8)

                    hh = float(t_str[:2]) * 3600.0
                    mm = float(t_str[2:4]) * 60.0
                    ss = float(t_str[4:])

                    dicom_headers.append({
                        'time_stamp': hh + mm + ss,
                        'z': float(ds.ImagePositionPatient[2]),
                        'path': filepath
                    })
            except:
                continue

    if not dicom_headers:
        raise ValueError("No valid DICOM files found")

    z_positions = sorted(list(set(d['z'] for d in dicom_headers)))
    target_z = z_positions[slice_idx]

    files = [d for d in dicom_headers if d['z'] == target_z]
    files.sort(key=lambda x: x['time_stamp'])

    start_time = files[0]['time_stamp']
    time_points = np.array([f['time_stamp'] - start_time for f in files])

    ref = pydicom.dcmread(files[0]['path'])
    rows, cols = ref.Rows, ref.Columns

    data = np.zeros((rows, cols, len(files)))

    for i, f in enumerate(files):
        ds = pydicom.dcmread(f['path'])
        img = ds.pixel_array.astype(np.float64)

        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept

        data[:, :, i] = img

    return data, time_points


def load_entire_time_series(folder_path):
    data, times = load_single_slice_time_series(folder_path, 0)
    return np.expand_dims(data, axis=0), times


def describe(pixel_array):
    return np.mean(pixel_array, axis=-1)

# =================================================================================
# 3. MASKING & INTENSITY
# =================================================================================
def draw_arterial_input(avg):
    viewer = napari.Viewer()
    viewer.add_image(avg)
    labels = viewer.add_labels(np.zeros_like(avg, dtype=np.uint8))
    labels.mode = "paint"
    labels.brush_size = 5

    print("Draw AIF mask then close.")
    napari.run()

    return (labels.data > 0)


def calculate_aif(data, mask):
    data = np.squeeze(data)
    mask = np.squeeze(mask)

    curve = []
    for t in range(data.shape[-1]):
        curve.append(np.mean(data[:, :, t][mask]))

    return np.array(curve)


def draw_kidney_masks(avg):
    viewer = napari.Viewer(title="1=Left Kidney, 2=Right Kidney")
    viewer.add_image(avg)
    labels = viewer.add_labels(np.zeros_like(avg, dtype=np.uint8))

    print("Draw LEFT (1) and RIGHT (2) kidneys, then close.")
    napari.run()

    left = labels.data == 1
    right = labels.data == 2

    return left, right


def extract_curves(data, left, right):
    n = data.shape[-1]
    l = np.zeros(n)
    r = np.zeros(n)

    for t in range(n):
        frame = data[:, :, t]
        if np.any(left):
            l[t] = np.mean(frame[left])
        if np.any(right):
            r[t] = np.mean(frame[right])

    return l, r

# =================================================================================
# 4. CSV SAVE
# =================================================================================
def save_csv(path, time, left, right):
    arr = np.column_stack((time, left, right))
    header = "Time(s),Left_Kidney,Right_Kidney"
    np.savetxt(path, arr, delimiter=",", header=header, comments='')
    print("Saved:", path)

# =================================================================================
# 5. MDR
# =================================================================================
def run_mdr(data, times, aif, options, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    coreg, fit, _, _ = mdreg.fit(
        data,
        fit_image={
            'func': mdreg.fit_2cm_lin,
            'time': times,
            'aif': aif,
            'baseline': 10
        },
        fit_coreg=options,
        maxit=3,
        path=out_dir,
        verbose=1
    )

    print("Time:", round(time.time() - t0), "s")

    return coreg

# =================================================================================
# MAIN
# =================================================================================
if __name__ == "__main__":

    results_folder = r"C:\Users\eia21frd\Documents\RESULTS\hyperparameters\followup"
    dicom_folder = r"C:\Users\eia21frd\Documents\DATA\hyperparametre\iBE-3128-136_followup\series_43\iBE-3128-136_followup\scans\43-DCE_kidneys_cor_oblique_fb\resources\DICOM\files"

    os.makedirs(results_folder, exist_ok=True)

    # ---- AIF ----
    data0, t0 = load_entire_time_series(dicom_folder)
    avg0 = describe(data0)

    aif_mask = draw_arterial_input(avg0)
    aif = calculate_aif(data0, aif_mask)

    # ---- Slice 4 ----
    data4, t4 = load_single_slice_time_series(dicom_folder, 4)
    avg4 = np.mean(data4, axis=-1)

    aif_interp = np.interp(t4, t0, aif)

    left_mask, right_mask = draw_kidney_masks(avg4)

    # ---- RAW CSV ----
    l_raw, r_raw = extract_curves(data4, left_mask, right_mask)
    save_csv(os.path.join(results_folder, "raw.csv"), t4, l_raw, r_raw)

    # ---- METHODS ----
    tests = [
        ("skimage", {'progress_bar': True}),
        ("elastix_25", {'package': 'elastix', 'FinalGridSpacingInPhysicalUnits': 25.0}),
        ("elastix_50", {'package': 'elastix', 'FinalGridSpacingInPhysicalUnits': 50.0}),
        ("elastix_75", {'package': 'elastix', 'FinalGridSpacingInPhysicalUnits': 75.0}),
        ("ants", {'package': 'ants', 'type_of_transform': 'SyNOnly'})
    ]

    for name, opts in tests:
        print("\nRunning", name)

        out_dir = os.path.join(results_folder, f"MDR_{name}")
        corrected = run_mdr(data4, t4, aif_interp, opts, out_dir)

        l, r = extract_curves(corrected, left_mask, right_mask)

        save_csv(
            os.path.join(out_dir, f"{name}.csv"),
            t4,
            l,
            r
        )

    print("\n✅ DONE — CSVs only, no extra files.")
"""DICOM loading and anatomical slice ordering utilities."""

from pathlib import Path

import numpy as np
import pydicom


def get_slice_position(ds) -> float:
    """Return the geometric position of a DICOM slice along its slice normal."""
    if hasattr(ds, "ImageOrientationPatient") and hasattr(ds, "ImagePositionPatient"):
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        position = np.asarray(ds.ImagePositionPatient, dtype=float)
        row = orientation[:3]
        col = orientation[3:]
        normal = np.cross(row, col)
        return float(np.dot(position, normal))

    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)

    if hasattr(ds, "InstanceNumber"):
        return float(ds.InstanceNumber)

    return 0.0


def load_dicom_series(series_path):
    """Load a DICOM series, skipping unreadable slices, and sort geometrically."""
    series_path = Path(series_path)
    slices = []
    bad_files = []

    for path in series_path.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(path)
            image = ds.pixel_array.astype(np.float32)

            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            image = image * slope + intercept

            slices.append((get_slice_position(ds), image))
        except Exception as exc:
            bad_files.append((str(path), str(exc)))
            continue

    if not slices:
        raise RuntimeError(f"No valid DICOM slices found in {series_path}")

    slices.sort(key=lambda item: item[0])
    volume = np.stack([image for _, image in slices], axis=0)

    if bad_files:
        print(f"[WARNING] {len(bad_files)} DICOM(s) ignored in {series_path}")

    return volume

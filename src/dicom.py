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
    """Load, geometrically sort and stack a DICOM series as [slices, height, width]."""
    series_path = Path(series_path)
    slices = []

    for path in series_path.glob("*.dcm"):
        ds = pydicom.dcmread(path)
        image = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image = image * slope + intercept

        slices.append((get_slice_position(ds), image))

    if not slices:
        raise ValueError(f"No DICOM slices found in {series_path}")

    slices.sort(key=lambda item: item[0])
    return np.stack([image for _, image in slices], axis=0)

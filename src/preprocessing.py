"""MRI volume preprocessing utilities."""

import cv2
import numpy as np


def select_slices(volume, num_slices=16):
    """Uniformly select a fixed number of slices from a volume."""
    if volume.ndim != 3 or volume.shape[0] == 0:
        raise ValueError("Expected a non-empty volume with shape [N, H, W].")

    indices = np.linspace(0, volume.shape[0] - 1, num_slices)
    indices = np.rint(indices).astype(int)
    return volume[indices]


def normalize_slice(image):
    """Percentile clip a slice and scale its intensities to [0, 1]."""
    image = image.astype(np.float32)
    low, high = np.percentile(image, (1, 99))

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def resize_volume(volume, size=288):
    """Normalize and resize every slice in a volume."""
    processed = []

    for image in volume:
        image = normalize_slice(image)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        processed.append(image)

    return np.stack(processed, axis=0).astype(np.float32)


def preprocess_volume(volume, num_slices=16, size=288):
    """Select, normalize and resize an MRI volume."""
    volume = select_slices(volume, num_slices=num_slices)
    return resize_volume(volume, size=size)

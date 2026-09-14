from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dicom import load_dicom_series
from .preprocessing import preprocess_volume


class KneeMRIDatasetV2(Dataset):
    """Multiplanar knee MRI dataset for DINOv2-based training.

    Each sample returns sagittal, coronal and axial tensors with shape
    [S, 3, H, W], a 12-target tensor and a gold/pseudo supervision flag.
    """

    def __init__(
        self,
        dataframe,
        root_dir,
        targets,
        num_slices=16,
        image_size=288,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.targets = list(targets)
        self.num_slices = num_slices
        self.image_size = image_size

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        ).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.dataframe)

    def _load_plane(self, study_uid, series_uid):
        series_path = self.root_dir / study_uid / series_uid
        volume = load_dicom_series(series_path)
        volume = preprocess_volume(
            volume,
            num_slices=self.num_slices,
            image_size=self.image_size,
        )

        volume = torch.from_numpy(volume).float().clamp(0.0, 1.0)
        volume = volume.unsqueeze(1).repeat(1, 3, 1, 1)
        volume = (volume - self.mean) / self.std
        return volume

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        study_uid = row["StudyInstanceUID"]

        sagittal = self._load_plane(
            study_uid,
            row["Sagittal_SeriesInstanceUID"],
        )
        coronal = self._load_plane(
            study_uid,
            row["Coronal_SeriesInstanceUID"],
        )
        axial = self._load_plane(
            study_uid,
            row["Axial_SeriesInstanceUID"],
        )

        targets = torch.tensor(
            row[self.targets].astype(float).values,
            dtype=torch.float32,
        )

        return {
            "StudyInstanceUID": study_uid,
            "sagittal": sagittal,
            "coronal": coronal,
            "axial": axial,
            "targets": targets,
            "is_gold": torch.tensor(float(row["is_gold"])),
        }

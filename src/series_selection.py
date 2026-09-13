"""Multiplanar MRI series-selection utilities."""

import pandas as pd

PLANES = ("Sagittal", "Coronal", "Axial")


def choose_best_series(candidates):
    """Choose the preferred sequence for one anatomical plane.

    Priority is fluid-sensitive plus fat-suppressed, then fluid-sensitive,
    then fat-suppressed, followed by any available series.
    """
    if len(candidates) == 0:
        return None

    candidates = candidates.copy()
    candidates["Fluid_Sensitive"] = (
        candidates["Fluid_Sensitive"].fillna(0).astype(int)
    )
    candidates["Fat_Suppression"] = (
        candidates["Fat_Suppression"].fillna(0).astype(int)
    )
    candidates["priority"] = (
        candidates["Fluid_Sensitive"] * 2 + candidates["Fat_Suppression"]
    )

    candidates = candidates.sort_values(
        ["priority", "Fluid_Sensitive", "Fat_Suppression"],
        ascending=False,
    )
    return candidates.iloc[0]


def choose_series_per_plane(study_series):
    """Choose one representative series for each anatomical plane."""
    selected = {}

    for plane in PLANES:
        candidates = study_series[study_series["Anatomical_Plane"] == plane]
        selected[plane] = choose_best_series(candidates)

    return selected


def build_series_manifest(series_df):
    """Build one row per study containing the selected series for all planes."""
    rows = []

    for study_uid, group in series_df.groupby("StudyInstanceUID"):
        selected = choose_series_per_plane(group)
        row = {"StudyInstanceUID": study_uid}

        for plane in PLANES:
            series = selected[plane]
            row[f"{plane}_SeriesUID"] = None if series is None else series["SeriesInstanceUID"]
            row[f"{plane}_Fluid"] = None if series is None else int(series["Fluid_Sensitive"])
            row[f"{plane}_FatSup"] = None if series is None else int(series["Fat_Suppression"])

        rows.append(row)

    return pd.DataFrame(rows)

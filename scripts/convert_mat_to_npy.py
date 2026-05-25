"""Convert MATLAB .mat and .xlsx data files to Python-native formats.

Run once to populate data/processed/ from data/raw/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml


def convert_mat_file(mat_path: Path, output_dir: Path) -> dict:
    """Convert a single .mat file to .npy files. Returns manifest entries."""
    data = sio.loadmat(str(mat_path))
    entries = []
    stem = mat_path.stem

    for key, value in data.items():
        if key.startswith("_"):
            continue
        if not hasattr(value, "shape"):
            continue
        # Skip duplicate entries (e.g., aero_cfg2.mat has both 'alpha' and 'alpha_cfg2')
        out_name = f"{stem}_{key}.npy"
        out_path = output_dir / out_name
        np.save(str(out_path), value.astype(np.float64))
        entries.append({
            "file": out_name,
            "source": str(mat_path),
            "key": key,
            "shape": list(value.shape),
            "dtype": "float64",
        })
        print(f"  {out_name}: shape={value.shape}")

    return entries


def convert_xlsx_file(xlsx_path: Path, output_dir: Path) -> dict:
    """Convert an Excel file to CSV."""
    df = pd.read_excel(str(xlsx_path))
    out_name = xlsx_path.stem + ".csv"
    out_path = output_dir / out_name
    df.to_csv(str(out_path), index=False)
    print(f"  {out_name}: {df.shape[0]} rows x {df.shape[1]} cols")
    return {
        "file": out_name,
        "source": str(xlsx_path),
        "shape": list(df.shape),
        "columns": list(df.columns),
    }


def main():
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    # Copy raw .mat and .xlsx files from mdl/ to data/raw/ if not already there
    mdl_dir = project_root / "mdl"
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True)

    for ext in ("*.mat", "*.xlsx"):
        for f in mdl_dir.glob(ext):
            dest = raw_dir / f.name
            if not dest.exists():
                import shutil
                shutil.copy2(str(f), str(dest))
                print(f"Copied {f.name} to data/raw/")

    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"description": "Converted tailsitter aerodynamic and propulsion data", "files": []}

    # Convert .mat files
    for mat_file in sorted(raw_dir.glob("*.mat")):
        print(f"\nConverting {mat_file.name}...")
        try:
            entries = convert_mat_file(mat_file, processed_dir)
            manifest["files"].extend(entries)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # Convert .xlsx files
    for xlsx_file in sorted(raw_dir.glob("*.xlsx")):
        print(f"\nConverting {xlsx_file.name}...")
        try:
            entry = convert_xlsx_file(xlsx_file, processed_dir)
            manifest["files"].append(entry)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # Save manifest
    manifest_path = processed_dir / "manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, default_flow_style=False)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Total files: {len(manifest['files'])}")


if __name__ == "__main__":
    main()

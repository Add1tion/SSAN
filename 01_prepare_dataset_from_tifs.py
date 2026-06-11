"""
Preprocessing script for geochemical raster data and sample patch generation.

This script reads all `.tif` geochemical feature layers from `geochemical_data/`,
stacks them into a multi-channel data cube, and extracts fixed-size image patches
around labeled sample coordinates stored as `.npy` files in `train_extend/` and
`verify_extend/`. Each sample file is expected to contain at least two values:
the row and column coordinate of the sample point. The script supports two labels,
`0` and `1`, organized in subfolders such as `train_extend/0`, `train_extend/1`,
`verify_extend/0`, and `verify_extend/1`.

The generated outputs include training and validation patches, labels, sample
coordinates, feature-wise normalization parameters, and an optional normalized
geological weight map derived from `fault.tif`. These outputs are saved as `.npy`
files and are intended to be used by the subsequent model training script.

Usage:

1. Place geochemical `.tif` files in `geochemical_data/`.
2. Place coordinate `.npy` sample files in the corresponding label folders under
   `train_extend/` and `verify_extend/`.
3. Place the geological constraint raster as `fault.tif` if geological weights
   are needed.
4. Adjust `WINDOW_SIZE`, `MASK_LOW_VALUES`, and `MASK_ZERO_DOMINANT_PIXELS`
   if required.
5. Run the script directly with Python.

Main outputs:

* `X_train.npy`, `y_train.npy`
* `X_val.npy`, `y_val.npy`
* `coords_train.npy`, `coords_val.npy`
* `feats_mean.npy`, `feats_std.npy`
* `geological_weights.npy` if `fault.tif` is available and valid
  """

import os
from pathlib import Path
import numpy as np
import tifffile
from typing import List, Tuple


TIF_DIR = Path('geochemical_data')
TRAIN_DIR = Path('train_extend')
VAL_DIR = Path('verify_extend')

GEO_WEIGHT_TIF = Path('fault.tif')


OUTPUT_DIR = Path('.')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


WINDOW_SIZE = 9


MASK_LOW_VALUES = False

MASK_ZERO_DOMINANT_PIXELS = False


def create_data_cube(tif_dir: Path) -> Tuple[np.ndarray, np.ndarray]:

    tif_files = sorted([f for f in tif_dir.glob('*.tif') if f.is_file()])
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in '{tif_dir}'. Please check the path.")

    feature_cubes = []
    for f in tif_files:
        data = tifffile.imread(f)

        if MASK_LOW_VALUES:
            data = np.where(data < 1.0, 0.0, data)

        feature_cubes.append(data)

    data_cube = np.stack(feature_cubes, axis=-1)  # shape: (H, W, 39)

    if MASK_ZERO_DOMINANT_PIXELS:
        zero_count = np.sum(data_cube == 0, axis=-1)
        valid_mask = zero_count <= 15  # shape: (H, W)
    else:
        valid_mask = np.ones(data_cube.shape[:2], dtype=bool)

    return data_cube, valid_mask


def load_samples_from_directory(
        sample_dir: Path,
        label: int,
        data_cube: np.ndarray,
        window_size: int,
        valid_mask: np.ndarray
) -> Tuple[List[np.ndarray], List[int], List[Tuple[int, int]]]:

    if not sample_dir.is_dir():

        return [], [], []

    patches, labels, coords = [], [], []
    H, W, B = data_cube.shape
    half = window_size // 2

    for npy_file in sorted(sample_dir.glob('*.npy')):
        arr = np.load(npy_file)

        if arr.ndim == 1 and arr.shape[0] >= 2:
            r, c = int(arr[0]), int(arr[1])

            if (r - half) < 0 or (r + half + 1) > H or \
                    (c - half) < 0 or (c + half + 1) > W or not valid_mask[r, c]:
                continue

            patch = data_cube[r - half: r + half + 1, c - half: c + half + 1, :]
            patches.append(patch)
            labels.append(label)
            coords.append((r, c))


        elif arr.shape == (window_size, window_size, B):

            continue
        else:
            print(f"Warning: Skipping file {npy_file.name}. Shape {arr.shape} is not valid.")

    return patches, labels, coords


def load_dataset(
        base_dir: Path,
        data_cube: np.ndarray,
        valid_mask: np.ndarray,
        window_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    all_patches = []
    all_labels = []
    all_coords = []

    for label in [0, 1]:
        sample_dir = base_dir / str(label)

        patches, labels, coords = load_samples_from_directory(sample_dir, label, data_cube, window_size, valid_mask)
        all_patches.extend(patches)
        all_labels.extend(labels)
        all_coords.extend(coords)


    return np.array(all_patches, dtype=np.float32), \
        np.array(all_labels, dtype=np.int64), \
        np.array(all_coords, dtype=np.int64)


def main():

    data_cube, valid_mask = create_data_cube(TIF_DIR)



    X_train, y_train, coords_train = load_dataset(TRAIN_DIR, data_cube, valid_mask, WINDOW_SIZE)


    X_val, y_val, coords_val = load_dataset(VAL_DIR, data_cube, valid_mask, WINDOW_SIZE)


    if len(X_train) == 0:
        return

    np.save(OUTPUT_DIR / 'X_train.npy', X_train)
    np.save(OUTPUT_DIR / 'y_train.npy', y_train)
    np.save(OUTPUT_DIR / 'X_val.npy', X_val)
    np.save(OUTPUT_DIR / 'y_val.npy', y_val)

    np.save(OUTPUT_DIR / 'coords_train.npy', coords_train)
    np.save(OUTPUT_DIR / 'coords_val.npy', coords_val)

    if not GEO_WEIGHT_TIF.exists():

        return

    geo_weights_map = tifffile.imread(GEO_WEIGHT_TIF).astype(np.float32)



    if geo_weights_map.shape != data_cube.shape[:2]:

        return


    min_val = np.min(geo_weights_map)
    max_val = np.max(geo_weights_map)

    if max_val > min_val:
        geo_weights_map_normalized = (geo_weights_map - min_val) / (max_val - min_val)

    else:

        geo_weights_map_normalized = np.full_like(geo_weights_map, 0.5, dtype=np.float32)

    output_path = OUTPUT_DIR / 'geological_weights.npy'
    np.save(output_path, geo_weights_map_normalized)


if __name__ == "__main__":
    main()

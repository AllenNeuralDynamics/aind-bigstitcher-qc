#!/usr/bin/env python3
"""
Pack per-crop OME-Zarr volumes into a 2D mosaic by updating their translation metadata.

The script scans a root folder for ``*.ome.zarr`` containers, optionally trims them
along the Z axis to the minimal common depth, computes an optimal XY layout using
``rectpack``, and writes updated translation vectors back into the NGFF metadata.

Example
-------
    python pack_ome_zarr_mosaic.py \\
        --root /results/ome-zarr \\
        --padding 20 \\
        --manifest /results/ome-zarr/mosaic_manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from rectpack import MaxRectsBssf, newPacker


@dataclass
class CropInfo:
    """Holds metadata for a single OME-Zarr crop."""

    path: Path
    group: zarr.hierarchy.Group
    dataset_path: str
    axes: List[Dict]
    shape: Tuple[int, ...]
    chunks: Tuple[int, ...]
    dtype: np.dtype
    compressor: Optional[object]
    fill_value: Optional[object]
    order: str
    scale: np.ndarray
    translation: np.ndarray


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory containing per-crop *.ome.zarr containers.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Optional padding (in physical units) between packed crops.",
    )
    parser.add_argument(
        "--trim-z",
        action="store_true",
        help="Physically trim all crops to the minimal common Z-depth.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON file summarising final translations.",
    )
    return parser.parse_args(argv)


def find_ome_zarrs(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.ome.zarr") if path.is_dir())


def load_crop(path: Path) -> CropInfo:
    group = zarr.open_group(str(path), mode="a")
    multiscales = group.attrs.get("multiscales")
    if not multiscales:
        raise ValueError(f"{path} has no 'multiscales' attribute")
    meta = multiscales[0]
    datasets = meta.get("datasets")
    if not datasets:
        raise ValueError(f"{path} has no datasets in NGFF metadata")
    dataset_entry = datasets[0]
    dataset_path = dataset_entry["path"]
    arr = group[dataset_path]
    axes = meta.get("axes", [])
    transforms = dataset_entry.get("coordinateTransformations", [])
    scale = _find_transform_vector(transforms, "scale")
    translation = _find_transform_vector(transforms, "translation")
    return CropInfo(
        path=path,
        group=group,
        dataset_path=dataset_path,
        axes=axes,
        shape=arr.shape,
        chunks=arr.chunks,
        dtype=arr.dtype,
        compressor=arr.compressor,
        fill_value=arr.fill_value,
        order=arr.order,
        scale=scale,
        translation=translation,
    )


def _find_transform_vector(transforms: Sequence[Dict], transform_type: str) -> np.ndarray:
    for transform in transforms:
        if transform.get("type") == transform_type:
            key = "scale" if transform_type == "scale" else "translation"
            values = transform.get(key)
            if values is None:
                continue
            return np.asarray(values, dtype=np.float64)
    raise ValueError(f"Missing '{transform_type}' transform in metadata")


def axis_indices(axes: Sequence[Dict]) -> Dict[str, int]:
    """Map axis labels to indices."""
    mapping: Dict[str, int] = {}
    for idx, axis in enumerate(axes):
        name = (axis.get("name") or "").lower()
        axis_type = (axis.get("type") or "").lower()
        if name.startswith("x"):
            mapping["x"] = idx
        elif name.startswith("y"):
            mapping["y"] = idx
        elif name.startswith("z"):
            mapping["z"] = idx
        elif axis_type == "space":
            # Spatial axes without explicit xyz names arrive as 'space' entries
            # in reverse order Z, Y, X in NGFF 0.4 style metadata.
            spatial_count = sum(1 for key in mapping if key in {"x", "y", "z"})
            if spatial_count == 0:
                mapping["z"] = idx
            elif spatial_count == 1:
                mapping["y"] = idx
            else:
                mapping["x"] = idx
    # If no axes info is available fall back to NGFF TCZYX order.
    if not mapping:
        mapping = {"t": 0, "c": 1, "z": 2, "y": 3, "x": 4}
    return mapping


def ensure_flat_cache(group: zarr.hierarchy.Group) -> None:
    # Force persistence metadata to disk when supported.
    store = getattr(group, "store", None)
    if store is not None and hasattr(store, "flush"):
        store.flush()


def trim_crops_z(crops: List[CropInfo], z_key: str = "z") -> None:
    """Trim each crop along Z to match the minimum depth."""
    if not crops:
        return
    z_lengths = []
    z_indices = []
    for crop in crops:
        indices = axis_indices(crop.axes)
        z_idx = indices.get(z_key, None)
        if z_idx is None:
            raise ValueError(f"Unable to determine Z axis for {crop.path}")
        z_lengths.append(crop.shape[z_idx])
        z_indices.append(z_idx)

    target_z = min(z_lengths)
    if all(length == target_z for length in z_lengths):
        return

    for crop, z_idx, current_z in zip(crops, z_indices, z_lengths):
        if current_z == target_z:
            continue
        start = (current_z - target_z) // 2
        end = start + target_z
        _trim_dataset_along_axis(crop, axis=z_idx, start=start, end=end)
        crop.shape = tuple(
            target_z if i == z_idx else dim for i, dim in enumerate(crop.shape)
        )
        # Update translation to account for removed slices.
        crop.translation[z_idx] += start * crop.scale[z_idx]
        _write_translation(crop)


def _trim_dataset_along_axis(crop: CropInfo, axis: int, start: int, end: int) -> None:
    source = crop.group[crop.dataset_path]
    new_shape = list(source.shape)
    new_shape[axis] = end - start
    tmp_path = f"{crop.dataset_path}__tmp"
    if tmp_path in crop.group:
        del crop.group[tmp_path]
    target = crop.group.create_dataset(
        tmp_path,
        shape=tuple(new_shape),
        chunks=source.chunks,
        dtype=source.dtype,
        compressor=source.compressor,
        filters=source.filters,
        fill_value=source.fill_value,
        order=source.order,
        overwrite=True,
    )
    slices_new = [slice(None)] * source.ndim
    for dest_slices in _chunk_slices(new_shape, source.chunks):
        for dim, slc in enumerate(dest_slices):
            slices_new[dim] = slc
        source_slices = list(slices_new)
        src_slice = slice(start + dest_slices[axis].start, start + dest_slices[axis].stop)
        source_slices[axis] = src_slice
        target[tuple(slices_new)] = source[tuple(source_slices)]

    del crop.group[crop.dataset_path]
    crop.group.move(tmp_path, crop.dataset_path)


def _chunk_slices(shape: Sequence[int], chunks: Sequence[int]) -> Iterable[Tuple[slice, ...]]:
    chunk_sizes = [chunk if chunk else dim for dim, chunk in zip(shape, chunks)]
    ranges = [range(0, dim, chunk) for dim, chunk in zip(shape, chunk_sizes)]
    for offsets in product(*ranges):
        slices = []
        for offset, chunk, dim in zip(offsets, chunk_sizes, shape):
            slices.append(slice(offset, min(offset + chunk, dim)))
        yield tuple(slices)


def pack_positions(
    crops: List[CropInfo], padding: float
) -> Dict[Path, Tuple[float, float]]:
    rectangles = []
    packer = newPacker(rotation=False)
    total_width = 0.0
    max_height = 0.0
    for idx, crop in enumerate(crops):
        indices = axis_indices(crop.axes)
        x_idx = indices.get("x")
        y_idx = indices.get("y")
        if x_idx is None or y_idx is None:
            raise ValueError(f"Unable to determine XY axes for {crop.path}")
        width = crop.shape[x_idx] * crop.scale[x_idx] + padding
        height = crop.shape[y_idx] * crop.scale[y_idx] + padding
        rectangles.append((idx, width, height))
        packer.add_rect(width, height, idx)
        total_width += width
        max_height = max(max_height, height)

    if not rectangles:
        return {}

    packer.add_bin(total_width, max_height)
    packer.pack()
    positions: Dict[Path, Tuple[float, float]] = {}
    for rect in packer.rect_list():
        _, x, y, w, h, idx = rect
        crop = crops[idx]
        offset = padding * 0.5
        positions[crop.path] = (x + offset, y + offset)
    return positions


def update_translations(
    crops: List[CropInfo], positions: Dict[Path, Tuple[float, float]]
) -> Dict[str, Dict[str, float]]:
    manifest: Dict[str, Dict[str, float]] = {}
    for crop in crops:
        if crop.path not in positions:
            continue
        x_pos, y_pos = positions[crop.path]
        indices = axis_indices(crop.axes)
        x_idx = indices.get("x")
        y_idx = indices.get("y")
        if x_idx is None or y_idx is None:
            continue
        crop.translation[x_idx] = x_pos
        crop.translation[y_idx] = y_pos
        _write_translation(crop)
        manifest[str(crop.path)] = {
            "translation": crop.translation.tolist(),
            "scale": crop.scale.tolist(),
            "shape": list(crop.shape),
        }
    return manifest


def _write_translation(crop: CropInfo) -> None:
    multiscales = crop.group.attrs["multiscales"]
    dataset_entry = multiscales[0]["datasets"][0]
    transforms = dataset_entry.get("coordinateTransformations", [])
    for transform in transforms:
        if transform.get("type") == "translation":
            transform["translation"] = crop.translation.tolist()
    crop.group.attrs["multiscales"] = multiscales
    ensure_flat_cache(crop.group)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.root.is_dir():
        raise SystemExit(f"{args.root} is not a directory")

    ome_zarrs = find_ome_zarrs(args.root)
    if not ome_zarrs:
        raise SystemExit(f"No *.ome.zarr directories found under {args.root}")

    crops = [load_crop(path) for path in ome_zarrs]
    if args.trim_z:
        trim_crops_z(crops)

    positions = pack_positions(crops, args.padding)
    manifest = update_translations(crops, positions)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()

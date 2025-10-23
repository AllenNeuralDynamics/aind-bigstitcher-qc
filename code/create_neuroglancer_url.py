#!/usr/bin/env python3
"""
Generate a Neuroglancer URL that visualises all per-crop OME-Zarr volumes
in a single image layer using a custom RGB shader.

Each crop is assumed to contain three channels (RGB) that should be rendered
as a colour image. The script emits a Neuroglancer state URL referencing every
crop as a sub-source and applies the stored NGFF translation metadata to place
them in the global mosaic coordinate system.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr


@dataclass
class CropInfo:
    path: Path
    dataset_path: str
    axes: List[Dict]
    shape: Tuple[int, ...]
    scale: np.ndarray
    translation: np.ndarray


DEFAULT_SHADER = """\
void main () {
  emitRGB(vec3(toNormalized(getDataValue(0)),
               toNormalized(getDataValue(1)),
               toNormalized(getDataValue(2))));
}
""".strip()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory containing *.ome.zarr crops.",
    )
    parser.add_argument(
        "--viewer",
        default="https://neuroglancer.demo.appspot.com/#!",
        help="Base Neuroglancer viewer URL prefix.",
    )
    parser.add_argument(
        "--layer-name",
        default="mosaic",
        help="Name for the Neuroglancer layer.",
    )
    parser.add_argument(
        "--url-template",
        default="zarr://{data_path}",
        help=(
            "Template used to convert each crop path into a Neuroglancer volume URL. "
            "Available placeholders: {data_path} (path after applying prefix), "
            "{dataset} (dataset path)."
        ),
    )
    parser.add_argument(
        "--path-prefix",
        type=str,
        help=(
            "Optional prefix to replace the root filesystem path when generating URLs. "
            "Example: https://my-bucket/data"
        ),
    )
    parser.add_argument(
        "--shader-file",
        type=Path,
        help="Optional file containing a Neuroglancer shader. If omitted, a default RGB shader is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the URL (and state JSON) to disk.",
    )
    return parser.parse_args(argv)


def find_ome_zarrs(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.ome.zarr") if path.is_dir())


def load_crop(path: Path) -> CropInfo:
    group = zarr.open_group(str(path), mode="r")
    multiscales = group.attrs.get("multiscales")
    if not multiscales:
        raise ValueError(f"{path} missing 'multiscales' attribute")
    meta = multiscales[0]
    datasets = meta.get("datasets")
    if not datasets:
        raise ValueError(f"{path} missing NGFF datasets")
    dataset = datasets[0]
    arr = group[dataset["path"]]
    transforms = dataset.get("coordinateTransformations", [])
    scale = _find_transform(transforms, "scale")
    translation = _find_transform(transforms, "translation")
    return CropInfo(
        path=path,
        dataset_path=dataset["path"],
        axes=meta.get("axes", []),
        shape=arr.shape,
        scale=scale,
        translation=translation,
    )


def _find_transform(transforms: Sequence[Dict], key: str) -> np.ndarray:
    field = "scale" if key == "scale" else "translation"
    for transform in transforms:
        if transform.get("type") == key:
            values = transform.get(field)
            if values is None:
                continue
            return np.asarray(values, dtype=np.float64)
    raise ValueError(f"Missing transform '{key}' in dataset metadata")


def axis_indices(axes: Sequence[Dict]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, axis in enumerate(axes):
        name = str(axis.get("name", "")).lower()
        axis_type = str(axis.get("type", "")).lower()
        if name.startswith("x"):
            mapping["x"] = idx
        elif name.startswith("y"):
            mapping["y"] = idx
        elif name.startswith("z"):
            mapping["z"] = idx
        elif axis_type == "space":
            spatial_count = sum(1 for key in mapping if key in {"x", "y", "z"})
            if spatial_count == 0:
                mapping["z"] = idx
            elif spatial_count == 1:
                mapping["y"] = idx
            else:
                mapping["x"] = idx
    if not mapping:
        mapping = {"t": 0, "c^": 1, "z": 2, "y": 3, "x": 4}
    return mapping


def build_source_url(
    crop: CropInfo,
    root: Path,
    path_prefix: Optional[str],
    template: str,
) -> str:
    if path_prefix:
        rel = crop.path.relative_to(root)
        data_path = f"{path_prefix.rstrip('/')}/{rel.as_posix()}"
    else:
        data_path = crop.path.as_posix()
    return template.format(data_path=data_path, dataset=crop.dataset_path)


def load_shader(shader_file: Optional[Path]) -> str:
    if shader_file is None:
        return DEFAULT_SHADER
    return shader_file.read_text(encoding="utf-8")


def build_state(
    crops: List[CropInfo],
    root: Path,
    path_prefix: Optional[str],
    template: str,
    layer_name: str,
    shader: str,
) -> Dict:
    if not crops:
        raise ValueError("No crops found to include in Neuroglancer state")

    axis_infos = extract_axis_info(crops[0])
    canonical_axes = determine_axes_order(axis_infos)

    ndim = len(axis_infos)
    sources: List[Dict] = []
    dimension_map = build_dimension_map(axis_infos)

    for crop in crops:
        url = build_source_url(crop, root, path_prefix, template)
        matrix = [[1.0 if i == j else 0.0 for j in range(ndim + 1)] for i in range(ndim)]
        transform = {
            "matrix": matrix,
            "outputDimensions": {key: value[:] for key, value in dimension_map.items()},
        }
        sources.append(
            {
                "url": url,
                "transform": transform,
                "subsourceId": crop.path.name,
            }
        )
    min_corner, max_corner = bounding_box(crops, canonical_axes)
    center = (min_corner + max_corner) * 0.5
    extent = max((max_corner - min_corner).tolist() or [1.0])
    cross_section = max(1.0, extent * 0.5)
    projection = max(1024.0, extent * 2.0)

    layer: Dict = {
        "type": "image",
        "name": layer_name,
        "visible": True,
        "shader": shader,
        "source": sources,
    }

    state = {
        "dimensions": {key: value[:] for key, value in dimension_map.items()},
        "position": center.tolist()[: len(canonical_axes)],
        "crossSectionScale": cross_section,
        "projectionScale": projection,
        "layers": [layer],
        "selectedLayer": {"visible": True, "layer": layer_name},
        "layout": "xy",
    }
    return state


@dataclass
class AxisInfo:
    label: str
    canonical: str
    unit: str
    scale: float
    index: int


def determine_axes_order(axis_infos: List[AxisInfo]) -> List[str]:
    label_to_info = {info.canonical: info for info in axis_infos}
    order: List[str] = []
    for key in ("x", "y", "z", "t", "c^"):
        if key in label_to_info:
            order.append(key)
    for info in axis_infos:
        if info.canonical not in order:
            order.append(info.canonical)
    return order


def axis_label(axis: Dict, default_prefix: str = "axis") -> str:
    name = axis.get("name")
    if name:
        return str(name)
    axis_type = axis.get("type")
    if axis_type:
        if axis_type == "channel":
            return "c^"
        if axis_type == "time":
            return "t"
        if axis_type == "space":
            # Will be mapped later, default placeholder.
            return "space"
        return str(axis_type)
    return default_prefix


def canonical_axis_label(label: str) -> str:
    lower = label.lower()
    if lower.startswith("x"):
        return "x"
    if lower.startswith("y"):
        return "y"
    if lower.startswith("z"):
        return "z"
    if lower.startswith("t"):
        return "t"
    if lower.startswith("c"):
        return "c^"
    return label


def build_dimension_map(axis_infos: List[AxisInfo]) -> Dict[str, List[float]]:
    default_units = {
        "x": "m",
        "y": "m",
        "z": "m",
        "t": "s",
        "c^": "",
    }
    dims: Dict[str, List[float]] = {}

    for info in axis_infos:
        key = info.canonical
        unit = default_units.get(key, "")
        scale_value = info.scale
        if key in {"x", "y", "z"}:
            scale_value *= 1e-6
        dims[key] = [scale_value, unit]

    for key, unit in default_units.items():
        if key not in dims:
            default_scale = 1e-6 if key in {"x", "y", "z"} else 1.0
            dims[key] = [default_scale, unit]

    return dims


def extract_axis_info(crop: CropInfo) -> List[AxisInfo]:
    base_mapping = axis_indices(crop.axes)
    reverse_mapping = {idx: label for label, idx in base_mapping.items()}
    infos: List[AxisInfo] = []
    for idx, axis in enumerate(crop.axes):
        label = reverse_mapping.get(idx)
        if label is None:
            label = axis_label(axis, f"axis{idx}")
        canonical = canonical_axis_label(label)
        unit = axis.get("unit") or ""
        scale_value = float(crop.scale[idx]) if idx < len(crop.scale) else 1.0
        infos.append(
            AxisInfo(
                label=label,
                canonical=canonical,
                unit=unit,
                scale=scale_value,
                index=idx,
            )
        )
    return infos


def bounding_box(
    crops: List[CropInfo], axes_order: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for crop in crops:
        infos = extract_axis_info(crop)
        label_to_info = {info.canonical: info for info in infos}
        min_corner: List[float] = []
        max_corner: List[float] = []
        for label in axes_order:
            info = label_to_info.get(label)
            if info is None or info.index >= len(crop.translation):
                min_corner.append(0.0)
                max_corner.append(0.0)
                continue
            scale_value = info.scale
            translation_value = crop.translation[info.index]
            if label in {"x", "y", "z"}:
                scale_value *= 1e-6
                translation_value *= 1e-6
            size = crop.shape[info.index] * scale_value
            min_corner.append(translation_value)
            max_corner.append(translation_value + size)
        mins.append(min_corner)
        maxs.append(max_corner)
    return np.min(np.asarray(mins), axis=0), np.max(np.asarray(maxs), axis=0)


def encode_url(state: Dict, viewer_prefix: str) -> str:
    state_json = json.dumps(state, separators=(",", ":"))
    from urllib.parse import quote

    encoded = quote(state_json, safe="[]{}:,\"")
    prefix = viewer_prefix.rstrip("#!")
    return f"{prefix}#!{encoded}"


def write_output(path: Path, url: str, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(url + "\n", encoding="utf-8")
    state_path = path.with_suffix(".json")
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    ome_zarrs = find_ome_zarrs(root)
    if not ome_zarrs:
        raise SystemExit(f"No *.ome.zarr directories found under {root}")

    crops = [load_crop(path) for path in ome_zarrs]
    shader = load_shader(args.shader_file)
    state = build_state(
        crops=crops,
        root=root,
        path_prefix=args.path_prefix,
        template=args.url_template,
        layer_name=args.layer_name,
        shader=shader,
    )
    url = encode_url(state, args.viewer)
    print(url)
    if args.output:
        write_output(args.output, url, state)


if __name__ == "__main__":
    main()

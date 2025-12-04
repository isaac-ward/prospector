# src/prospector/caves/utils/build_cavemap.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

from omegaconf import DictConfig

from prospector.utils.cacher import Cacher
from prospector.caves.cave_map_2d import CaveMap2D
from prospector.caves.cave_map_3d import CaveMap3D


CaveMapType = Union[CaveMap2D, CaveMap3D]


def build_cavemap(
    cave_name: str,
    cave_cfg: DictConfig,
    *,
    repo_root: Path,
    use_3d: bool,
) -> Tuple[CaveMapType, Path]:
    """
    Build (or load from cache) a CaveMap2D or CaveMap3D for the given cave.

    Parameters
    ----------
    cave_name : str
        Name/key of the cave in the config.
    cave_cfg : DictConfig
        Configuration node under args.caves[cave_name].
    repo_root : Path
        Root of the repo (used to resolve relative PLY path).
    use_3d : bool
        If True, build a CaveMap3D; otherwise build a CaveMap2D.

    Returns
    -------
    cave_map : CaveMap2D or CaveMap3D
    ply_path : Path
        Resolved path to the underlying PLY file.
    """
    rel_path = str(cave_cfg.relative_path)
    ply_path = repo_root / rel_path.lstrip("/")

    voxel_size = float(cave_cfg.voxel_size)
    known_internal_xyz = list(cave_cfg.known_internal_xyz)

    if use_3d:
        print(
            f"[build_cavemap] Building 3D cave map (CaveMap3D) for '{cave_name}' "
            f"from {ply_path} with voxel_size={voxel_size}"
        )

        cacher_3d = Cacher(
            computation_inputs=(
                "CaveMap3D_v1",
                cave_name,
                str(ply_path),
                voxel_size,
                tuple(known_internal_xyz),
            ),
            tag="cavemap3d",
        )

        if cacher_3d.exists():
            cache_3d = cacher_3d.load()
            cave_map: CaveMap3D = cache_3d["cave_map_3d"]
        else:
            cave_map = CaveMap3D.from_ply(
                ply_path=ply_path,
                voxel_size=voxel_size,
            )
            cave_map.label_accessible_space(known_internal_xyz)
            cacher_3d.save({"cave_map_3d": cave_map})

        return cave_map, ply_path

    # 2D case
    slice_z = float(cave_cfg.slice_for_2d.z)
    slice_thickness = float(cave_cfg.slice_for_2d.thickness)

    print(
        f"[build_cavemap] Building 2D cave map (CaveMap2D) for '{cave_name}' "
        f"from {ply_path} with voxel_size={voxel_size}, "
        f"slice_z={slice_z}, thickness={slice_thickness}"
    )

    cacher_2d = Cacher(
        computation_inputs=(
            "CaveMap2D_v1",
            cave_name,
            str(ply_path),
            voxel_size,
            slice_z,
            slice_thickness,
            tuple(known_internal_xyz),
        ),
        tag="cavemap2d",
    )

    if cacher_2d.exists():
        cache_2d = cacher_2d.load()
        cave_map_2d: CaveMap2D = cache_2d["cave_map_2d"]
    else:
        cave_map_2d = CaveMap2D.from_ply(
            ply_path=ply_path,
            voxel_size=voxel_size,
            slice_for_2d_z=slice_z,
            slice_for_2d_thickness=slice_thickness,
        )
        cave_map_2d.label_accessible_space(known_internal_xyz[:2])  # only need XY for 2D
        cacher_2d.save({"cave_map_2d": cave_map_2d})

    return cave_map_2d, ply_path

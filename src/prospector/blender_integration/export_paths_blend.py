# src/prospector/blender_integration/export_paths_blend.py
"""
Write a small .blend containing only the completed flight paths of one run
(one bevelled curve per agent, fully drawn, agent colours), so they can be
appended into any scene (File > Append) without loading a large cave file.

Run with Blender (no scene file needed):

    blender -b --factory-startup --python src/prospector/blender_integration/export_paths_blend.py -- \
        <trial_folder> [<out.blend>]

<trial_folder> must contain trajectories/agent_*.json. Objects are named
traj_agent_<i> (agent 0 = blue, agent 1 = orange) in collection
'prospector_paths'.
"""

import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prospector.blender_integration.paths import find_agent_jsons  # noqa: E402
from prospector.blender_integration.trajectories import load_agent_trajectory  # noqa: E402
from prospector.blender_integration.blender_scene import create_bezier_curve_from_points  # noqa: E402
from prospector.blender_integration.materials import get_or_create_curve_material  # noqa: E402
from prospector.blender_integration.colors import get_agent_color_list  # noqa: E402


def main(trial_folder: Path, out_path: Path) -> None:
    # Empty the factory scene (cube, camera, light)
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)

    coll = bpy.data.collections.new("prospector_paths")
    bpy.context.scene.collection.children.link(coll)

    trajs = [load_agent_trajectory(p) for p in find_agent_jsons(trial_folder)]
    colors = get_agent_color_list(len(trajs))
    for t, rgb in zip(trajs, colors):
        mat = get_or_create_curve_material(name=f"traj_mat_agent_{t.agent_id}", rgb=rgb, alpha=0.7)
        create_bezier_curve_from_points(
            name=f"traj_agent_{t.agent_id}", points=t.positions, collection=coll, material=mat
        )
        print(f"[export_paths_blend] traj_agent_{t.agent_id}: {len(t.positions)} points, rgb={rgb}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path), compress=True)
    print(f"[export_paths_blend] wrote {out_path}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    trial = Path(argv[0]).resolve()
    out = Path(argv[1]).resolve() if len(argv) > 1 else trial / "paths.blend"
    main(trial, out)

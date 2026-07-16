"""
Paste this into Blender's Python editor and run it to visualize a cave graph JSON.

It creates/refreshes:
    Scene Collection / <cave collection> / <cave collection>-graph

Change CAVE_COLLECTION_NAME and NODE_COUNT below to inspect a different graph.
"""

import json
import sys
from pathlib import Path

import bpy


CAVE_COLLECTION_NAME = "tunnels"
NODE_COUNT = 30
GRAPH_JSON_REL = f"src/assets/graphs/n={NODE_COUNT}/{CAVE_COLLECTION_NAME}_3d.json"

NODE_RADIUS = 0.35
EDGE_RADIUS = 0.045
NODE_RGB = (0.05, 0.55, 1.0)
EDGE_RGB = (1.0, 0.68, 0.08)


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in (start,) + tuple(start.parents):
        if (p / "src" / "prospector").exists():
            return p
        if (p / "pyproject.toml").exists():
            return p
        if (p / ".git").exists():
            return p
    raise RuntimeError(f"Could not find Prospector repo root walking up from: {start}")


def add_src_to_syspath(repo_root: Path) -> None:
    src = repo_root / "src"
    if not src.exists():
        raise RuntimeError(f"Expected src/ at: {src}")
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def get_repo_root() -> Path:
    if bpy.data.filepath:
        return find_repo_root(Path(bpy.data.filepath).resolve().parent)
    return find_repo_root(Path.cwd())


def get_or_create_child_collection(parent: bpy.types.Collection, name: str) -> bpy.types.Collection:
    coll = parent.children.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        parent.children.link(coll)
    return coll


def clear_collection(coll: bpy.types.Collection) -> None:
    for child in list(coll.children):
        clear_collection(child)
        coll.children.unlink(child)
        if child.users == 0:
            bpy.data.collections.remove(child)

    for obj in list(coll.objects):
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)


def remove_collection_tree(coll: bpy.types.Collection) -> None:
    clear_collection(coll)
    for parent in list(bpy.data.collections):
        if coll.name in parent.children:
            parent.children.unlink(coll)
    if coll.name in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.unlink(coll)
    if coll.users == 0:
        bpy.data.collections.remove(coll)


def remove_existing_graph_collections(
    parent: bpy.types.Collection,
    old_name: str,
    new_name: str,
) -> None:
    matches = [
        child for child in parent.children
        if (
            child.name == old_name
            or child.name.startswith(old_name + ".")
            or child.name == new_name
            or child.name.startswith(new_name + ".")
        )
    ]
    for child in matches:
        remove_collection_tree(child)


def make_node_sphere(
    *,
    name: str,
    location,
    radius: float,
    collection: bpy.types.Collection,
    material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=12,
        radius=radius,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + "_mesh"
    obj.data.materials.append(material)
    obj.visible_shadow = False

    for owner in list(obj.users_collection):
        if owner != collection:
            owner.objects.unlink(obj)
    if collection.name not in [c.name for c in obj.users_collection]:
        collection.objects.link(obj)

    return obj


repo_root = get_repo_root()
add_src_to_syspath(repo_root)

from prospector.blender_integration.blender_scene import create_bezier_curve_from_points
from prospector.blender_integration.materials import get_or_create_curve_material


graph_path = repo_root / GRAPH_JSON_REL
with graph_path.open("r", encoding="utf-8") as f:
    graph = json.load(f)

nodes = graph["nodes"]
edges = graph["edges"]

cave_collection_name = CAVE_COLLECTION_NAME
graph_collection_name = f"{CAVE_COLLECTION_NAME}-graph"
cave_coll = bpy.data.collections.get(CAVE_COLLECTION_NAME)
if cave_coll is None:
    cave_coll = get_or_create_child_collection(bpy.context.scene.collection, CAVE_COLLECTION_NAME)

remove_existing_graph_collections(cave_coll, "graph", graph_collection_name)
graph_coll = get_or_create_child_collection(cave_coll, graph_collection_name)

node_mat = get_or_create_curve_material(
    name="graph_node_material",
    rgb=NODE_RGB,
    alpha=1.0,
    delete_existing=True,
)
edge_mat = get_or_create_curve_material(
    name="graph_edge_material",
    rgb=EDGE_RGB,
    alpha=1.0,
    delete_existing=True,
)

for i, node in enumerate(nodes):
    make_node_sphere(
        name=f"graph_node_{i:02d}",
        location=(float(node[0]), float(node[1]), float(node[2])),
        radius=NODE_RADIUS,
        collection=graph_coll,
        material=node_mat,
    )

for i, (a, b) in enumerate(edges):
    pa = nodes[int(a)]
    pb = nodes[int(b)]
    create_bezier_curve_from_points(
        name=f"graph_edge_{i:02d}_{int(a):02d}_{int(b):02d}",
        points=[
            (float(pa[0]), float(pa[1]), float(pa[2])),
            (float(pb[0]), float(pb[1]), float(pb[2])),
        ],
        collection=graph_coll,
        material=edge_mat,
        bevel_depth=EDGE_RADIUS,
        resolution_u=1,
    )

print(f"[Prospector] Loaded {graph_path}")
print(
    f"[Prospector] Created {len(nodes)} graph nodes and {len(edges)} graph edges "
    f"in {cave_collection_name}/{graph_collection_name}"
)

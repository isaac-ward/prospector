import bpy
from typing import Tuple

def get_or_create_curve_material(
    *,
    name: str,
    rgb: Tuple[float, float, float],
    alpha: float = 1.0,
    delete_existing=True,
) -> bpy.types.Material:
    """
    Create or reuse a solid-color material for trajectory curves.
    """
    mat = bpy.data.materials.get(name)
    # Delete existing material to reset
    if delete_existing and mat is not None:
        bpy.data.materials.remove(mat)
        mat = None

    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        nodes.clear()

        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        # bpy.data.materials["traj_mat_agent_0"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (125.553, 115.168, 147.003, 1)
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        bsdf.inputs["Alpha"].default_value = alpha
        bsdf.inputs["Roughness"].default_value = 1.0

        # And make the emission color the same and the strength 1
        bsdf.inputs[27].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        bsdf.inputs[28].default_value = 0.333

        out = nodes.new(type="ShaderNodeOutputMaterial")
        out.location = (300, 0)

        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        # Enable transparency if alpha < 1
        if alpha < 1.0:
            mat.blend_method = "BLEND"
            #mat.shadow_method = "NONE"

    return mat

import os
import time

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy import ndarray

import bpy


@dataclass
class SceneInfo:
    name: str
    scene_file_path: str
    reference_camera_pos: List[ndarray[np.float32]]
    # camera_xy: List[Tuple[float]]
    # height_range: Tuple[float]


# location xyz:  <Vector (-5.9483, -1.4727, 1.8115)>
# rotation wxyz:  <Quaternion (w=0.6835, x=-0.6048, y=-0.2766, z=-0.3009)>
# rotation euler:  <Euler (x=-1.3979, y=-0.8362, z=-0.1147), order='XYZ'>


BARBERSHOP_INFO = SceneInfo(
    name="barbershop",
    scene_file_path="./scenes/barbershop_interior.blend",
    reference_camera_pos=[
        np.array([2.8629, 6.0945, 1.5725, 1.1384, 0.0130, -5.0755]),
        np.array([2.8629, 6.0945, 1.5725, 1.1384, 0.0130, -5.0755]),
    ],
)


SCENE_INFO = {"barbershop": BARBERSHOP_INFO}


@dataclass
class Obj3D:
    x: float
    y: float
    z: float


def setup_renderer(use_gpu: bool = False):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.use_adaptive_sampling = False
    bpy.context.scene.render.resolution_x = 64
    bpy.context.scene.render.resolution_y = 64

    if use_gpu:
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"

        # Enable all available GPUs
        for device in bpy.context.preferences.addons[
            "cycles"
        ].preferences.devices:
            if device.type == "CUDA":
                device.use = True


# Using compositing to get the depth, albedo and normals.
def activate_compositing(path: str):
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render

    scene.use_nodes = True
    scene.view_layers["RenderLayer"].use_pass_normal = True
    scene.view_layers["RenderLayer"].use_pass_diffuse_color = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    # Clear default nodes.
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node.
    render_layers = nodes.new("CompositorNodeRLayers")

    # Create depth output nodes.
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = "Depth Output"
    depth_file_output.base_path = f"{path}/depth"
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = "PNG"

    # depth_file_output.format.color_depth = args
    depth_file_output.format.color_mode = "BW"

    # Remap as other types can not represent the full range of depth.
    map = nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.offset = [-0.7]
    map.size = [1.4]
    map.use_min = True
    map.min = [0]

    links.new(render_layers.outputs["Depth"], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])


def deactivate_compositing():
    bpy.context.scene.use_nodes = False

    # print(bpy.context.scene.view_layers.keys())

    bpy.context.scene.view_layers["RenderLayer"].use_pass_z = False
    bpy.context.scene.view_layers["RenderLayer"].use_pass_combined = False
    bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = False
    bpy.context.scene.view_layers["RenderLayer"].use_pass_diffuse_color = False
    bpy.context.scene.view_layers["RenderLayer"].use_pass_glossy_color = False

    # bpy.context.view_layer.use_pass_z = False
    # bpy.context.view_layer.use_pass_combined = False
    # bpy.context.view_layer.use_pass_normal = False
    # bpy.context.view_layer.use_pass_diffuse_color = False
    # bpy.context.view_layer.use_pass_glossy_color = False


# Warning: Not thread safe.
def render(
    camera: bpy.types.Object,
    position: Obj3D,
    euler_rotation: Obj3D,
    output_folder: str,
    samples: int = 1,
    composite_output_nodes: dict[
        str, bpy.types.CompositorNodeOutputFile
    ] = None,
):
    x, y, z = position.x, position.y, position.z
    eu_x, eu_y, eu_z = euler_rotation.x, euler_rotation.y, euler_rotation.z

    camera.location[0] = x
    camera.location[1] = y
    camera.location[2] = z

    camera.rotation_euler[0] = eu_x
    camera.rotation_euler[1] = eu_y
    camera.rotation_euler[2] = eu_z

    bpy.context.scene.cycles.samples = samples

    file_name = f"image_x_{x}_y_{y}_z_{z}_eux_{eu_x}_euy_{eu_y}_euz_{eu_z}"

    bpy.context.scene.render.filepath = (
        f"{output_folder}/samples_{samples}/{file_name}.png"
    )

    bpy.ops.render.render(write_still=True)


def main():
    scene_name = "barbershop"
    num_data_points = 1600
    spps = [1, 2, 4, 8, 1024]

    output_folder = f"output/{scene_name}"
    os.makedirs(output_folder, exist_ok=True)

    scene_info = SCENE_INFO[scene_name]

    # Load the scene.
    bpy.ops.wm.open_mainfile(filepath=scene_info.scene_file_path)
    reference_positions = scene_info.reference_camera_pos
    camera = bpy.data.objects.get("Camera")

    setup_renderer(use_gpu=False)

    # Main loop.
    start_time = time.time()

    for data_idx in range(num_data_points):
        # Sample two reference positions and interpolate between them.
        idxs = np.random.choice(len(reference_positions), replace=False, size=2)
        reference_pos_1 = reference_positions[idxs[0]]
        reference_pos_2 = reference_positions[idxs[1]]

        interpolation_coeff = np.random.uniform()
        camera_pos = (
            interpolation_coeff * reference_pos_1
            + (1.0 - interpolation_coeff) * reference_pos_2
        )

        # Loop over all samples per pixel to render at.
        for spp in spps:
            # Get auxiliary features only for low spp images.
            if spp != 1024:
                activate_compositing(f"output/{scene_name}/samples_{spp}")
            else:
                deactivate_compositing()

            render(
                camera,
                Obj3D(*camera_pos[:3]),
                Obj3D(*camera_pos[3:]),
                output_folder,
                spp,
                None,
            )

        # Print progess to console.
        frac_done = (data_idx + 1) / num_data_points
        print(f"Rendered {frac_done:.3f}, in {time.time() - start_time} sec.\n")


if __name__ == "__main__":
    main()

from dataclasses import dataclass
import itertools

import numpy as np
import bpy
import sys
import time
import argparse
import os


@dataclass
class Obj3D:
    x: float
    y: float
    z: float


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script to render images for the Amazon Lumberyard Bistro script"
    )

    parser.add_argument(
        "-f",
        "--fbx_scene_path",
        default="./scenes/bistro/exterior.obj",
        type=str,
        help="Path to the BistroExterior.fbx file",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        default=".",
        type=str,
        help="Folder where the rendering output is saved",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="The first index (inclusive) of images to be rendered "
        "(as per the script's own rendering order)",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=None,
        help="The last index (inclusive) of images to be rendered "
        "(as per the script's own rendering order)",
    )
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="Use GPU (Nvidia/CUDA only)"
    )

    # Parse arguments after --
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


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
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_combined = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_glossy_color = True

    bpy.context.scene.node_tree.nodes.clear()

    nodes = {}

    composite = bpy.context.scene.node_tree.nodes.new("CompositorNodeComposite")
    render_layers = bpy.context.scene.node_tree.nodes.new(
        "CompositorNodeRLayers"
    )

    normalize = bpy.context.scene.node_tree.nodes.new(
        type="CompositorNodeNormalize"
    )
    nodes["depth"] = bpy.context.scene.node_tree.nodes.new(
        "CompositorNodeOutputFile"
    )
    nodes["depth"].base_path = path + "/depth/"

    nodes["glossy_color"] = bpy.context.scene.node_tree.nodes.new(
        "CompositorNodeOutputFile"
    )
    nodes["glossy_color"].base_path = path + "/glossy_color/"
    nodes["glossy_color"].format.color_mode = "RGB"

    nodes["diffuse_color"] = bpy.context.scene.node_tree.nodes.new(
        "CompositorNodeOutputFile"
    )
    nodes["diffuse_color"].base_path = path + "/diffuse_color/"
    nodes["diffuse_color"].format.color_mode = "RGB"

    nodes["normal"] = bpy.context.scene.node_tree.nodes.new(
        "CompositorNodeOutputFile"
    )
    nodes["normal"].base_path = path + "/normal/"
    nodes["normal"].format.color_mode = "RGB"

    # Link Render Layers node to other nodes
    bpy.context.scene.node_tree.links.new(
        render_layers.outputs["Image"], composite.inputs["Image"]
    )
    bpy.context.scene.node_tree.links.new(
        render_layers.outputs["Depth"], normalize.inputs["Value"]
    )
    bpy.context.scene.node_tree.links.new(
        normalize.outputs["Value"], nodes["depth"].inputs["Image"]
    )
    bpy.context.scene.node_tree.links.new(
        render_layers.outputs["GlossCol"], nodes["glossy_color"].inputs["Image"]
    )
    bpy.context.scene.node_tree.links.new(
        render_layers.outputs["DiffCol"], nodes["diffuse_color"].inputs["Image"]
    )
    bpy.context.scene.node_tree.links.new(
        render_layers.outputs["Normal"], nodes["normal"].inputs["Image"]
    )

    return nodes


def deactivate_compositing():
    bpy.context.scene.use_nodes = False
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = False
    bpy.context.scene.view_layers["ViewLayer"].use_pass_combined = False
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = False
    bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_color = False
    bpy.context.scene.view_layers["ViewLayer"].use_pass_glossy_color = False


# Warning: Not threadsafe
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

    camera.location[0] = position.x
    camera.location[1] = position.y
    camera.location[2] = position.z

    camera.rotation_euler[0] = euler_rotation.x * pi_by_180
    camera.rotation_euler[1] = euler_rotation.y * pi_by_180
    camera.rotation_euler[2] = euler_rotation.z * pi_by_180

    bpy.context.scene.cycles.samples = samples

    file_name = f"image_x_{x}_y_{y}_z_{z}_eux_{eu_x}_euz_{eu_z}"

    bpy.context.scene.render.filepath = (
        f"{output_folder}/samples_{samples}/{file_name}.png"
    )

    bpy.ops.render.render(write_still=True)

    if composite_output_nodes is not None:
        for key in composite_output_nodes.keys():
            os.rename(
                f"{output_folder}/{key}/Image0001.png",
                f"{output_folder}/{key}/{file_name}.png",
            )

        if not os.path.exists(f"{output_folder}/view_space_matrix"):
            os.makedirs(f"{output_folder}/view_space_matrix")

        np.save(
            f"{output_folder}/view_space_matrix/{file_name}.npy",
            np.array(camera.matrix_world.inverted().to_3x3()),
        )


if __name__ == "__main__":
    pi_by_180 = 3.14159 / 180

    # Handpicked (x, y) coordinates which are good camera positions for the Bistro exterior scene
    points = [
        (20, 65),
        (23, 60),
        (20, 50),
        (13, 40),
        (7, 30),
        (-2, 20),
        (-10, 10),
        (-12, 4),
        (-12, -2),
        (-10, -6),
        (-5, -10),
    ]
    points += [
        (5, -14),
        (9, -16),
        (13, -18),
        (35, -26),
        (40, -28),
        (45, -30),
        (50, -32),
        (53, -33),
        (56, -34),
        (60, -36),
    ]
    points += [
        (70, -25),
        (77, -25),
        (85, -30),
        (85, -35),
        (84, -42),
        (80, -52),
        (76, -57),
        (70, -59),
        (65, -55),
        (60, -50),
    ]
    points += [(-25, 13), (-35, 23), (-39, 30), (-31, 37)]

    sample_count = [1, 512]
    heights = [i for i in range(2, 17)]
    camera_euler_rotations_x = [60, 90, 120]
    camera_euler_rotations_z = [i for i in range(0, 360, 45)]

    total_images_to_render = (
        len(points)
        * len(sample_count)
        * len(heights)
        * len(camera_euler_rotations_x)
        * len(camera_euler_rotations_z)
    )
    args = parse_arguments()

    # bpy.ops.import_scene.fbx(filepath = args.fbx_scene_path)
    bpy.ops.wm.obj_import(filepath=args.fbx_scene_path)

    setup_renderer(args.gpu)

    cnt = 0
    start = args.start
    end = args.end if args.end is not None else total_images_to_render

    print(
        f"Rendering {end-start+1} images, with indices {start}..{end}\n",
        file=sys.stderr,
    )

    axes = [points, heights, camera_euler_rotations_x, camera_euler_rotations_z]
    camera = bpy.data.objects.get("Camera")

    for samples in sample_count:
        if samples == 1:
            nodes = activate_compositing(args.output_folder)
        else:
            nodes = deactivate_compositing()

        for (x, y), z, eu_x, eu_z in itertools.product(*axes):
            cnt += 1
            if cnt < start or cnt > end:
                print(
                    f"Skipping image {cnt}/{total_images_to_render}",
                    file=sys.stderr,
                )
                continue

            print(
                f"Rendering image {cnt}/{total_images_to_render}, with parameters "
                f"x_{x}_y_{y}_z_{z}_eux_{eu_x}_euz_{eu_z}, samples={samples}",
                file=sys.stderr,
            )

            t = time.time()

            render(
                camera,
                Obj3D(x, y, z),
                Obj3D(eu_x, 0, eu_z),
                args.output_folder,
                samples,
                nodes,
            )

            print(
                f"Rendered {cnt}/{total_images_to_render}, in {time.time() - t} sec.\n",
                file=sys.stderr,
            )

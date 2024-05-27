import os
import time

import random
import numpy as np

from typing import List
from dataclasses import dataclass

import bpy


@dataclass
class SceneInfo:
    name: str
    scene_file_path: str
    view_layer_name: str
    camera_name: str
    aux_feature_image_name: str
    reference_camera_pos: List[np.ndarray[np.float32]]


CLASSROOM_INFO = SceneInfo(
    name="classroom",
    scene_file_path="./scenes/classroom/classroom.blend",
    view_layer_name="interior",
    camera_name="Camera",
    aux_feature_image_name="Image0001.png",
    reference_camera_pos=[
        np.array(
            [
                2.576395273208618,
                -4.465750694274902,
                1.094475507736206,
                1.5819953680038452,
                3.25860833072511e-06,
                0.25481462478637695,
            ]
        ),
        np.array(
            [
                2.8309147357940674,
                -3.001325845718384,
                1.7191882133483887,
                1.4511008262634277,
                9.57443717197748e-07,
                1.6475881338119507,
            ]
        ),
        np.array(
            [
                0.8357277512550354,
                -4.250850677490234,
                2.0564661026000977,
                1.5977354049682617,
                1.5161617739067879e-05,
                -0.7609918713569641,
            ]
        ),
        np.array(
            [
                -0.4849654734134674,
                -2.8069472312927246,
                0.8680052161216736,
                1.2443486452102661,
                6.792260592192179e-06,
                2.493175745010376,
            ]
        ),
        np.array(
            [
                -0.09100060164928436,
                1.1138038635253906,
                1.5776495933532715,
                1.498311161994934,
                -5.939983111602487e-06,
                1.6266303062438965,
            ]
        ),
        np.array(
            [
                -0.3006440997123718,
                0.6313178539276123,
                1.6596485376358032,
                1.273171067237854,
                -7.224236014735652e-06,
                0.3254891037940979,
            ]
        ),
        np.array(
            [
                0.8298668265342712,
                0.5137124061584473,
                1.731061339378357,
                1.5821146965026855,
                -6.1487144193961285e-06,
                -0.006999018602073193,
            ]
        ),
        np.array(
            [
                1.1274628639221191,
                1.3732091188430786,
                1.8063400983810425,
                1.3229436874389648,
                -2.946389713542885e-06,
                -0.7662237286567688,
            ]
        ),
        np.array(
            [
                2.940042018890381,
                2.120189666748047,
                1.909437894821167,
                0.6736802458763123,
                1.2430627066351008e-05,
                -3.913100004196167,
            ]
        ),
        np.array(
            [
                -1.1751595735549927,
                -4.300800323486328,
                2.0824930667877197,
                1.2182408571243286,
                2.0577628674800508e-05,
                -0.551630437374115,
            ]
        ),
        np.array(
            [
                0.6757554411888123,
                -1.2991529703140259,
                1.4262173175811768,
                1.3255845308303833,
                1.758587131917011e-05,
                -3.766547679901123,
            ]
        ),
    ],
)

SCENE_INFO = {
    "classroom": CLASSROOM_INFO,
}


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
def activate_compositing(path: str, view_layer_name):
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers[view_layer_name].use_pass_z = True
    bpy.context.scene.view_layers[view_layer_name].use_pass_combined = True
    bpy.context.scene.view_layers[view_layer_name].use_pass_normal = True
    bpy.context.scene.view_layers[view_layer_name].use_pass_diffuse_color = True
    bpy.context.scene.view_layers[view_layer_name].use_pass_glossy_color = True

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


def deactivate_compositing(view_layer_name):
    bpy.context.scene.use_nodes = False

    bpy.context.scene.view_layers[view_layer_name].use_pass_z = False
    bpy.context.scene.view_layers[view_layer_name].use_pass_combined = False
    bpy.context.scene.view_layers[view_layer_name].use_pass_normal = False
    bpy.context.scene.view_layers[view_layer_name].use_pass_diffuse_color = (
        False
    )
    bpy.context.scene.view_layers[view_layer_name].use_pass_glossy_color = False

    return None


# Warning: Not thread safe.
def render(
    camera: bpy.types.Object,
    position: Obj3D,
    euler_rotation: Obj3D,
    output_folder: str,
    output_file_name: str,
    aux_feature_image_name: str,
    samples: int = 1,
    composite_output_nodes: dict[
        str, bpy.types.CompositorNodeOutputFile
    ] = None,
):
    x, y, z = position.x, position.y, position.z
    eu_x, eu_y, eu_z = euler_rotation.x, euler_rotation.y, euler_rotation.z

    print(f"Old camera location: {camera.location}")

    camera.location[0] = x
    camera.location[1] = y
    camera.location[2] = z

    camera.rotation_euler[0] = eu_x
    camera.rotation_euler[1] = eu_y
    camera.rotation_euler[2] = eu_z

    print(f"New camera location: {camera.location}")

    bpy.context.scene.cycles.samples = samples

    bpy.context.scene.render.filepath = (
        f"{output_folder}/samples_{samples}/{output_file_name}.png"
    )

    bpy.ops.render.render(write_still=True)

    if composite_output_nodes is not None:
        for key in composite_output_nodes.keys():
            os.rename(
                f"{output_folder}/samples_{samples}/{key}/{aux_feature_image_name}",
                f"{output_folder}/samples_{samples}/{key}/{output_file_name}.png",
            )

        if not os.path.exists(
            f"{output_folder}/samples_{samples}/view_space_matrix"
        ):
            os.makedirs(f"{output_folder}/samples_{samples}/view_space_matrix")

        np.save(
            f"{output_folder}/samples_{samples}/view_space_matrix/{output_file_name}.npy",
            np.array(camera.matrix_world.inverted().to_3x3()),
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    # scene_name = "barbershop"
    # # num_data_points = 1600
    # num_data_points = 1568
    # seed = 1
    # spps = [1, 2, 4, 8, 1024]

    scene_name = "classroom"
    num_data_points = 1600
    seed = 0
    spps = [1, 2, 4, 8, 1024]

    output_folder = f"output/{scene_name}"
    os.makedirs(output_folder, exist_ok=True)

    scene_info = SCENE_INFO[scene_name]

    # Set seed.
    set_seed(seed)

    # Load the scene.
    bpy.ops.wm.open_mainfile(filepath=scene_info.scene_file_path)
    reference_positions = scene_info.reference_camera_pos
    camera = bpy.data.objects.get(scene_info.camera_name)

    print(camera)

    bpy.context.scene.camera = camera

    print(bpy.context.scene.objects.keys())
    print(bpy.context.scene)
    print(bpy.context.scene.camera)

    for ob in bpy.context.scene.objects:
        if ob.type == "CAMERA":
            print("FOUND CAMERA")
            print(ob)

    setup_renderer(use_gpu=False)

    # Main loop.
    start_time = time.time()

    for data_idx in range(num_data_points):
        # Sample two reference positions and interpolate between them.
        idxs = np.random.choice(len(reference_positions), replace=False, size=2)

        print(idxs)

        reference_pos_1 = reference_positions[idxs[0]]
        reference_pos_2 = reference_positions[idxs[1]]

        print(reference_pos_1)
        print(reference_pos_2)

        interpolation_coeff = np.random.uniform()
        camera_pos = (
            interpolation_coeff * reference_pos_1
            + (1.0 - interpolation_coeff) * reference_pos_2
        )

        # print(f"Interpolated camera pos: {camera_pos}")

        # Loop over all samples per pixel to render at.
        for spp in spps:
            # Get auxiliary features only for low spp images.
            if spp != 1024:
                nodes = activate_compositing(
                    view_layer_name=scene_info.view_layer_name,
                    path=f"output/{scene_name}/samples_{spp}",
                )
            else:
                nodes = deactivate_compositing(
                    view_layer_name=scene_info.view_layer_name
                )

            render(
                camera=camera,
                position=Obj3D(*camera_pos[:3]),
                euler_rotation=Obj3D(*camera_pos[3:]),
                output_folder=output_folder,
                output_file_name=f"image_{data_idx}",
                samples=spp,
                composite_output_nodes=nodes,
                aux_feature_image_name=scene_info.aux_feature_image_name,
            )

        # Print progess to console.
        frac_done = (data_idx + 1) / num_data_points
        print(f"Rendered {frac_done:.3f}, in {time.time() - start_time} sec.\n")


if __name__ == "__main__":
    main()

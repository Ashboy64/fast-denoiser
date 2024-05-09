import os
import numpy as np

import subprocess


PBRT_PATH = "./pbrt"
REFERENCES_PATH = "./references"
SCENE_PATH = "./scene-files/watercolor"


SCENE_TEMPLATE = """
Sampler "halton"
Integrator "volpath" "integer maxdepth" 15

Film "{film}"
    "integer yresolution" [ 64 ]    "integer xresolution" [ 64 ]
    "string filename" [ "{filename}" ]
Scale -1 1 1
LookAt {eye}
    {lookat}
    {up}
Camera "perspective"
    "float fov" [ {fov} ]

WorldBegin

Include "lights-no-windowglass.pbrt"
Include "materials.pbrt"
Include "geometry.pbrt"
"""

COMMAND_TEMPLATE = (
    """{pbrt_path} {scene_file_path} --outfile {render_out_file} --spp {spp}"""
)


def str_position_to_array(str_position):
    return np.array([float(x) for x in str_position.split(" ")])


reference_str_positions = [
    "221.141205 122.646004 2.43404675 220.141205 122.646004 2.43404675 0 1 0 43.6028175",
    "-2.70786691 85.4769516 240.523529 -3.30121756 85.4485712 239.718582 -0.00141029898 0.999997199 -0.00191321515 22.6198654",
    "247.908615 63.4503365 125.32412 246.917603 63.4553365 125.1903 0 1 0 20.4079475",
    "246.201401 177.455338 38.538826 245.696762 176.740402 38.0548897 -0.516015887 0.699185967 -0.494840115 22.6198654",
    "231.791519 163.256424 77.3447189 231.243347 162.608231 76.8161774 -0.466618747 0.76148057 -0.44990477 22.6198654",
]

reference_positions = [
    str_position_to_array(x) for x in reference_str_positions
]


def create_scene_file(
    scene_file_path, render_out_file, camera_pos, film="gbuffer"
):
    eye_str = " ".join([str(p) for p in camera_pos[:3]])
    lookat_str = " ".join([str(p) for p in camera_pos[3:6]])
    up_str = " ".join([str(p) for p in camera_pos[6:9]])
    fov_str = str(camera_pos[9])

    scene_str = SCENE_TEMPLATE.format(
        film=film,
        eye=eye_str,
        lookat=lookat_str,
        up=up_str,
        fov=fov_str,
        filename=render_out_file,
    )

    with open(scene_file_path, "w") as f:
        f.write(scene_str)


def visualize_references():
    for reference_idx, reference_pos in enumerate(reference_positions):
        scene_file_path = (
            f"./scene-files/watercolor/reference_{reference_idx}.pbrt"
        )
        render_out_file = f"./references/reference_{reference_idx}.exr"

        create_scene_file(
            scene_file_path, render_out_file, reference_pos, film="gbuffer"
        )

        command_str = COMMAND_TEMPLATE.format(
            pbrt_path=PBRT_PATH,
            scene_file_path=scene_file_path,
            render_out_file=render_out_file,
            spp=1000,
        )

        subprocess.run(command_str.split(" "))


if __name__ == "__main__":
    visualize_references()

import os
import numpy as np

import OpenEXR
import Imath

import cv2 as cv

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def np_dtype_to_imath_pixel_type(dtype):
    if dtype == np.float16:
        return Imath.PixelType(Imath.PixelType.HALF)
    elif dtype == np.float32:
        return Imath.PixelType(Imath.PixelType.FLOAT)
    return None


def compute_exr_image_dims(exr_file):
    height = (
        exr_file.header()["displayWindow"].max.y
        + 1
        - exr_file.header()["displayWindow"].min.y
    )
    width = (
        exr_file.header()["displayWindow"].max.x
        + 1
        - exr_file.header()["displayWindow"].min.x
    )
    return height, width


def pack_into_numpy(
    exr_channels, exr_file, height=None, width=None, dtype=np.float32
):
    if height is None or width is None:
        height, width = compute_exr_image_dims(exr_file)
    imath_pixel_type = np_dtype_to_imath_pixel_type(dtype)

    output_arrays = []

    for channel in exr_channels:
        raw_bytes = exr_file.channel(channel, imath_pixel_type)
        np_array = np.frombuffer(raw_bytes, dtype=np.float32)
        np_array = np_array.reshape(height, width)
        output_arrays.append(np_array)

    return np.stack(output_arrays)


def get_gbuffer_feature_metadata():
    return {
        "rgb": 3,
        "albedo": 3,
        "position": 3,
        "surface_coords": 2,
        "grad_camera_depth": 2,
        "surface_normals": 3,
        "shading_normals": 3,
        "rgb_sample_variance": 3,
        "rgb_relative_sample_variance": 3,
    }


def read_gbufferfilm_exr(filepath, dtype=np.float32):
    """GBufferFilm fields are described in https://pbrt.org/users-guide-v4."""

    exr_file = OpenEXR.InputFile(filepath)
    # print(exr_file.header()["channels"])
    height, width = compute_exr_image_dims(exr_file)

    img_data = {}

    img_data["rgb"] = pack_into_numpy(
        ["R", "G", "B"], exr_file, height=height, width=width, dtype=dtype
    )

    img_data["albedo"] = pack_into_numpy(
        ["Albedo.R", "Albedo.G", "Albedo.B"],
        exr_file,
        height=height,
        width=width,
        dtype=dtype,
    )

    img_data["position"] = pack_into_numpy(
        ["P.X", "P.Y", "P.Z"],
        exr_file,
        height=height,
        width=width,
        dtype=dtype,
    )

    img_data["surface_coords"] = pack_into_numpy(
        ["u", "v"],
        exr_file,
        height=height,
        width=width,
        dtype=dtype,
    )

    img_data["grad_camera_depth"] = pack_into_numpy(
        ["dzdx", "dzdy"],
        exr_file,
        height=height,
        width=width,
        dtype=dtype,
    )

    img_data["surface_normals"] = pack_into_numpy(
        ["N.X", "N.Y", "N.Z"],
        exr_file,
        height=height,
        width=width,
        dtype=dtype,
    )

    img_data["shading_normals"] = pack_into_numpy(
        ["Ns.X", "Ns.Y", "Ns.Z"],
        exr_file,
        height=height,
        width=width,
        dtype=dtype,
    )

    img_data["rgb_sample_variance"] = pack_into_numpy(
        ["Variance.R", "Variance.G", "Variance.B"],
        exr_file,
        height=height,
        width=width,
        dtype=dtype,
    )

    img_data["rgb_relative_sample_variance"] = pack_into_numpy(
        ["RelativeVariance.R", "RelativeVariance.G", "RelativeVariance.B"],
        exr_file,
        height=height,
        width=width,
        dtype=dtype,
    )

    return img_data


def read_depth_exr_file(filepath):
    exrfile = OpenEXR.InputFile(filepath)

    print(exrfile.header()["channels"])

    raw_bytes = exrfile.channel("B", Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = (
        exrfile.header()["displayWindow"].max.y
        + 1
        - exrfile.header()["displayWindow"].min.y
    )
    width = (
        exrfile.header()["displayWindow"].max.x
        + 1
        - exrfile.header()["displayWindow"].min.x
    )
    depth_map = np.reshape(depth_vector, (height, width))
    return depth_map


def visualize_exr_images(filepaths):
    for filepath in filepaths:
        img = cv.imread(
            filepath,
            cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH,
        )

        cv.imshow("filepath", img)
        cv.waitKey(0)


def load_teapot():
    img_1k_spp = cv.imread(
        "test-images/watercolor/camera_1-100_spp.exr",
        cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH,
    )

    img_4_spp = cv.imread(
        "test-images/watercolor/camera_1-4_spp.exr",
        cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH,
    )

    img_1_spp = cv.imread(
        "test-images/watercolor/camera_1-1_spp.exr",
        cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH,
    )

    cv.imshow("camera_1-100_spp.exr", img_1k_spp)
    cv.imshow("camera_1-4_spp.exr", img_4_spp)
    cv.imshow("camera_1-1_spp.exr", img_1_spp)
    cv.waitKey(0)

    print("Teapot 1 features:")
    features = read_gbufferfilm_exr("test-images/watercolor/camera_1-1_spp.exr")
    for feature_name, feature in features.items():
        print(feature_name)
        print(feature)
        print()


if __name__ == "__main__":
    load_teapot()

// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/cpu/render.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/memory.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/options.h>
#include <pbrt/parser.h>
#include <pbrt/scene.h>
#include <pbrt/util/args.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/log.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/string.h>
#include <pbrt/wavefront/wavefront.h>

#include <string>
#include <vector>

using namespace pbrt;

static void usage(const std::string &msg = {}) {
    if (!msg.empty())
        fprintf(stderr, "pbrt: %s\n\n", msg.c_str());

    fprintf(stderr,
            R"(usage: pbrt [<options>] <filename.pbrt...>

Rendering options:
  --cropwindow <x0,x1,y0,y1>    Specify an image crop window w.r.t. [0,1]^2.
  --debugstart <values>         Inform the Integrator where to start rendering for
                                faster debugging. (<values> are Integrator-specific
                                and come from error message text.)
  --disable-image-textures      Always return the average value of image textures.
  --disable-pixel-jitter        Always sample pixels at their centers.
  --disable-texture-filtering   Point-sample all textures.
  --disable-wavelength-jitter   Always sample the same %d wavelengths of light.
  --displacement-edge-scale <s> Scale target triangle edge length by given value.
                                (Default: 1)
  --display-server <addr:port>  Connect to display server at given address and port
                                to display the image as it's being rendered.
  --force-diffuse               Convert all materials to be diffuse.)
  --fullscreen                  Render fullscreen. Only supported with --interactive.)"
#ifdef PBRT_BUILD_GPU_RENDERER
            R"(
  --gpu                         Use the GPU for rendering. (Default: disabled)
  --gpu-device <index>          Use specified GPU for rendering.)"
#endif
            R"(
  --help                        Print this help text.
  --interactive                 Enable interactive rendering mode.
  --mse-reference-image         Filename for reference image to use for MSE computation.
  --mse-reference-out           File to write MSE error vs spp results.
  --nthreads <num>              Use specified number of threads for rendering.
  --outfile <filename>          Write the final image to the given filename.
  --pixel <x,y>                 Render just the specified pixel.
  --pixelbounds <x0,x1,y0,y1>   Specify an image crop window w.r.t. pixel coordinates.
  --pixelmaterial <x,y>         Print information about the material visible in the
                                center of the pixel's extent.
  --pixelstats                  Record per-pixel statistics and write additional images
                                with their values.
  --quick                       Automatically reduce a number of quality settings
                                to render more quickly.
  --quiet                       Suppress all text output other than error messages.
  --render-coord-sys <name>     Coordinate system to use for the scene when rendering,
                                where name is "camera", "cameraworld", or "world".
  --seed <n>                    Set random number generator seed. Default: 0.
  --stats                       Print various statistics after rendering completes.
  --spp <n>                     Override number of pixel samples specified in scene
                                description file.
  --wavefront                   Use wavefront volumetric path integrator.
  --write-partial-images        Periodically write the current image to disk, rather
                                than waiting for the end of rendering. Default: disabled.

Logging options:
  --log-file <filename>         Filename to write logging messages to. Default: none;
                                messages are printed to standard error. Implies
                                --log-level verbose if specified.
  --log-level <level>           Log messages at or above this level, where <level>
                                is "verbose", "error", or "fatal". Default: "error".
  --log-utilization             Periodically print processor and memory use in verbose-
                                level logging.

Reformatting options:
  --format                      Print a reformatted version of the input file(s) to
                                standard output. Does not render an image.
  --toply                       Print a reformatted version of the input file(s) to
                                standard output and convert all triangle meshes to
                                PLY files. Does not render an image.
  --upgrade                     Upgrade a pbrt-v3 file to pbrt-v4's format.

Multiview options:
  --num-views-to-render                   Number of randomized camera positions to render.
)",
            NSpectrumSamples);
    exit(msg.empty() ? 0 : 1);
}

// camera_params=(
//   "221.141205 122.646004 2.43404675 220.141205 122.646004 2.43404675 0 1 0 43.6028175"
//   "-2.70786691 85.4769516 240.523529 -3.30121756 85.4485712 239.718582 -0.00141029898
//   0.999997199 -0.00191321515 22.6198654" "247.908615 63.4503365 125.32412
//   246.917603 63.4553365 125.1903 0 1 0 20.4079475" "246.201401 177.455338 38.538826
//   245.696762 176.740402 38.0548897 -0.516015887 0.699185967 -0.494840115 22.6198654"
//   "231.791519 163.256424 77.3447189 231.243347 162.608231 76.8161774 -0.466618747
//   0.76148057 -0.44990477 22.6198654"
// )

const std::vector<std::vector<Float>> REFERENCE_CAMERA_POSITIONS = {
    {221.141205, 122.646004, 2.43404675, 220.141205, 122.646004, 2.43404675, 0, 1, 0,
     43.6028175},
    {-2.70786691, 85.4769516, 240.523529, -3.30121756, 85.4485712, 239.718582,
     -0.00141029898, 0.999997199, -0.00191321515, 22.6198654},
    {247.908615, 63.4503365, 125.32412, 246.917603, 63.4553365, 125.1903, 0, 1, 0,
     20.4079475},
    {246.201401, 177.455338, 38.538826, 245.696762, 176.740402, 38.0548897, -0.516015887,
     0.699185967, -0.494840115, 22.6198654},
    {231.791519, 163.256424, 77.3447189, 231.243347, 162.608231, 76.8161774, -0.466618747,
     0.76148057, -0.44990477, 22.6198654},
};

std::vector<Float> SampleCameraPosition() {
    int num_reference_positions = REFERENCE_CAMERA_POSITIONS.size();

    int reference_idx_1 = rand() % num_reference_positions;

    return REFERENCE_CAMERA_POSITIONS[reference_idx_1];
    
    // int shift_from_1 = rand() % (num_reference_positions - 1) + 1;
    // int reference_idx_2 = (reference_idx_1 + shift_from_1) % num_reference_positions;

    // const std::vector<Float> &reference_1 = REFERENCE_CAMERA_POSITIONS[reference_idx_1];
    // const std::vector<Float> &reference_2 = REFERENCE_CAMERA_POSITIONS[reference_idx_2];

    // float coeff = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    // std::vector<Float> sampled_pos;
    // for (int i = 0; i < reference_1.size(); i++) {
    //     sampled_pos.push_back(coeff * reference_1[i] + (1. - coeff) * reference_2[i]);
    // }

    // return sampled_pos;
}

void ResetFilm(Film film) {
    Point2i resolution = film.FullResolution();

    for (int i = 0; i < resolution.x; i++) {
        for (int j = 0; j < resolution.y; j++) {
            film.ResetPixel(Point2i(i, j));
        }
    }
}

std::pair<std::string, std::string> SplitFilepath(std::string filepath) {
    size_t last_slash_idx = filepath.find_last_of("/");

    if (last_slash_idx == std::string::npos) {
        return {".", filepath};
    }

    std::string dir_path = filepath.substr(0, last_slash_idx);
    std::string filename = filepath.substr(last_slash_idx + 1);

    return {dir_path, filename};
}

void RunMultiviewPBRT(std::vector<std::string> &filenames, int num_views_to_render) {
    // Parse provided scene description files
    BasicScene scene;
    BasicSceneBuilder builder(&scene);
    ParseFiles(&builder, filenames);

    // Initial setup for scene done once for all camera views.
    SceneCache scene_cache = SetupScene(scene);

    // Get the directory and base filename we will save outputs to.
    std::string original_filepath = scene.GetCamera().GetFilm().GetFilename();

    auto [dir_path, original_filename] = SplitFilepath(original_filepath);

    // Main loop over all the camera positions to render.
    for (int view_idx = 0; view_idx < num_views_to_render; view_idx++) {
        const std::vector<float> camera_pos = SampleCameraPosition();

        builder.LookAt(camera_pos[0], camera_pos[1], camera_pos[2], camera_pos[3],
                       camera_pos[4], camera_pos[5], camera_pos[6], camera_pos[7],
                       camera_pos[8], {});

        builder.SetForced(true);
        builder.UpdateCameraTransform();
        builder.WorldBegin({});
        builder.SetForced(false);

        std::string new_filename =
            "idx_" + std::to_string(view_idx) + "-" + original_filename;
        std::string new_filepath = dir_path + "/" + new_filename;
        scene.GetCamera().GetFilm().SetFilename(new_filepath);

        // Render the scene
        if (Options->useGPU || Options->wavefront) {
            RenderWavefront(scene);
        } else {
            RenderCPU(scene, scene_cache);
        }
    }

    // Cleanup image texture cache.
    ImageTextureBase::ClearCache();

    LOG_VERBOSE("Memory used after post-render cleanup: %s", GetCurrentRSS());
    // Clean up after rendering the scene
    CleanupPBRT();
}

// main program
int main(int argc, char *argv[]) {
    // Convert command-line arguments to vector of strings
    std::vector<std::string> args = GetCommandLineArguments(argv);

    // Declare variables for parsed command line
    PBRTOptions options;
    std::vector<std::string> filenames;
    std::string logLevel = "error";
    std::string renderCoordSys = "cameraworld";
    bool format = false, toPly = false;

    int num_views_to_render = 1;

    // Process command-line arguments
    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        if ((*iter)[0] != '-') {
            filenames.push_back(*iter);
            continue;
        }

        auto onError = [](const std::string &err) {
            usage(err);
            exit(1);
        };

        std::string cropWindow, pixelBounds, pixel, pixelMaterial;
        if (ParseArg(&iter, args.end(), "cropwindow", &cropWindow, onError)) {
            std::vector<Float> c = SplitStringToFloats(cropWindow, ',');
            if (c.size() != 4) {
                usage("Didn't find four values after --cropwindow");
                return 1;
            }
            options.cropWindow = Bounds2f(Point2f(c[0], c[2]), Point2f(c[1], c[3]));
        } else if (ParseArg(&iter, args.end(), "pixel", &pixel, onError)) {
            std::vector<int> p = SplitStringToInts(pixel, ',');
            if (p.size() != 2) {
                usage("Didn't find two values after --pixel");
                return 1;
            }
            options.pixelBounds =
                Bounds2i(Point2i(p[0], p[1]), Point2i(p[0] + 1, p[1] + 1));
        } else if (ParseArg(&iter, args.end(), "pixelbounds", &pixelBounds, onError)) {
            std::vector<int> p = SplitStringToInts(pixelBounds, ',');
            if (p.size() != 4) {
                usage("Didn't find four integer values after --pixelbounds");
                return 1;
            }
            options.pixelBounds = Bounds2i(Point2i(p[0], p[2]), Point2i(p[1], p[3]));
        } else if (ParseArg(&iter, args.end(), "pixelmaterial", &pixelMaterial,
                            onError)) {
            std::vector<int> p = SplitStringToInts(pixelMaterial, ',');
            if (p.size() != 2) {
                usage("Didn't find two values after --pixelmaterial");
                return 1;
            }
            options.pixelMaterial = Point2i(p[0], p[1]);
        } else if (
#ifdef PBRT_BUILD_GPU_RENDERER
            ParseArg(&iter, args.end(), "gpu", &options.useGPU, onError) ||
            ParseArg(&iter, args.end(), "gpu-device", &options.gpuDevice, onError) ||
#endif
            ParseArg(&iter, args.end(), "debugstart", &options.debugStart, onError) ||
            ParseArg(&iter, args.end(), "disable-image-textures",
                     &options.disableImageTextures, onError) ||
            ParseArg(&iter, args.end(), "disable-pixel-jitter",
                     &options.disablePixelJitter, onError) ||
            ParseArg(&iter, args.end(), "disable-texture-filtering",
                     &options.disableTextureFiltering, onError) ||
            ParseArg(&iter, args.end(), "disable-wavelength-jitter",
                     &options.disableWavelengthJitter, onError) ||
            ParseArg(&iter, args.end(), "displacement-edge-scale",
                     &options.displacementEdgeScale, onError) ||
            ParseArg(&iter, args.end(), "display-server", &options.displayServer,
                     onError) ||
            ParseArg(&iter, args.end(), "force-diffuse", &options.forceDiffuse,
                     onError) ||
            ParseArg(&iter, args.end(), "format", &format, onError) ||
            ParseArg(&iter, args.end(), "log-level", &logLevel, onError) ||
            ParseArg(&iter, args.end(), "log-utilization", &options.logUtilization,
                     onError) ||
            ParseArg(&iter, args.end(), "log-file", &options.logFile, onError) ||
            ParseArg(&iter, args.end(), "interactive", &options.interactive, onError) ||
            ParseArg(&iter, args.end(), "fullscreen", &options.fullscreen, onError) ||
            ParseArg(&iter, args.end(), "mse-reference-image", &options.mseReferenceImage,
                     onError) ||
            ParseArg(&iter, args.end(), "mse-reference-out", &options.mseReferenceOutput,
                     onError) ||
            ParseArg(&iter, args.end(), "nthreads", &options.nThreads, onError) ||
            ParseArg(&iter, args.end(), "outfile", &options.imageFile, onError) ||
            ParseArg(&iter, args.end(), "pixelstats", &options.recordPixelStatistics,
                     onError) ||
            ParseArg(&iter, args.end(), "quick", &options.quickRender, onError) ||
            ParseArg(&iter, args.end(), "quiet", &options.quiet, onError) ||
            ParseArg(&iter, args.end(), "render-coord-sys", &renderCoordSys, onError) ||
            ParseArg(&iter, args.end(), "seed", &options.seed, onError) ||
            ParseArg(&iter, args.end(), "spp", &options.pixelSamples, onError) ||
            ParseArg(&iter, args.end(), "stats", &options.printStatistics, onError) ||
            ParseArg(&iter, args.end(), "toply", &toPly, onError) ||
            ParseArg(&iter, args.end(), "wavefront", &options.wavefront, onError) ||
            ParseArg(&iter, args.end(), "write-partial-images",
                     &options.writePartialImages, onError) ||
            ParseArg(&iter, args.end(), "upgrade", &options.upgrade, onError) ||
            ParseArg(&iter, args.end(), "num-views-to-render", &num_views_to_render,
                     onError)) {
            // success
        } else if (*iter == "--help" || *iter == "-help" || *iter == "-h") {
            usage();
            return 0;
        } else {
            usage(StringPrintf("argument \"%s\" unknown", *iter));
            return 1;
        }
    }

    // Print welcome banner
    if (!options.quiet && !format && !toPly && !options.upgrade) {
        printf("pbrt version 4 (built %s at %s)\n", __DATE__, __TIME__);
#ifdef PBRT_DEBUG_BUILD
        LOG_VERBOSE("Running debug build");
        printf("*** DEBUG BUILD ***\n");
#endif
        printf("Copyright (c)1998-2021 Matt Pharr, Wenzel Jakob, and Greg Humphreys.\n");
        printf("The source code to pbrt (but *not* the book contents) is covered "
               "by the Apache 2.0 License.\n");
        printf("See the file LICENSE.txt for the conditions of the license.\n");
        fflush(stdout);
    }

    // Check validity of provided arguments
    if (renderCoordSys == "camera")
        options.renderingSpace = RenderingCoordinateSystem::Camera;
    else if (renderCoordSys == "cameraworld")
        options.renderingSpace = RenderingCoordinateSystem::CameraWorld;
    else if (renderCoordSys == "world")
        options.renderingSpace = RenderingCoordinateSystem::World;
    else
        ErrorExit("%s: unknown rendering coordinate system.", renderCoordSys);

    if (!options.mseReferenceImage.empty() && options.mseReferenceOutput.empty())
        ErrorExit("Must provide MSE reference output filename via "
                  "--mse-reference-out");
    if (!options.mseReferenceOutput.empty() && options.mseReferenceImage.empty())
        ErrorExit("Must provide MSE reference image via --mse-reference-image");

    if (options.pixelMaterial && options.useGPU) {
        Warning("Disabling --use-gpu since --pixelmaterial was specified.");
        options.useGPU = false;
    }

    if (options.useGPU && options.wavefront)
        Warning("Both --gpu and --wavefront were specified; --gpu takes precedence.");

    if (options.pixelMaterial && options.wavefront) {
        Warning("Disabling --wavefront since --pixelmaterial was specified.");
        options.wavefront = false;
    }

    if (options.interactive && !(options.useGPU || options.wavefront))
        ErrorExit("The --interactive option is only supported with the --gpu "
                  "and --wavefront integrators.");

    if (options.fullscreen && !options.interactive) {
        ErrorExit("The --fullscreen option is only supported in interactive mode");
    }

    if (options.interactive && options.quickRender) {
        ErrorExit("The --quick option is not supported in interactive mode");
    }

    options.logLevel = LogLevelFromString(logLevel);

    // Initialize pbrt
    InitPBRT(options);

    if (format || toPly || options.upgrade) {
        FormattingParserTarget formattingTarget(toPly, options.upgrade);
        ParseFiles(&formattingTarget, filenames);
    } else {
        RunMultiviewPBRT(filenames, num_views_to_render);
    }
    return 0;
}

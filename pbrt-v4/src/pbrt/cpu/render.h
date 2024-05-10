// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_RENDER_H
#define PBRT_CPU_RENDER_H

#include <map>
#include <vector>

#include "primitive.h"
#include <pbrt/pbrt.h>

namespace pbrt {

class BasicScene;

struct SceneCache {
    std::map<std::string, Medium> media;
    std::vector<Light> lights;
    std::map<std::string, pbrt::Material> namedMaterials;
    std::vector<pbrt::Material> materials;
    Primitive accel;
};

SceneCache SetupScene(BasicScene &scene);

void RenderCPU(BasicScene &scene, SceneCache &scene_cache);

void RenderCPU(BasicScene &scene, bool cleanup = true);

}  // namespace pbrt

#endif  // PBRT_CPU_RENDER_H

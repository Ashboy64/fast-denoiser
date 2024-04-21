# Fast Denoiser

Includes the PBRT renderer in `/pbrt-v4` to generate path traced images at specified
samples per pixel.
- PBRT source: https://github.com/mmp/pbrt-v4
- PBRT textbook: https://pbr-book.org/4ed/contents

## Using PBRT (path tracing renderer)

To build the renderer:

```
cd pbrt-v4

mkdir build

cd build

cmake ..

make -j 8
```

To render the scene in `scenes/teapot`, execute
`./pbrt ../../scenes/teapot/scene-v4.pbrt --outfile teapot.png`. This will save
the image in `build/teapot.png`. To manually set the number of samples per 
pixel, add the flag `--spp {DESIRED_SPP}`.

## Todo

- Literature review.
    - See https://docs.google.com/document/d/1fvU3g4tNicwZnzI7WH_lvSu6Oyq6xNM1WnRFULMiHE0/edit?usp=sharing
- Script to generate the dataset.
- Try to get SBMC codebase working: https://github.com/adobe/sbmc
    - Will require modifying PBRT to save individual samples to disk in addition
    to the final image.
- Build image denoising.
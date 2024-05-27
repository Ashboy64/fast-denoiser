## Usage

Render dataset:

```
On Mac:
    export BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender

On GCP:
    export BLENDER_PATH=/home/ashishrao/cs348k/blender_data_gen/blender-4.1.1-linux-x64/blender

$BLENDER_PATH -b -P render.py
```

Transfer folder to GCP instance:

```
scp -r blender_data_gen ashishrao@cs244b-2.us-west4-a.cs-244b:/home/ashishrao/cs348k/fast-denoiser
```
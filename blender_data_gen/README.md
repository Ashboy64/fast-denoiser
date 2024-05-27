## Usage
```
export BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender

$BLENDER_PATH -b -P render.py

Script to render images for the Amazon Lumberyard Bistro script

options:
  -h, --help            show this help message and exit
  -f FBX_SCENE_PATH, --fbx_scene_path FBX_SCENE_PATH
                        Path to the BistroExterior.fbx file
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Folder where the rendering output is saved
  -s START, --start START
                        The first index (inclusive) of images to be rendered (as per the script's own rendering order)
  -e END, --end END     The last index (inclusive) of images to be rendered (as per the script's own rendering order)
  -g, --gpu             Use GPU (Nvidia/CUDA only)
```

2. `gen_split.py`: A script which generates 3 text files containing the names of images corresponding to a training,
validation and test split. The data is chosen such that each set has unique points from all sections of the scene. 
The data is split in an 60-20-20 ratio.

## Usage
```
python gen_split.py
```

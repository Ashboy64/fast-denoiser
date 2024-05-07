#!/bin/bash

# Set the base command
base_command="./pbrt"

# Set the output directory for .pbrt files
pbrt_output_directory="../scenes/watercolor/watercolor"

# Set the output directory for rendered images
rendered_output_directory="rendered_images"

# Set the image dimensions
width=64
height=64

# Set the number of samples per pixel for each image
spp_low=8
spp_high=4096

# Set the number of camera positions to generate
num_cameras=10

# Create the rendered output directory if it doesn't exist
mkdir -p "$rendered_output_directory"

# Define the camera parameters
camera_params=(
  "221.141205 122.646004 2.43404675 220.141205 122.646004 2.43404675 0 1 0 43.6028175"
  "-2.70786691 85.4769516 240.523529 -3.30121756 85.4485712 239.718582 -0.00141029898 0.999997199 -0.00191321515 22.6198654"
  "247.908615 63.4503365 125.32412 246.917603 63.4553365 125.1903 0 1 0 20.4079475"
  "246.201401 177.455338 38.538826 245.696762 176.740402 38.0548897 -0.516015887 0.699185967 -0.494840115 22.6198654"
  "231.791519 163.256424 77.3447189 231.243347 162.608231 76.8161774 -0.466618747 0.76148057 -0.44990477 22.6198654"
)

# Generate and render images for each camera position
for ((i=1; i<=num_cameras; i++))
do
  # Generate random camera parameters by interpolating between the given examples
  index1=$((RANDOM % ${#camera_params[@]}))
  index2=$((RANDOM % ${#camera_params[@]}))
  
  IFS=' ' read -r -a params1 <<< "${camera_params[$index1]}"
  IFS=' ' read -r -a params2 <<< "${camera_params[$index2]}"
  
  ratio=$(awk -v min=0 -v max=1 'BEGIN{srand(); print min+rand()*(max-min)}')
  
  eye_x=$(awk -v v1="${params1[0]}" -v v2="${params2[0]}" -v r="$ratio" 'BEGIN{print v1*(1-r)+v2*r}')
  eye_y=$(awk -v v1="${params1[1]}" -v v2="${params2[1]}" -v r="$ratio" 'BEGIN{print v1*(1-r)+v2*r}')
  eye_z=$(awk -v v1="${params1[2]}" -v v2="${params2[2]}" -v r="$ratio" 'BEGIN{print v1*(1-r)+v2*r}')
  
  lookat_x=$(awk -v v1="${params1[3]}" -v v2="${params2[3]}" -v r="$ratio" 'BEGIN{print v1*(1-r)+v2*r}')
  lookat_y=$(awk -v v1="${params1[4]}" -v v2="${params2[4]}" -v r="$ratio" 'BEGIN{print v1*(1-r)+v2*r}')
  lookat_z=$(awk -v v1="${params1[5]}" -v v2="${params2[5]}" -v r="$ratio" 'BEGIN{print v1*(1-r)+v2*r}')
  
  up_x=$(awk -v v1="${params1[6]}" -v v2="${params2[6]}" -v r="$ratio" 'BEGIN{print v1*(1-r)+v2*r}')
  up_y=$(awk -v v1="${params1[7]}" -v v2="${params2[7]}" -v r="$ratio" 'BEGIN{print v1*(1-r)+v2*r}')
  up_z=$(awk -v v1="${params1[8]}" -v v2="${params2[8]}" -v r="$ratio" 'BEGIN{print v1*(1-r)+v2*r}')
  
  fov=$(awk -v v1="${params1[9]}" -v v2="${params2[9]}" -v r="$ratio" 'BEGIN{print v1*(1-r)+v2*r}')
  
  # Create the .pbrt file with the interpolated camera parameters
  pbrt_file="${pbrt_output_directory}/random_camera_$i.pbrt"
  cat > "$pbrt_file" <<EOL
Sampler "halton"

Integrator "volpath" "integer maxdepth" 15

Film "gbuffer"
     "integer yresolution" [ $height ] "integer xresolution" [ $width ]
     "string filename" [ "random_camera_$i.exr" ]

Scale -1 1 1

LookAt $eye_x $eye_y $eye_z
       $lookat_x $lookat_y $lookat_z
       $up_x $up_y $up_z

Camera "perspective"
       "float fov" [ $fov ]

WorldBegin
Include "lights-no-windowglass.pbrt"
Include "materials.pbrt"
Include "geometry.pbrt"
EOL

  # Render the low-sample image
  low_sample_command="$base_command $pbrt_file --outfile ${rendered_output_directory}/random_camera_${i}_low.exr --spp $spp_low"
  $low_sample_command

  # Render the high-sample image
  high_sample_command="$base_command $pbrt_file --outfile ${rendered_output_directory}/random_camera_${i}_high.exr --spp $spp_high"
  $high_sample_command
done
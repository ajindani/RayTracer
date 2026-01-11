# RayTracer
A ray tracer that implements Ambient, Diffuse, and Specular (ADS) lighting, with support for shadows and reflective surfaces. Capable of rendering scenes with multiple objects and configurable light sources, allowing fine-grained control over lighting and material properties.


## Features:
*Ambient, Diffuse, and Specular (ADS) lighting model
*Shadows and mirror-like reflections
*Multiple objects per scene
*Configurable lighting and material parameters
*Outputs rendered images in PPM format

## Requirements:
*Python 3
*NumPy

## Usage
The program assumes the input scene file is located in the same directory as RayTracer.py.

Run the ray tracer using:
python RayTracer.py inputFileName.txt
or
py RayTracer.py inputFileName.txt

Note: Running the program on a test file will overwrite the existing PPM output for that test case.

## Test Cases
Sample PPM outputs from testing are included for reference and validation.

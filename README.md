# Face Morphing

## Process
1. Locate face points
	- Use dlib Face Recognition library to locate facial landmarks points
	- Find averages of facial landmark points
2. Align faces 
	- Center, align, & resize or crop input images using facial landmarks as reference
3. Warp 
	- Triangulate face points
	- Apply affine transforms to each triangle with bilinear interpolation
4. Morph 
	- Collect weighted-averaged-morphs to form a video of morphing one inputted image to the other

Please refer to attached PPT for a more detailed description of the face morphing process

## Usage

### Requirements
* Python3
* (Optional: Python3 module `virtualenv`)
* [dlib face recognition library](https://github.com/davisking/dlib)
* `numpy`: lineary algebra library
* `scikit_image`: image processing library
* `opencv_python`: computer vision library
* `Pillow`: image processing library


### Installation
* Verify you have Python3 installed (with pip)
* Verify you have the Python module `virtualenv` installed
* Run:
```
virtualenv virt
source virt/bin/activate
pip install -r requirements.txt
```

### Running 
After installation, run the code with `python morphing/run.py -s <SOURCE_IMAGE> -d <DESTINATION_IMAGE> -o <OUTPUT_FILENAME`
Example: `python morphing/run.py -s demos/esther.jpeg -d demos/benji.jpeg -o morphing_video.mp4`
Note: program works best if the two images are already the same size


# Face Morphing Notes

## Process
1. Align faces 
	- Center, align, & resize or crop input images
2. Locate face points
	- Use dlib Face Recognition library to locate facial landmarks points
	- Find averages of facial landmark points
3. Warp 
	- Triangulate face points
	- Apply affine transforms to each triangle with bilinear interpolation
4. Morph 
	- Collect weighted-averaged-morphs to form a video of morphing one inputted image to the other
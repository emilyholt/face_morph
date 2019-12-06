'''
Implementation of face morphing faces

'''
from detect_landmarks import crop_images, find_landmarks_set, average_landmarks
from triangulate import delaunay_triangulation, draw_delaunay_triangles
from warp import generate_midmorphs, create_video
import subprocess
import argparse
import shutil
import os

# Make sure we can find pre-trained model
MORPH_PATH = 'morphing'
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, MORPH_PATH)

def morph(predictor, src_img, dest_img, output):
	cropped_images = crop_images(src_img, dest_img)
	cropped_src = cropped_images[0]
	cropped_dest = cropped_images[1]
	[src_size, dest_size, src_landmarks, dest_landmarks] = find_landmarks_set(predictor, cropped_src, cropped_dest)
	
	avg_landmarks = average_landmarks(src_landmarks, dest_landmarks, dest_size)
	if(dest_size[0] == 0):
		print("error: couldn't find a face in the image " + dest_size[1])
		return
	
	delaunay_list = delaunay_triangulation(dest_size[1], dest_size[0], avg_landmarks)
	# Uncomment following two lines to view delaunay triangulation animation (resource-intensive and slows down program)
	# draw_delaunay_triangles(cropped_src, src_size[1], src_size[0], src_landmarks)
	# draw_delaunay_triangles(cropped_dest, dest_size[1], dest_size[0], dest_landmarks)
	mid_morphs = generate_midmorphs(cropped_src, cropped_dest, src_landmarks, dest_landmarks, delaunay_list)
	create_video(mid_morphs, dest_size, output)

if __name__ == "__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument("-s", "--src", help="source image")
	parser.add_argument("-d", "--dest", help="destination image")
	parser.add_argument("-o", "--output", help="output video")
	args=parser.parse_args()

	with open(args.src,'rb') as src_img, open(args.dest,'rb') as dest_img:
		morph(os.path.join(MORPH_PATH, 'shape_predictor_68_face_landmarks.dat'), src_img, dest_img, args.output)

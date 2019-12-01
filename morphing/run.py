from face_landmark_detection import crop_images, average_faces
from delaunay import delaunay_triangulation
from morph import morph_video
import subprocess
import argparse
import shutil
import os

MORPH_PATH = '/Users/moose/GraphTheory/face_morph/morphing'
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, MORPH_PATH)

def morph(predictor, src_img, dest_img, output):
	[dest_size, cropped_src, cropped_dest, src_landmarks, dest_landmarks, avg_landmarks] = average_faces(predictor, src_img, dest_img)
	if(dest_size[0] == 0):
		print("error: couldn't find a face in the image " + dest_size[1])
		return
	delaunay_list = delaunay_triangulation(dest_size[1], dest_size[0], avg_landmarks)
	morph_video(cropped_src, cropped_dest, src_landmarks, dest_landmarks, delaunay_list, dest_size, output)

if __name__ == "__main__":

	parser=argparse.ArgumentParser()
	parser.add_argument("-s", "--src", help="source image")
	parser.add_argument("-d", "--dest", help="destination image")
	parser.add_argument("-o", "--output", help="output video")
	args=parser.parse_args()

	with open(args.src,'rb') as src_img, open(args.dest,'rb') as dest_img:
		morph(os.path.join(MORPH_PATH, 'shape_predictor_68_face_landmarks.dat'), src_img, dest_img, args.output)

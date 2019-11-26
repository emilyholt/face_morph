from face_landmark_detection import average_faces
from delaunay import makeDelaunay
from faceMorph import makeMorphs
import subprocess
import argparse
import shutil
import os

MORPH_PATH = '/Users/moose/GraphTheory/face_morph/morphing'
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, MORPH_PATH)

def morph(predictor, src_img, dest_img, length, frame_rate, output):
	[dest_size, cropped_src, cropped_dest, src_landmarks, dest_landmarks, avg_landmarks] = average_faces(predictor, src_img, dest_img)
	if(dest_size[0] == 0):
		print("error: couldn't find a face in the image " + dest_size[1])
		return
	delaunay_list = makeDelaunay(dest_size[1], dest_size[0], avg_landmarks)
	makeMorphs(length, frame_rate, cropped_src, cropped_dest, src_landmarks, dest_landmarks, delaunay_list, dest_size, output)

if __name__ == "__main__":

	parser=argparse.ArgumentParser()
	parser.add_argument("-s", "--src", help="source image")
	parser.add_argument("-d", "--dest", help="destination image")
	parser.add_argument("-l", "--length", type=int, help="length of video")
	parser.add_argument("-f", "--frame_rate", type=int, help="frame rate or video")
	parser.add_argument("-o", "--output", help="output video")
	args=parser.parse_args()

	with open(args.src,'rb') as src_img, open(args.dest,'rb') as dest_img:
		morph(os.path.join(MORPH_PATH, 'shape_predictor_68_face_landmarks.dat'), src_img, dest_img, args.length, args.frame_rate, args.output)

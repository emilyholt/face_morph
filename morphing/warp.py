'''
Implementation of warping delaunay-triangulated faces into a video

Reference implementations: 
- https://github.com/cirbuk/face-morphing
- https://github.com/alyssaq/face_morpher

'''

import numpy as np
import cv2
import sys
import os
import math
from subprocess import Popen, PIPE
from PIL import Image, ImageDraw

VIDEO_LENGTH = 5
FRAME_RATE = 25

# Apply affine transform calculated using src_tri and dest_tri
def apply_affine_transform(src, src_tri, dest_tri, size) :
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dest_tri))
    
    # Apply affine transform to the src image
    dest = cv2.warpAffine( src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dest

# Alpha blend rectangular patches
def blend(src_affine_transform, dest_affine_transform, alpha):
    blended_rect = (1.0 - alpha) * src_affine_transform + alpha * dest_affine_transform
    return blended_rect

def generate_patches(img, tri_bounding_rect):
    rect_patch = img[tri_bounding_rect[1]:tri_bounding_rect[1] + tri_bounding_rect[3], 
                     tri_bounding_rect[0]:tri_bounding_rect[0] + tri_bounding_rect[2]]
    return rect_patch

# Warps and alpha blends triangular regions from src_img and dest_img to img
def warp_triangle(src_img, dest_img, output_img, src_tri, dest_tri, avg_tri, alpha) :
    
    # Find bounding rectangle for each triangle
    src_tri_bounding_rect = cv2.boundingRect(np.float32([src_tri]))
    dest_tri_bounding_rect = cv2.boundingRect(np.float32([dest_tri]))
    avg_tri_bounding_rect = cv2.boundingRect(np.float32([avg_tri]))

    # Offset points by left top corner of the respective rectangles
    offset_src_tri = []
    offset_dest_tri = []
    offset_avg_tri = []

    for i in range(0, 3):
        offset_avg_tri.append(((avg_tri[i][0] - avg_tri_bounding_rect[0]), 
                               (avg_tri[i][1] - avg_tri_bounding_rect[1])))
        offset_src_tri.append(((src_tri[i][0] - src_tri_bounding_rect[0]), 
                               (src_tri[i][1] - src_tri_bounding_rect[1])))
        offset_dest_tri.append(((dest_tri[i][0] - dest_tri_bounding_rect[0]), 
                                (dest_tri[i][1] - dest_tri_bounding_rect[1])))

    # Get mask by filling triangle
    mask = np.zeros((avg_tri_bounding_rect[3], avg_tri_bounding_rect[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(offset_avg_tri), (1.0, 1.0, 1.0), 16, 0);

    # Apply warp to small rectangular patches
    src_img_patch = generate_patches(src_img, src_tri_bounding_rect)
    dest_img_patch = generate_patches(dest_img, dest_tri_bounding_rect)

    size = (avg_tri_bounding_rect[2], avg_tri_bounding_rect[3])
    src_affine_transform = apply_affine_transform(src_img_patch, offset_src_tri, offset_avg_tri, size)
    dest_affine_transform = apply_affine_transform(dest_img_patch, offset_dest_tri, offset_avg_tri, size)

    # Blend patches
    blended_rect = blend(src_affine_transform, dest_affine_transform, alpha)

    # Copy triangular region of the rectangular patch to the output image
    y_start = avg_tri_bounding_rect[1]
    y_end = avg_tri_bounding_rect[1] + avg_tri_bounding_rect[3]
    x_start = avg_tri_bounding_rect[0]
    x_end = avg_tri_bounding_rect[0] + avg_tri_bounding_rect[2]
    output_img[y_start:y_end, x_start:x_end] = output_img[y_start:y_end, x_start:x_end] * (1 - mask) + blended_rect * mask

def weighted_average(src_landmarks, dest_landmarks, alpha):
    # Generate a set of landmarks that are weighted to be closer in distance
    # to either the src or dest based on the value of alpha
    midpts = []
    for i in range(0, len(src_landmarks)):
        x = ( 1 - alpha ) * src_landmarks[i][0] + alpha * dest_landmarks[i][0]
        y = ( 1 - alpha ) * src_landmarks[i][1] + alpha * dest_landmarks[i][1]
        midpts.append((x,y))
    return midpts

def generate_midmorphs(src_img, dest_img, src_landmarks, dest_landmarks, delaunay_list):
    # Generate a series of morphed faces based on the weighted 
    # averages of the facial landmark points
    collected_mid_morphs = []

    # Specify the number of 'mid-morph' images needed for the video
    total_midpts=int(VIDEO_LENGTH * FRAME_RATE)
    for j in range(0, total_midpts):

        # Convert Mat to float data type
        src_img = np.float32(src_img)
        dest_img = np.float32(dest_img)

        # Compute weighted average point coordinates
        alpha = j / (total_midpts - 1)
        midpts = weighted_average(src_landmarks, dest_landmarks, alpha)

        # Allocate space for final output
        mid_morph = np.zeros(src_img.shape, dtype = src_img.dtype)

        # Read triangles from delaunay_list
        for i in range(len(delaunay_list)):    
            a = int(delaunay_list[i][0])
            b = int(delaunay_list[i][1])
            c = int(delaunay_list[i][2])
            
            src_tri = [src_landmarks[a], src_landmarks[b], src_landmarks[c]]
            dest_tri = [dest_landmarks[a], dest_landmarks[b], dest_landmarks[c]]
            avg_tri = [midpts[a], midpts[b], midpts[c]]

            # Morph one triangle at a time.
            warp_triangle(src_img, dest_img, mid_morph, src_tri, dest_tri, avg_tri, alpha)
        
        collected_mid_morphs.append(mid_morph)

    return collected_mid_morphs

def create_video(mid_morphs, dest_size, output):
    # Start the process for creating a video
    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(FRAME_RATE),'-s',str(dest_size[1])+'x'+str(dest_size[0]), '-i', '-', '-c:v', 'libx264', '-crf', '25','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-pix_fmt','yuv420p', output], stdin=PIPE)
    
    for mid_morph_img in mid_morphs:

        temp_res = cv2.cvtColor(np.uint8(mid_morph_img),cv2.COLOR_BGR2RGB)
        res = Image.fromarray(temp_res)
        res.save(p.stdin, 'JPEG')

    p.stdin.close()
    p.wait()

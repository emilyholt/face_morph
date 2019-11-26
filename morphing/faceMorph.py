import numpy as np
import cv2
import sys
import os
import math
from subprocess import Popen, PIPE
from PIL import Image

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from src_img and dest_img to img
def morph_triangle(src_img, dest_img, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    src_img_rect = src_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    dest_img_rect = dest_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(src_img_rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(dest_img_rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


def makeMorphs(length, frame_rate, src_img, dest_img, src_landmarks, dest_landmarks, delaunay_list, size, output):

    total_midpts=int(length * frame_rate)

    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate),'-s',str(size[1])+'x'+str(size[0]), '-i', '-', '-c:v', 'libx264', '-crf', '25','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-pix_fmt','yuv420p', output], stdin=PIPE)
    
    for j in range(0, total_midpts):

        # Convert Mat to float data type
        src_img = np.float32(src_img)
        dest_img = np.float32(dest_img)

        # Read array of corresponding points
        points = []
        alpha = j / (total_midpts - 1)

        # Compute weighted average point coordinates
        for i in range(0, len(src_landmarks)):
            x = ( 1 - alpha ) * src_landmarks[i][0] + alpha * dest_landmarks[i][0]
            y = ( 1 - alpha ) * src_landmarks[i][1] + alpha * dest_landmarks[i][1]
            points.append((x,y))


        # Allocate space for final output
        imgMorph = np.zeros(src_img.shape, dtype = src_img.dtype)

        # Read triangles from delaunay_list
        for i in range(len(delaunay_list)):    
            a = int(delaunay_list[i][0])
            b = int(delaunay_list[i][1])
            c = int(delaunay_list[i][2])
            
            t1 = [src_landmarks[a], src_landmarks[b], src_landmarks[c]]
            t2 = [dest_landmarks[a], dest_landmarks[b], dest_landmarks[c]]
            t = [points[a], points[b], points[c]]

            # Morph one triangle at a time.
            morph_triangle(src_img, dest_img, imgMorph, t1, t2, t, alpha)

        temp_res=cv2.cvtColor(np.uint8(imgMorph),cv2.COLOR_BGR2RGB)
        res=Image.fromarray(temp_res)
        res.save(p.stdin,'JPEG')

    p.stdin.close()
    p.wait()

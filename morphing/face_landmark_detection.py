import sys
import os
import dlib
import glob
import numpy as np
from skimage import io
import cv2

def crop_images(src, dest):
    # Load images:
    if(isinstance(src,str)):
        src_img = cv2.imread(src)
    else:
        src_img = cv2.imdecode(np.fromstring(src.read(), np.uint8),1)
    if(isinstance(dest,str)):
        dest_img = cv2.imread(dest)
    else:
        dest_img = cv2.imdecode(np.fromstring(dest.read(), np.uint8),1)

    src_size = src_img.shape
    dest_size = dest_img.shape

    x_diff = (src_size[0] - dest_size[0])//2
    y_diff = (src_size[1] - dest_size[1])//2
    x_avg = (src_size[0] + dest_size[0])//2
    y_avg = (src_size[1] + dest_size[1])//2

    if(src_size[0] == dest_size[0] and src_size[1] == dest_size[1]):
        return [src_img,dest_img]

    elif(src_size[0] <= dest_size[0] and src_size[1] <= dest_size[1]):
        x_scale=src_size[0]/dest_size[0]
        y_scale=src_size[1]/dest_size[1]

        if(x_scale > y_scale):
            resized_dest = cv2.resize(dest_img, None, x_scale, x_scale, interpolation=cv2.INTER_AREA)
        else:
            resized_dest = cv2.resize(dest_img, None, y_scale, y_scale, interpolation=cv2.INTER_AREA)
        return cropping_dimensions(src_img, resized_dest)

    elif(src_size[0] >= dest_size[0] and src_size[1] >= dest_size[1]):
        x_scale = dest_size[0]/src_size[0]
        y_scale = dest_size[1]/src_size[1]

        if(x_scale > y_scale):
            resized_src = cv2.resize(src_img, None, x_scale, x_scale, interpolation=cv2.INTER_AREA)
        else:
            resized_src = cv2.resize(src_img, None, y_scale, y_scale, interpolation=cv2.INTER_AREA)
        return cropping_dimensions(resized_src, dest_img)

    elif(src_size[0] >= dest_size[0] and src_size[1] <= dest_size[1]):
        return [src_img[x_diff:x_avg, :], dest_img[:, -y_diff:y_avg]]

    else:
        return [src_img[:, y_diff:y_avg], dest_img[-x_diff:x_avg, :]]

def cropping_dimensions(src_img, dest_img):

    src_size = src_img.size
    dest_size = dest_img.size

    x_diff = (src_size[0] - dest_size[0])//2
    y_diff = (src_size[1] - dest_size[1])//2
    x_avg = (src_size[0] + dest_size[0])//2
    y_avg = (src_size[1] + dest_size[1])//2

    # if src and dest are the same size, there's no need to crop:
    if(src_size[0] == dest_size[0] and src_size[1] == dest_size[1]):
        return [src_img, dest_img]

    # if src is smaller than dest, crop dest
    elif(src_size[0] <= dest_size[0] and src_size[1] <= dest_size[1]):
        return [src_img, dest_img[-x_diff:x_avg, -y_diff:y_avg]]

    # if src is larger than dest, crop src
    elif(src_size[0] >= dest_size[0] and src_size[1] >= dest_size[1]):
        return [src_img[x_diff:x_avg, y_diff:y_avg], dest_img]

    # if src's width is larger that dest's width, but dest's height is larger that src's height,
    elif(src_size[0] >= dest_size[0] and src_size[1] <= dest_size[1]):
        return [src_img[x_diff:x_avg, :], dest_img[:, -y_diff:y_avg]]

    else:
        return [src_img[:, y_diff:y_avg], dest_img[-x_diff:x_avg, :]]

def find_landmarks(predictor_path, cropped_img, landmarks_matrix):
    size=(cropped_img.shape[0], cropped_img.shape[1])

    # Detect the points of face.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Detect face in img
    dets = detector(cropped_img, 1)
    if(len(dets) == 0):
        return("ERROR: No faces detected")

    landmarks_list = []
    landmarks_matrix = array = np.zeros((68,2))

    # Predict facial landmarks
    for k, d in enumerate(dets):
        
        # Get the landmarks/parts for the face in box d.
        shape = predictor(cropped_img, d)
        for i in range(0,68):
            landmarks_list.append((int(shape.part(i).x),int(shape.part(i).y)))
            landmarks_matrix[i][0] += shape.part(i).x
            landmarks_matrix[i][1] += shape.part(i).y

        landmarks_list.append((1,1))
        landmarks_list.append((size[1]-1,1))
        landmarks_list.append(((size[1]-1)//2,1))
        landmarks_list.append((1,size[0]-1))
        landmarks_list.append((1,(size[0]-1)//2))
        landmarks_list.append(((size[1]-1)//2,size[0]-1))
        landmarks_list.append((size[1]-1,size[0]-1))
        landmarks_list.append(((size[1]-1)//2,(size[0]-1)//2))

    return landmarks_list, landmarks_matrix

def makeCorrespondence(predictor_path, src_img, dest_img):

    # Detect the points of face.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Setting up some initial values.
    zeros_landmarks = np.zeros((68,2))
    size=(0,0)
    cropped_images = crop_images(src_img, dest_img)

    src_img_landmarks=[]
    dest_img_landmarks=[]

    src_img_landmarks, src_landmarks_matrix = find_landmarks(predictor_path, cropped_images[0], zeros_landmarks)
    dest_img_landmarks, landmarks_matrix = find_landmarks(predictor_path, cropped_images[1], src_landmarks_matrix)
    
    size=(cropped_images[1].shape[0], cropped_images[1].shape[1])
    normalized_landmarks=landmarks_matrix/2
    normalized_landmarks=np.append(normalized_landmarks,[[1,1]],axis=0)
    normalized_landmarks=np.append(normalized_landmarks,[[size[1]-1,1]],axis=0)
    normalized_landmarks=np.append(normalized_landmarks,[[(size[1]-1)//2,1]],axis=0)
    normalized_landmarks=np.append(normalized_landmarks,[[1,size[0]-1]],axis=0)
    normalized_landmarks=np.append(normalized_landmarks,[[1,(size[0]-1)//2]],axis=0)
    normalized_landmarks=np.append(normalized_landmarks,[[(size[1]-1)//2,size[0]-1]],axis=0)
    normalized_landmarks=np.append(normalized_landmarks,[[size[1]-1,size[0]-1]],axis=0)
    normalized_landmarks=np.append(normalized_landmarks,[[(size[1]-1)//2,(size[0]-1)//2]],axis=0)

    return [size, cropped_images[0], cropped_images[1], src_img_landmarks, dest_img_landmarks, normalized_landmarks]


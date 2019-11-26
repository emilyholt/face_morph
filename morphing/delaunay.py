import cv2
import numpy as np
import random

# Check if a point is inside a rectangle
def point_in_rect(rect, point) :
    if (point[0] < rect[0]) or (point[1] < rect[1]) or (point[0] > rect[2]) or (point[1] > rect[3]) :
        return False
    return True

def makeDelaunay(img_height,img_width,avg_landmarks):
    
    # Make a rectangle
    rect = (0, 0, img_height, img_width)

    # Create a Subdiv
    subdiv = cv2.Subdiv2D(rect);

    # Make a landmark_coords list and a searchable landmark_coords_dict 
    avg_landmarks = avg_landmarks.tolist()
    landmark_coords = [(int(x[0]), int(x[1])) for x in avg_landmarks]
    landmark_coords_dict = {x[0]:x[1] for x in list(zip(landmark_coords, range(76)))}
    
    # Insert landmark_coords into subdiv
    for p in landmark_coords :
        subdiv.insert(p)
        
    # create the delaunay triangulation list
    triangleList = subdiv.getTriangleList();
    delaunay_list=[]

    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if point_in_rect(rect, pt1) and point_in_rect(rect, pt2) and point_in_rect(rect, pt3) :
            delaunay_list.append((landmark_coords_dict[pt1],landmark_coords_dict[pt2],landmark_coords_dict[pt3]))
    
    return delaunay_list


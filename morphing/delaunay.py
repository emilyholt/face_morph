'''
Implementation of Delaunay triangulation

Reference implementation: https://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/

'''

import cv2
import numpy as np
import random

# Check if a point is inside a rectangle
def point_in_rect(rect, point) :
    if (point[0] < rect[0]) or (point[1] < rect[1]) or (point[0] > rect[2]) or (point[1] > rect[3]) :
        return False
    return True

def delaunay_triangulation(img_height, img_width, avg_landmarks):

    # Make a landmark_coords list and a searchable landmark_coords_dict 
    landmark_coords = [(int(x[0]), int(x[1])) for x in avg_landmarks]
    landmark_coords_dict = {x[0]:x[1] for x in list(zip(landmark_coords, range(76)))}

    # Make a bounding rectangle
    rect = (0, 0, img_height, img_width)

    # Create a Subdiv 
    subdiv = cv2.Subdiv2D(rect);

    # Insert landmark_coords into the subdiv
    for coord in landmark_coords :
        subdiv.insert(coord)
        
    # use the subdiv create the delaunay triangulation list
    subdiv_triangles = subdiv.getTriangleList();
    delaunay_triangles=[]

    for triang in subdiv_triangles:
        triang_coord1 = (int(triang[0]), int(triang[1]))
        triang_coord2 = (int(triang[2]), int(triang[3]))
        triang_coord3 = (int(triang[4]), int(triang[5]))
        
        if point_in_rect(rect, triang_coord1) and \
           point_in_rect(rect, triang_coord2) and \
           point_in_rect(rect, triang_coord3) :
            delaunay_triangles.append((landmark_coords_dict[triang_coord1], 
                                       landmark_coords_dict[triang_coord2], 
                                       landmark_coords_dict[triang_coord3]))
    
    return delaunay_triangles


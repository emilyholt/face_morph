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


# Draw a point
def draw_point(img, p, color ) :
    cv2.circle( img, p, 2, color, -1, cv2.LINE_AA, 0 )


# Draw delaunay triangles
def draw_delaunay_lines(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        if point_in_rect(r, pt1) and point_in_rect(r, pt2) and point_in_rect(r, pt3) :
        
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)



def draw_delaunay_triangles(img, img_height, img_width, landmarks):

    # Define window names
    win_delaunay = "Delaunay Triangulation"

    # Turn on animation while drawing triangles
    animate = True
    
    # Define colors for drawing.
    delaunay_color = (255,255,255)
    points_color = (0, 0, 255)

    # Read in the image.
    # img = cv2.imread("demos/esther.jpeg");
    
    # Keep a copy around
    img_orig = img.copy();
    
    # Rectangle to be used with Subdiv2D
    # size = img.shape
    rect = (0, 0, img_height, img_width)
    
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect);

    # Insert points into subdiv
    for p in landmarks :
        p = (int(p[0]), int(p[1]))
        subdiv.insert(p)
        
        # Show animation
        if animate :
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay_lines( img_copy, subdiv, (255, 255, 255) );
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(100)

    # Draw delaunay triangles
    draw_delaunay_lines( img, subdiv, (255, 255, 255) );

    # Draw points
    for p in landmarks :
        p = (int(p[0]), int(p[1]))
        draw_point(img, p, (0,0,255))

    # Show results
    cv2.imshow(win_delaunay,img)
    cv2.waitKey(0)

#######################################################################

def delaunay_triangulation(img_height, img_width, avg_landmarks):
    
    '''
    Triangulation of the dest_img needs to have the same patterns of the triangulation 
    of the src_img, meaning the connection of the points has to be the same. After 
    the triangulation of the src_img, we use the indices of the landmark points in 
    the triangulation so we can replicate the same triangulation on the dest_img 
    '''

    # Make a landmark_coords list and a searchable landmark_coords_dict 
    landmark_coords = [(int(x[0]), int(x[1])) for x in avg_landmarks]
    landmark_coords_dict = {x[0]:x[1] for x in list(zip(landmark_coords, range(76)))}

    # Make a bounding rectangle
    rect = (0, 0, img_height, img_width)

    # Use OpenCV to subdivide the bounding rect 
    subdiv = cv2.Subdiv2D(rect);

    # Add landmark_coords to the subdiv
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


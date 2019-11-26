import cv2
import numpy as np
import random

# Check if a point is inside a rectangle
def point_in_rect(rect, point) :
    if (point[0] < rect[0]) or (point[1] < rect[1]) or (point[0] > rect[2]) or (point[1] > rect[3]) :
        return False
    return True

def makeDelaunay(theSize1,theSize0,theList):
    
    # Make a rectangle.
    rect = (0, 0, theSize1,theSize0)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect);

    # Make a points list and a searchable dictionary. 
    theList=theList.tolist()
    points=[(int(x[0]),int(x[1])) for x in theList]
    dictionary={x[0]:x[1] for x in list(zip(points,range(76)))}
    
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)
        
    # Make a delaunay triangulation list.
    delaunay_list=[]

    triangleList = subdiv.getTriangleList();
    r = (0, 0, theSize1,theSize0)

    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if point_in_rect(r, pt1) and point_in_rect(r, pt2) and point_in_rect(r, pt3) :
            delaunay_list.append((dictionary[pt1],dictionary[pt2],dictionary[pt3]))
    
    # Return the list.
    return delaunay_list


import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import skimage.exposure


def find_cetnroid(image:np.array)->tuple:
    ''''
    takes as input an image and outputs its centroid
    Use the formula of summation of points coordinates over their multiplication
    put the image
    https://stackoverflow.com/questions/49582008/center-of-mass-in-contour-python-opencv
    https://en.wikipedia.org/wiki/Centroid#Of_a_finite_set_of_points
    amelioration : 
        use moments instead of the formula below : 
        https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/
    '''
    kernel = np.ones((5,5),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE,kernel)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(0,0), sigmaX=3, sigmaY=3,borderType=cv2.BORDER_DEFAULT)
    gray = skimage.exposure.rescale_intensity(gray, in_range=(127.5,255),out_range=(0,255))
    gray = np.array(gray,dtype=np.uint8)


    Xc=0
    Yc= 0

    #find contours and calculate derivatives using xz
    contours,_=cv2.findContours(gray,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        contour=contours [0]
    except:
        return np.uint8(Xc),np.uint8(Yc)
    KpCnt = len(contour)
    
    for point in contour:
        Xc+=point[0][0]
        Yc+=point[0][1]

    return np.uint8(np.ceil(Xc/KpCnt)),np.uint8(np.ceil(Yc/KpCnt))


def find_extreme_point(image:np.array,thresh : int=30)->tuple:
    '''
    takes an image as input and outputs the extreme point of the contour
    based on the changes of the values of the normal
    '''

    #filtering and smoothing
    kernel = np.ones((5,5),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE,kernel)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(0,0), sigmaX=3, sigmaY=3,borderType=cv2.BORDER_DEFAULT)
    gray = skimage.exposure.rescale_intensity(gray, in_range=(127.5,255),out_range=(0,255))
    gray = np.array(gray,dtype=np.uint8)
    # Apply morphological closing to close gaps in the image
    kernel = np.ones((55, 55), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(gray)
    # plt.show()
    #find contours and calculate derivatives using xz
    contours,_=cv2.findContours(gray,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        contour=max(contours,key = cv2.contourArea)
    except:
        print("no contours were found")
        return [[0,0]]
    

    angles = []
    angles_diff = []
    possible_edges = []
    gradient_x=cv2.Sobel(gray,cv2.CV_64F, 1, 0,ksize=5)
    gradient_y=cv2.Sobel(gray,cv2.CV_64F, 0, 1,ksize=5)

    for i, point in enumerate(contour):
        x,y = point[0]
        gradient_x_at_point=gradient_x[y,x]
        gradient_y_at_point=gradient_y[y,x]

        normal_vector = (gradient_x_at_point,gradient_y_at_point)
        angle = math.atan2(gradient_y_at_point,gradient_x_at_point)*180/math.pi
        angles.append(angle)
        angles_diff.append(abs(abs(angles[i])-abs(angles[i-1])))
        if abs(angles[i]-angles[i-1])<thresh and thresh != 0:
            continue
        else :
            Xp=x
            Yp=y
            possible_edges.append((Xp,Yp))

    # plt.imshow(gradient_x)
    # plt.show()
    # plt.imshow(gradient_y)
    # plt.show()
    return possible_edges
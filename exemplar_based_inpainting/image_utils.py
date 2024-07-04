import cv2
import numpy as np

# https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
# compute gradients along the x and y axis, respectively
def image_gradients(img):
    if len(img.shape) == 3:    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 1:
        gray = img

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    # Compute the gradient magnitude
    magnitude = np.sqrt((gx ** 2) + (gy ** 2))
    magnitude[magnitude == 0] = 1

    return (magnitude, gx, gy)

# Get a ROI from an image using a ROI in [[min_y, max_y][min_x, max_x]] format
def get_roi(image, roi):
    return image[
        roi[0][0]:roi[0][1]+1,
        roi[1][0]:roi[1][1]+1
    ]

# Compute area of a ROI in [[min_y, max_y][min_x, max_x]] format
def roi_area(roi):
    return (1+roi[0][1]-roi[0][0]) * (1+roi[1][1]-roi[1][0])

def roi_shape(roi):
    return (1+roi[0][1]-roi[0][0]), (1+roi[1][1]-roi[1][0])

# Fill a ROI in an image. ROI in [[min_y, max_y][min_x, max_x]] format
def fill_roi(img, roi, roi_data):
    img[
        roi[0][0]:roi[0][1]+1,
        roi[1][0]:roi[1][1]+1
    ] = roi_data

def to_three_channels(image):
    height, width = image.shape
    return image.reshape(height, width, 1).repeat(3, axis=2)
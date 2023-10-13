import cv2
import numpy as np


def simple_image_masker(img, brush_color=(0, 0, 255)):
    img_cp = img.copy()

    cv2.namedWindow('Draw a mask', cv2.WINDOW_GUI_NORMAL)

    brush_size = 5
    def change_brush_size(val):
        nonlocal brush_size
        brush_size = val

    cv2.createTrackbar('Brush size', 'Draw a mask', 1, 1000, change_brush_size)
    
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    drawing = False
    def click_event(event, x, y, flags, param):
        nonlocal mask, img_cp, drawing
        if event == cv2.EVENT_LBUTTONDOWN:            
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(img_cp, (x, y), brush_size, brush_color, -1)
            cv2.circle(mask, (x, y), brush_size, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    cv2.setMouseCallback('Draw a mask', click_event)

    print("Draw a mask over the image. Usage:")
    print("  - Adapt brush size with the trackbar.")
    print("  - Press 'c' to clear the current mask.")
    print("  - Press 'a', 'ESC' or 'Return' to finish.")
    while True:
        cv2.imshow('Draw a mask', img_cp)
        k = cv2.waitKey(20) & 0xFF
        if k == 27 or k == 13 or k == ord('a'):
            break
        if k == ord("c"):
            img_cp = img.copy()
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)        

    return mask
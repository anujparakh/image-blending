import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

def GetMask(image):
    ### You can add any number of points by using 
    ### mouse left click. Delete points with mouse
    ### right click and finish adding by mouse
    ### middle click.  More info:
    ### https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ginput.html

    plt.imshow(image)
    plt.axis('image')
    points = plt.ginput(-1, timeout=-1)
    plt.close()

    ### The code below is based on this answer from stackoverflow
    ### https://stackoverflow.com/a/15343106

    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([points], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    np.set_printoptions(threshold=sys.maxsize)
    return mask


img = cv2.imread("../Images/tempmask_09.png", 0)
for y in range(0, img.shape[0]):
    for x in range(0, img.shape[1]):
        if (img [y] [x] != 0):
            img[y][x] = 255

cv2.imwrite("../Images/mask_09.jpg", img)

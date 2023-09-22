import cv2
import numpy as np


def removenoise(image):
    try:
        bg=cv2.imread("bagground.jpg",0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge=cv2.Canny(gray,20,70,5)
        cv2.imshow("edge",edge)
        # gray = cv2.bilateralFilter(edge.T, 13, 95, 95)
        thresh = cv2.threshold(edge.T, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        thresh=cv2.dilate(thresh,(3,3),iterations=4)
        height, width = thresh.shape
        # if height*width- np.count_nonzero(thresh==255)>(height*width)//2:
        #     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]


 
    except:
        thresh=image
    Thres=thresh.copy()
    height, width = thresh.shape
    

    summ = width*255
    for y in range(0, height):
        x_sum = np.sum(thresh[y][:])
        if summ-x_sum > summ//2:
            thresh[y][:] = 0
    # cv2.imshow("threshh",thresh.T)
    for y in range(0, height):
        if np.sum(thresh[y][:])==0:

            if y<200:
                thresh[:y][:]=0
                break

    for y in range( height-1,0,-1):
        if np.sum(thresh[y][:])==0:
            if y>height-200:
                thresh[y:][:]=0
                break
    return thresh.T,Thres.T


image = cv2.imread('image863.jpg')


new,thresh = removenoise(image)

cv2.imshow("new",new)
cv2.imshow("thresh",thresh)


cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
def blob_only(thresh):
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(
        thresh)

    sizes = stats[:, -1]

    sizes = sizes[1:]
    nb_blobs -= 1

    min_size = 200

    im_result = np.zeros_like(thresh)

    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:

            im_result[im_with_separated_blobs == blob + 1] = 255
    # print(im_result.shape)

 
    return im_result
def correctPosition(Image, rect):
        localImg = Image.copy()
        angle = rect[2]
        print(angle)
        if (abs(angle-0) < 10):
            angle = 0-angle

        if (abs(angle-90) < 10):
            angle = 90-angle
        if (abs(angle+90) < 10):
            angle = 90+angle
        if (abs(angle-180) < 10):
            angle = 180-angle
        if (abs(angle+180) < 10):
            angle = 180+angle
        print(angle)
        M = cv2.getRotationMatrix2D(rect[0], -angle, 1.0)
        rotated = cv2.warpAffine(localImg, M, (Image.shape[1], Image.shape[0]))
        cv2.imshow("rotated",rotated)
        cv2.waitKey(0)
        return rotated

def detectLocation( Image):
        localImg = Image.copy()
        gray = cv2.cvtColor(localImg, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray,3,95,95)
        thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        cv2.imshow("bilateral",gray)
        edge = cv2.Canny(gray, 20, 70, 10)
        
        cv2.imshow("Canny",edge)
        blob=blob_only(thresh)
        contours, _ = cv2.findContours(
            edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cntr in contours:
          cv2.drawContours(edge, [cntr], 0, (255,25,255), -1)
        contours = np.concatenate(contours)
        # cv2.drawContours(localImg, contours, -1, (255, 50, 255), 1)
        boxx=cv2.boundingRect(contours)
        print(boxx)
        rect = cv2.minAreaRect(contours)
        # print(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(localImg, [box], 0, (0, 0, 255), 1)
        cv2.imshow("Comtour", localImg)
        cv2.imshow("edge", edge)
        cv2.imshow("image", Image)
        cv2.imshow("blob", blob)
        cv2.imshow("thresh", thresh)



        cv2.waitKey(0)
        rotated = correctPosition(Image, rect)
        return rect, rotated
# Read image
img = cv2.imread('image548.jpg')


detectLocation(img)
# thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# # apply morphology
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
# morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# # get contours
# contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]

# # draw white contours on black background as mask
# mask = np.zeros((hh,ww), dtype=np.uint8)
# for cntr in contours:
#     cv2.drawContours(mask, [cntr], 0, (255,255,255), -1)

# # get convex hull
# # points = np.column_stack(np.where(thresh.transpose() > 0))
# # hullpts = cv2.convexHull(points)
# # ((centx,centy), (width,height), angle) = cv2.fitEllipse(hullpts)
# # print("center x,y:",centx,centy)
# # print("diameters:",width,height)
# # print("orientation angle:",angle)

# # # draw convex hull on image
# # hull = img.copy()
# # cv2.polylines(hull, [hullpts], True, (0,0,255), 1)

# # # create new circle mask from ellipse 
# # circle = np.zeros((hh,ww), dtype=np.uint8)
# # cx = int(centx)
# # cy = int(centy)
# # radius = (width+height)/4
# # cv2.circle(circle, (cx,cy), int(radius), 255, -1)

# # # erode circle a bit to avoid a white ring
# # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
# # circle = cv2.morphologyEx(circle, cv2.MORPH_ERODE, kernel)

# # # combine inverted morph and circle
# # mask2 = cv2.bitwise_and(255-morph, 255-morph, mask=circle)

# # # apply mask to image
# # result = cv2.bitwise_and(img, img, mask=mask2)

# # # save results
# # cv2.imwrite('pills_thresh2.jpg', thresh)
# # cv2.imwrite('pills_morph2.jpg', morph)
# # cv2.imwrite('pills_mask2.jpg', mask)
# # cv2.imwrite('pills_hull2.jpg', hull)
# # cv2.imwrite('pills_circle.jpg', circle)
# # cv2.imwrite('pills_result2.jpg', result)

# cv2.imshow('thresh', thresh)
# cv2.imshow('morph', morph)
# cv2.imshow('mask', mask)
# # cv2.imshow('hull', hull)
# # cv2.imshow('circle', circle)
# # cv2.imshow('mask2', mask2)
# # cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
from pypylon import pylon
import numpy as np
import cv2
import random
from unwrap import *
wid_all=600
def hconcat_resize(img_list, 
                   interpolation 
                   = cv2.INTER_CUBIC):
      # take minimum hights
    h_min = min(img.shape[0] 
                for img in img_list)
      
    # image resizing 
    im_list_resize = [cv2.resize(img,
                       (int(img.shape[1] * h_min / img.shape[0]),
                        h_min), interpolation
                                 = interpolation) 
                      for img in img_list]
      
    # return final image
    return cv2.hconcat(im_list_resize)



#############################          X and Y correction             ###################################


def Unwrap(image,two_point=None):
    # shape = {"tag": "label", "shape": [{"x": 0.0131332142857142842, "y": 0.05259140625},
    #                                   {"x": 0.50655701811449864, "y": -0.06994404355243445227},
    #                                   {"x": 0.976632539682539681, "y": 0.05219171875},
    #                                   {"x": 0.9766567460317459, "y": 0.9669258984375},
    #                                   {"x": 0.50647501824501454, "y": 1.09952298867391453},#x=0.484,y=0.995
    #                                   {"x": 0.013134920634920626, "y": 0.9669258984375}]}
  
    shape = {"tag": "label", "shape": [{"x": 0.01134, "y": 0.15525},
                                      {"x": 0.5, "y": 0.049210},#1111 
                                      {"x": 0.98866, "y": 0.15525},
                                      {"x": 0.98866, "y": 0.84475},
                                      {"x": 0.5, "y":0.95079},#x=0.484,y=0.995  0.02259140625    #0.99889999298867991453       
                                      {"x": 0.01134, "y": 0.84475}]}
    points=[[ two_point[0][0],   two_point[0][1]],
            [ two_point[1][0],   two_point[1][1]],
            [ 1-two_point[0][0], two_point[0][1]],
            [ 1-two_point[0][0], 1-two_point[0][1]],
            [ two_point[1][0] ,  1-two_point[1][1]],
            [ two_point[0][0] ,  1-two_point[0][1]] ]

    # print(points)
    # points = []
    # for point in shape['shape']:
    #     points.append([point['x'], point['y']])

    imcv = image.copy()
    
    padimage=imcv.copy() #cv2.copyMakeBorder(imcv, 0, 0, 20, 20, cv2.BORDER_CONSTANT, None, value = 0)
    # cv2.imshow("pad",padimage)
    unwrapper = LabelUnwrapper(src_image=padimage, percent_points=points)

    dst_image = unwrapper.unwrap(True)
    for point in unwrapper.points:
        cv2.line(unwrapper.src_image, tuple(point), tuple(point), color=YELLOW_COLOR, thickness=3)


    unwrapper.draw_mesh()
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    h,w=dst_image.shape[:2]
    # x_correction = dst_image[int(h*0.04):h-int(h*0.07),int(w*0.12):w-int(w*0.12)]
    x_correction = dst_image[int(h*0.01):h-int(h*0.02),int(w*0.15):w-int(w*0.15)]

    x_s=x_correction.copy()
    for points in range(0,x_s.shape[0],20):
      sharpen=  cv2.line(x_s,(0,points),(100000,points),(0,255,0),1)
    # cv2.imshow('x_', x_s)


    # cv2.imshow("image_with_mask.png", padimage)
    shape = {"tag": "label", "shape": [{"x": 0.0131332142857142842, "y": 0.020259140625},
                                      {"x": 0.50655701811449864, "y":0.0100627994404355243445227 },
                                      {"x": 0.996632539682539681, "y":0.020259140625},
                                      {"x": 0.9966567460317459, "y": 0.969258984375},
                                      {"x": 0.50647501824501454, "y":.979329952298867391453 },#x=0.484,y=0.995             
                                      {"x": 0.013134920634920626, "y":0.969258984375}]}

    

    points = []
    for point in shape['shape']:
        points.append([point['x'], point['y']])
    imcv = cv2.rotate(x_correction.copy(), cv2.ROTATE_90_CLOCKWISE)
    
    padimage=cv2.copyMakeBorder(imcv, 0, 0, 20, 20, cv2.BORDER_CONSTANT, None, value = (255,255,255))#imcv #
    # cv2.imshow("pad",padimage)
    unwrapper = LabelUnwrapper(src_image=padimage, percent_points=points)

    dst_image = unwrapper.unwrap(True)
    for point in unwrapper.points:
        cv2.line(unwrapper.src_image, tuple(point), tuple(point), color=YELLOW_COLOR, thickness=3)


    unwrapper.draw_mesh()
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    h,w=dst_image.shape[:2]
    # y_correctio = dst_image[int(h*0.02):h-int(h*0.02),int(w*0.1):w-int(w*0.1)]
    y_correctio = dst_image[int(h*0.001):h-int(h*0.001),int(w*0.15):w-int(w*0.15)]
    

  
    # cv2.imshow('purwe', dst_image)


    # cv2.imshow("y_image_with_mask.png", padimage)
    sharpen = cv2.rotate(y_correctio, cv2.ROTATE_90_COUNTERCLOCKWISE)
   
    return sharpen
    # y_x_unr=y_Unwrap(sharpen)
    # return y_x_unr
    #cv2.imwrite("image_with_mask.png", imcv)
    #cv2.imwrite("unwrapped.jpg", dst_image)


   

def removenoise(image,plan_rod=True):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.bilateralFilter(gray, 3, 95, 95)

        # cv2.imshow("edge",edge)
        # gray = cv2.bilateralFilter(gray.T, 21, 95, 95)
        # thresh = cv2.threshold(
        #     edge.T, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        
        if plan_rod:
            gray = cv2.bilateralFilter(gray.T, 21, 95, 95)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            
            height, width = thresh.shape
            
            if height*width - np.count_nonzero(thresh == 255) > (height*width)//2:
                thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
                print("inverse")
        else:
            thresh=cv2.Canny(gray.T,11,50,2)

        thresh=cv2.dilate(thresh,(3,3),iterations=10)

    except:
        thresh = image
    if plan_rod:
        mul=4.5
    else:mul=1.5
    
    Thres = thresh.copy()
    height, width = thresh.shape

    summ = width*255
    for y in range(0, height):
        x_sum = np.sum(thresh[y][:])
        if summ-x_sum >summ/mul: #summ/1.6:
            thresh[y][:] = 0
    # cv2.imshow("threshh",thresh.T)
    for y in range(0, height):
        if np.sum(thresh[y][:]) == 0:

            if y < 200:
                thresh[:y][:] = 0
                break

    for y in range(height-1, 0, -1):
        if np.sum(thresh[y][:]) == 0:
            if y > height-200:
                thresh[y:][:] = 0
                break
    return thresh.T, Thres.T

def vconcat_resize(img_list, interpolation 
                   = cv2.INTER_CUBIC):
      # take minimum width
    w_min = min(img.shape[1] 
                for img in img_list)
      
    # resizing images
    im_list_resize = [cv2.resize(img,
                      (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation = interpolation)
                      for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)
def predict_points(d1):
    d = np.array([20, 25, 30, 35, 40, 40.36, 45, 50,52.66, 55,28.42])
    
    x_1 = np.array([0.31745, 0.27181, 0.26618, 0.18053,
                        0.13490, 0.13161, 0.08926, 0.04362, 0.01934, -0.00202,0.24114])

    y_2 = np.array([0.12069, 0.11109, 0.10106, 0.09055,
                        0.07956, 0.07874, 0.06802, 0.05590, 0.04921, 0.04317,0.10428])
    

    func_up = np.polyfit(d, x_1, 1)
    func1 = np.poly1d(func_up)
    x1 = func1(d1)

    func_low = np.polyfit(d, y_2, 1)
    func2 = np.poly1d(func_low)
    y2 = func2(d1)
    
    return x1,y2

def unrap(frame,plan_rod=False,two_points=None):
    global wid_all
    
    fam=frame.copy()
    new, thresh = removenoise(frame,plan_rod=plan_rod)
    new=cv2.dilate(new,(3,3),iterations=4)
    
    contours, _ = cv2.findContours(
        new, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cntr in contours:
        cv2.drawContours(frame, [cntr], 0, (255, 25, 255), -1)
    
    contours = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(contours)

    cv2.rectangle(frame, (x, y), (x+w, y+frame.shape[1]), (25, 255, 255), 3)
    # cv2.imshow("frammm",frame)
    # print(w-10)
    if w-10!=wid_all:
        dw=wid_all//2-(w-10)//2
        # print(dw)
        final_img=fam[y:y+h,x-dw:x+w-10+dw]
    else:
        final_img=fam[y:y+h,x:x+w-10]
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("12", new)
    # cv2.imshow("final_image",final_img)
    final=Unwrap(final_img,two_point=two_points)
#     cv2.imshow("1", fam)

# # cv2.imshow("11", frame)
    # cv2.waitKey(0)


    # cv2.destroyAllWindows()
    return final,thresh,new
    

from pypylon import pylon
import numpy as np
import cv2
import random
from unwrap import *
import json


s_img=0
y_points=[]
points=[]
def click_event(event, x, y, flags, params):
   global points
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      if len(points)<6:
           points.append([x,y])
 
def click_ypoints(event, x, y, flags, params):
   global y_points
   if event == cv2.EVENT_LBUTTONDOWN:
       print(f'({x},{y})')

       if len(y_points)<6:
              y_points.append([x,y])

class camera_streams:

    def __init__(self):

        self.capture = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.capture.Open()
        self.capture.ExposureTime.SetValue(50000)
        
        self.capture.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_RGB16packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
        self.bgr_img = self.frame = np.ndarray(shape=(self.capture.Height.Value, self.capture.Width.Value, 3),
                                               dtype=np.uint8)

    def get_frame(self):

        grabResult = self.capture.RetrieveResult(
            5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            image = self.converter.Convert(grabResult)
            frame = np.ndarray(buffer=image.GetBuffer(), shape=(image.GetHeight(), image.GetWidth(), 3),
                               dtype=np.uint16)
            self.bgr_img[:, :, 0] = frame[:, :, 2]
            self.bgr_img[:, :, 1] = frame[:, :, 1]
            self.bgr_img[:, :, 2] = frame[:, :, 0]
            # print(self.bgr_img.shape, "----")
            self.frame = self.bgr_img.copy()
        
        else:
            print("Error in frame grabbing.")
        grabResult.Release()

        return   self.frame                                                                                                                                                      
    def __del__(self):
        self.capture.close()


x_upper= -0.000910834404355243445227
x_bottom=1.024152298867391453

y_upper=-0.01327994404355243445227
y_bottom=1.01329952298867391453

xy_corr={'x_corr':[x_upper,x_bottom],'y_corr':[y_upper,y_bottom]}
shape = {"tag": "label", "shape": [{"x": 0.0131332142857142842, "y": 0.01259140625},
                                      {"x": 0.50655701811449864, "y": -0.12101083440435524},#1111 
                                      {"x": 0.966632539682539681, "y": 0.01259140625},
                                      {"x": 0.966632539682539681, "y": 0.9839258984375},
                                      {"x": 0.50655701811449864, "y":1.0999976929886738},#x=0.484,y=0.995  0.02259140625    #0.99889999298867991453       
                                      {"x": 0.0131332142857142842, "y": 0.9839258984375}]}


cor=json.dumps(shape)
with open(f"xy{52.66}.json", "w") as dic:
            dic.write(cor)
  
cv2.namedWindow("select_points",cv2.WINDOW_NORMAL)


# bind the callback function to window
cv2.setMouseCallback('select_points', click_event)

def Unwrap(image,points=None):
    # shape = {"tag": "label", "shape": [{"x": 0.0131332142857142842, "y": 0.05259140625},
    #                                   {"x": 0.50655701811449864, "y": -0.06994404355243445227},
    #                                   {"x": 0.976632539682539681, "y": 0.05219171875},
    #                                   {"x": 0.9766567460317459, "y": 0.9669258984375},
    #                                   {"x": 0.50647501824501454, "y": 1.09952298867391453},#x=0.484,y=0.995
    #                                   {"x": 0.013134920634920626, "y": 0.9669258984375}]}
  
    shape = {"tag": "label", "shape": [{"x": 0.0131332142857142842, "y": 0.01259140625},
                                      {"x": 0.50655701811449864, "y": -0.12101083440435524},#1111 
                                      {"x": 0.966632539682539681, "y": 0.01259140625},
                                      {"x": 0.966632539682539681, "y": 0.9839258984375},
                                      {"x": 0.50655701811449864, "y":1.1123976929886738},#x=0.484,y=0.995  0.02259140625    #0.99889999298867991453       
                                      {"x": 0.0131332142857142842, "y": 0.9839258984375}]}
    shape = {"tag": "label", "shape": [{"x": 0.0131332142857142842, "y": 0.01259140625},
                                      {"x": 0.50655701811449864, "y": -0.12101083440435524},#1111 
                                      {"x": 0.966632539682539681, "y": 0.01259140625},
                                      {"x": 0.966632539682539681, "y": 0.9839258984375},
                                      {"x": 0.50655701811449864, "y":1.1123976929886738},#x=0.484,y=0.995  0.02259140625    #0.99889999298867991453       
                                      {"x": 0.0131332142857142842, "y": 0.9839258984375}]}


    # points = []
    # for point in shape['shape']:
    #     points.append([point['x'], point['y']])

    imcv = image.copy()
    
    padimage=imcv.copy() #cv2.copyMakeBorder(imcv, 0, 0, 20, 20, cv2.BORDER_CONSTANT, None, value = 0)
    # cv2.imshow("pad",padimage)
    unwrapper = LabelUnwrapper(src_image=padimage, pixel_points=points)

    dst_image = unwrapper.unwrap(True)
    for point in unwrapper.points:
        cv2.line(unwrapper.src_image, tuple(point), tuple(point), color=YELLOW_COLOR, thickness=3)


    unwrapper.draw_mesh()
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    h,w=dst_image.shape[:2]
    x_correction = dst_image[int(h*0.04):h-int(h*0.07),int(w*0.12):w-int(w*0.12)]
    # x_correction = dst_image[int(h*0.01):h-int(h*0.07),int(w*0.15):w-int(w*0.15)]

    x_s=x_correction.copy()
    for points in range(0,x_s.shape[0],20):
      sharpen=  cv2.line(x_s,(0,points),(100000,points),(0,255,0),1)
    cv2.namedWindow("x_",cv2.WINDOW_NORMAL)
    
    cv2.imshow('x_', x_s)

    cv2.namedWindow("image_with_mask.png",cv2.WINDOW_NORMAL)

    cv2.imshow("image_with_mask.png", padimage)
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
    
    padimage=imcv #cv2.copyMakeBorder(imcv, 0, 0, 20, 20, cv2.BORDER_CONSTANT, None, value = 0)
    # cv2.imshow("pad",padimage)
    unwrapper = LabelUnwrapper(src_image=padimage, percent_points=points)

    dst_image = unwrapper.unwrap(True)
    for point in unwrapper.points:
        cv2.line(unwrapper.src_image, tuple(point), tuple(point), color=YELLOW_COLOR, thickness=3)


    unwrapper.draw_mesh()
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    h,w=dst_image.shape[:2]
    y_correctio = dst_image[int(h*0.02):h-int(h*0.02),int(w*0.1):w-int(w*0.1)]
    

  
    # cv2.imshow('_sharpen', sharpen)

    cv2.namedWindow("y_image_with_mask.png",cv2.WINDOW_NORMAL)

    cv2.imshow("y_image_with_mask.png", padimage)
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
        
        height, width = gray.shape
        if plan_rod:
            gray = cv2.bilateralFilter(gray.T, 21, 95, 95)
            thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            if height*width - np.count_nonzero(thresh == 255) > (height*width)//2:
                thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        else:
            thresh=cv2.Canny(gray.T,11,50,2)

        thresh=cv2.dilate(thresh,(3,3),iterations=10)

    except:
        thresh = image.T
    if plan_rod:
        mul=4.5
    else:mul=1.5
    Thres = thresh.copy()
    height, width = thresh.shape

    summ = width*255
    for y in range(0, height):
        x_sum = np.sum(thresh[y][:])

        if summ-x_sum > summ/mul:
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


cap = camera_streams()

x_m1=0
y_m1=0
x_m2=0
y_m2=0
while True:
    frame = cap.get_frame()
    fam=frame.copy()
    new, thresh = removenoise(frame,plan_rod=False)
    new=cv2.dilate(new,(3,3),iterations=4)
    
    contours, _ = cv2.findContours(
        new, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cntr in contours:
        cv2.drawContours(frame, [cntr], 0, (255, 25, 255), -1)
    
    contours = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(contours)
    # box=cv2.selectROI(frame)
    # print(box)
    # x,y,w,h=(66, 2, 570, 494)

    cv2.rectangle(frame, (x, y), (x+w, y+frame.shape[1]), (25, 255, 255), 3)
    final_img=fam[y:y+h,x:x+w]
    xy_corr = open('xy52.66.json')
    s_img=fam.copy()
    height,width=s_img.shape[:2]
    cv2.line(s_img, (width // 2, 0), (width // 2, height), (255, 0, 255), 2)
    cv2.line(s_img, (0, height // 2), (width, height // 2), (255, 0, 255), 2)
    for point in points:
        cv2.circle(s_img, (point[0],point[1]), 2, (0,255,255), -1)

    # for p in range(0,s_img.shape[1],30):
    #   sharpen=  cv2.line(s_img,(p,0),(p,100000),(0,255,0),1)
    k = cv2.waitKey(1)







    cv2.imshow("select_points",s_img)
    # points=[[54, 89], [340, 40], [632, 86], [632, 413], [338, 461], [55, 414]]
    # points=[[53, 70], [340, 15], [627, 70], [627, 430], [340, 485], [53, 430]]  
    # points=[[120,70],[340,30]]
    if len(points)==2:
        

        point3=[width-points[0][0],points[0][1]]
        point4=[point3[0],height-point3[1]]
        point5=[points[1][0],height-points[1][1]]
        point6=[points[0][0],height-points[0][1]]
        points.append(point3)
        points.append(point4)
        points.append(point5)
        points.append(point6)

    if len(points)==6 :
        print(points)

        xy_corr = json.load(xy_corr)
        y_x_unr=Unwrap(fam,points)
        cv2.namedWindow("final",cv2.WINDOW_NORMAL)

        cv2.imshow('final', y_x_unr)

    # except:pass
    cv2.namedWindow("1",cv2.WINDOW_NORMAL)
    cv2.namedWindow("11",cv2.WINDOW_NORMAL)

    cv2.imshow("1", fam)
    
    cv2.imshow("11", frame)
    if k == ord("q"):
        break
    if k == ord("s"):
        cv2.imwrite(f"image{random.randint(0,999)}.jpg", fam)

cv2.destroyAllWindows()

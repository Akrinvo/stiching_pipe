#!/usr/bin/env python3.10
import os

try:import cv2
except:
    os.system("pip3 install opencv-python")
    import cv2
import numpy as np
try:from pypylon import pylon
except:
    os.system("pip3 install pypylon")
    from pypylon import pylon

try: import matplotlib.pyplot as plt
except:
    os.system("pip3 install matplotlib")
    import matplotlib.pyplot as plt

    


import numpy as np
import random
import platform

import time

import json


from tkinter import *
from tkinter import ttk , messagebox
import json
from PIL import ImageTk, Image, ImageGrab

import multiprocessing
import threading
from threading import Thread, Lock
from tkinter.messagebox import askyesno
final_img=0
good_img,bad_img = 0,0
dia_lbl = None
wid_all=600

BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)


class Line(object):
    def __init__(self, point1, point2):
        """
        For line formula y(x) = k * x + b, calc k and b params
        If the line is vertical, set "vertical" attr to True and save "x" position of the line
        """
        self.point1 = point1
        self.point2 = point2
        self.vertical = False
        self.fixed_x = None
        self.k = None
        self.b = None

        # cached angle props
        self.angle = None
        self.angle_cos = None
        self.angle_sin = None

        self.set_line_props(point1, point2)

    def is_vertical(self):
        return self.vertical

    def set_line_props(self, point1, point2):
        if point2[0] - point1[0]:
            self.k = float(point2[1] - point1[1]) / (point2[0] - point1[0])
            self.b = point2[1] - self.k * point2[0]

            k_normal = - 1 / self.k
        else:
            self.vertical = True
            self.fixed_x = point2[0]

            k_normal = 0

        self.angle = np.arctan(k_normal)
        self.angle_cos = np.cos(self.angle)
        self.angle_sin = np.sin(self.angle)

    def get_x(self, y):
        if self.is_vertical():
            return self.fixed_x
        else:
            return int(round(float(y - self.b) / self.k))

    def get_y(self, x):
        return self.k * x + self.b


class LabelUnwrapper(object):
    COL_COUNT = 20
    ROW_COUNT = 30

    def __init__(self, src_image=None, pixel_points=None, percent_points=None):
        """
        Point lists are lists of 6 points - [A, B, C, D, E, F]

        :param pixel_points: List[Tuple] Points, whose coordinates specified as pixels
        :param percent_points: List[Tuple] Points, whose coordinates specified as fraction of image width/height

        In both cases points represent figure below:

        |        |                  |        |
        |    B   |                  A        C
        | /    \ |                  | \    / |
        A        C                  |   B    |
        |        |                  |        |
        |        |       OR         |        |
        |        |                  |        |
        F        D                  F        D
        | \    / |                  | \    / |
        |   E    |                  |   E    |
        |        |                  |        |

        So, A-B-C-D-E-F-A polygon represent raw wine label on bottle

        """
        self.src_image = src_image
        self.width = self.src_image.shape[1]
        self.height = src_image.shape[0]

        self.dst_image = None
        self.points = pixel_points
        self.percent_points = percent_points

        self.point_a = None  # top left
        self.point_b = None  # top center
        self.point_c = None  # top right
        self.point_d = None  # bottom right
        self.point_e = None  # bottom center
        self.point_f = None  # bottom left
        self.map=0
        self.center_line = None
        self.load_points()

    def load_points(self):
        if self.points is None:
            points = []
            for point in self.percent_points:
                x = int(point[0] * self.width)
                y = int(point[1] * self.height)
                points.append((x, y))

            self.points = points

        self.points = np.array(self.points)
        (self.point_a, self.point_b, self.point_c,
         self.point_d, self.point_e, self.point_f) = self.points

        center_top = (self.point_a + self.point_c) / 2
        center_bottom = (self.point_d + self.point_f) / 2
        #print(self.points)
        self.center_line = Line(center_bottom, center_top)
        if not len(self.points) == 6:
            raise ValueError("Points should be an array of 6 elements")

    def unwrap(self, interpolate=False):
        source_map = self.calc_source_map()
        if interpolate:
            self.unwrap_label_interpolation(source_map)
        else:
            self.unwrap_label_perspective(source_map)
        return self.dst_image

    def calc_dest_map(self):
        width, height = self.get_label_size()

        dx = float(width) / (self.COL_COUNT - 1)
        dy = float(height) / (self.ROW_COUNT - 1)

        rows = []
        for row_index in range(self.ROW_COUNT):
            row = []
            for col_index in range(self.COL_COUNT):
                row.append([int(dx * col_index),
                            int(dy * row_index)])

            rows.append(row)
        return np.array(rows)

    def unwrap_label_interpolation(self, source_map):
        """
        Unwrap label using interpolation - more accurate method in terms of quality
        """
        from scipy.interpolate import griddata

        width, height = self.get_label_size()

        dest_map = self.calc_dest_map()
        self.map = dest_map
        grid_x, grid_y = np.mgrid[0:width - 1:width * 1j, 0:height - 1:height * 1j]
        #print(len(grid_x),len(grid_x[0]))
        #print(len(grid_y),len(grid_y[0]))
        destination = dest_map.reshape(dest_map.size // 2, 2)
        source = source_map.reshape(source_map.size // 2, 2)
        #print(source_map)
        #print(source)
        grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
        map_x = np.append([], [ar[:, 0] for ar in grid_z]).reshape(width, height)
        map_y = np.append([], [ar[:, 1] for ar in grid_z]).reshape(width, height)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')
        warped = cv2.remap(self.src_image, map_x_32, map_y_32, cv2.INTER_CUBIC)
        self.dst_image = cv2.transpose(warped)

    def unwrap_label_perspective(self, source_map):
        """
        Unwrap label using transform, unlike unwrap_label_interpolation doesn't require scipy
        """
        width, height = self.get_label_size()
        self.dst_image = np.zeros((height, width, 3), np.uint8)

        dx = float(width) / (self.COL_COUNT - 1)
        dy = float(height) / (self.ROW_COUNT - 1)

        dx_int = int(np.ceil(dx))
        dy_int = int(np.ceil(dy))

        for row_index in range(self.ROW_COUNT - 1):
            for col_index in range(self.COL_COUNT - 1):
                src_cell = (source_map[row_index][col_index],
                            source_map[row_index][col_index + 1],
                            source_map[row_index + 1][col_index],
                            source_map[row_index + 1][col_index + 1])

                dst_cell = np.int32([[0, 0], [dx, 0], [0, dy], [dx, dy]])

                M = cv2.getPerspectiveTransform(np.float32(src_cell), np.float32(dst_cell))
                dst = cv2.warpPerspective(self.src_image, M, (dx_int, dy_int))
                x_offset = int(dx * col_index)
                y_offset = int(dy * row_index)

                self.dst_image[y_offset:y_offset + dy_int,
                               x_offset:x_offset + dx_int] = dst

    def get_roi_rect(self, points):
        max_x = min_x = points[0][0]
        max_y = min_y = points[0][1]
        for point in points:
            x, y = point
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y

        return np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])

    def get_roi(self, image, points):
        rect = self.get_roi_rect(points)
        return image[np.floor(rect[0][1]):np.ceil(rect[2][1]),
                     np.floor(rect[0][0]):np.ceil(rect[1][0])]

    def calc_source_map(self):
        top_points = self.calc_ellipse_points(self.point_a, self.point_b, self.point_c,
                                              self.COL_COUNT)
        bottom_points = self.calc_ellipse_points(self.point_f, self.point_e, self.point_d,
                                                 self.COL_COUNT)

        rows = []
        for row_index in range(self.ROW_COUNT):
            row = []
            for col_index in range(self.COL_COUNT):
                top_point = top_points[col_index]
                bottom_point = bottom_points[col_index]

                delta = (top_point - bottom_point) / float(self.ROW_COUNT - 1)
                #print(delta, top_point,bottom_point)
                point = top_point - delta * row_index
                row.append(point)
            rows.append(row)
        return np.array(rows)

    def draw_mesh(self, color=RED_COLOR, thickness=2):
        mesh = self.calc_source_map()
        for row in mesh:
            for x, y in row:
                point = (int(round(x)), int(round(y)))
                cv2.line(self.src_image, point, point, color=color, thickness=thickness)

    def draw_poly_mask(self, color=WHITE_COLOR):
        cv2.polylines(self.src_image, np.int32([self.points]), 1, color)


    def draw_mask(self, color=WHITE_COLOR, thickness=1, img=None):
        """
        Draw mask, if image not specified - draw to source image
        """
        if img is None:
            img = self.src_image

        cv2.line(img, tuple(self.point_f.tolist()), tuple(self.point_a.tolist()), color, thickness)
        cv2.line(img, tuple(self.point_c.tolist()), tuple(self.point_d.tolist()), color, thickness)

        self.draw_ellipse(img, self.point_a, self.point_b, self.point_c, color, thickness)
        self.draw_ellipse(img, self.point_d, self.point_e, self.point_f, color, thickness)

    def get_label_contour(self, color=WHITE_COLOR, thickness=1):
        mask = np.zeros(self.src_image.shape)
        self.draw_mask(color, thickness, mask)
        return mask

    def get_label_mask(self):
        """
        Generate mask of the label, fully covering it
        """
        mask = np.zeros(self.src_image.shape)
        pts = np.array([[self.point_a, self.point_c, self.point_d, self.point_f]])
        cv2.fillPoly(mask, pts, WHITE_COLOR)
        self.draw_filled_ellipse(mask, self.point_a, self.point_b, self.point_c, True)
        self.draw_filled_ellipse(mask, self.point_f, self.point_e, self.point_d, False)
        return mask

    def draw_ellipse(self, img, left, top, right, color=WHITE_COLOR, thickness=1):
        """
        Draw ellipse using opencv function
        """
        is_arc, center_point, axis, angle = self.get_ellipse_params(left, top, right)

        if is_arc:
            start_angle, end_angle = 0, 180
        else:
            start_angle, end_angle = 180, 360

        cv2.ellipse(img, center_point, axis, angle, start_angle, end_angle, color, thickness)

    def draw_filled_ellipse(self, img, left, top, right, is_top=False):
        is_arc, center_point, axis, angle = self.get_ellipse_params(left, top, right)

        if is_arc ^ is_top:
            color = WHITE_COLOR
        else:
            color = BLACK_COLOR

        cv2.ellipse(img, center_point, axis, angle, 0, 360, color=color, thickness=-1)

    def get_ellipse_params(self, left, top, right):
        center = (left + right) / 2
        center_point = tuple(map(lambda x: int(np.round(x)), center.tolist()))

        axis = (int(np.linalg.norm(left - right) / 2), int(np.linalg.norm(center - top)))

        x, y = left - right
        angle = np.arctan(float(y) / x) * 57.296

        is_arc = False
        if (top - center)[1] > 0:
            is_arc = True

        return is_arc, center_point, axis, angle

    def calc_ellipse_points(self, left, top, right, points_count):
        center = (left + right) / 2

        # get ellipse axis
        a = np.linalg.norm(left - right) / 2
        b = np.linalg.norm(center - top)

        # get start and end angles
        if (top - center)[1] > 0:
            delta = np.pi / (points_count - 1)

        else:
            delta = - np.pi / (points_count - 1)

        cos_rot = (right - center)[0] / a
        sin_rot = (right - center)[1] / a

        points = []
        for i in range(points_count):
            phi = i * delta
            dx, dy = self.get_ellipse_point(a, b, phi)

            x = round(center[0] + dx * cos_rot - dy * sin_rot)
            y = round(center[1] + dx * sin_rot + dy * cos_rot)

            points.append([x, y])

        points.reverse()
        return np.array(points)

    def get_ellipse_point(self, a, b, phi):
        """
        Get ellipse radius in polar coordinates
        """
        return a * np.cos(phi), b * np.sin(phi)

    def get_label_size(self):
        top_left = self.point_a
        top_right = self.point_c
        bottom_right = self.point_d
        bottom_left = self.point_f

        width1 = np.linalg.norm(top_left - top_right)
        width2 = np.linalg.norm(bottom_left - bottom_right)
        avg_width = int((width1 + width2) * np.pi / 4)

        height1 = np.linalg.norm(top_left - bottom_left)
        height2 = np.linalg.norm(top_right - bottom_right)
        avg_height = int((height1 + height2) / 2)
        return avg_width, avg_height


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

  

    points=[[ two_point[0][0],   two_point[0][1]],
            [ two_point[1][0],   two_point[1][1]],
            [ 1-two_point[0][0], two_point[0][1]],
            [ 1-two_point[0][0], 1-two_point[0][1]],
            [ two_point[1][0] ,  1-two_point[1][1]],
            [ two_point[0][0] ,  1-two_point[0][1]] ]



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

    sharpen = cv2.rotate(y_correctio, cv2.ROTATE_90_COUNTERCLOCKWISE)
   
    return sharpen



   

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

    if w-10!=wid_all:
        dw=wid_all//2-(w-10)//2
        # print(dw)
        final_img=fam[y:y+h,x-dw:x+w-10+dw]
    else:
        final_img=fam[y:y+h,x:x+w-10]

    final=Unwrap(final_img,two_point=two_points)

    return final,thresh,new
    






def camera_opner():
    scale_percent=30
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
   
    
    

    imagelist=[]
    for device,i in zip(devices,range(len(devices))):

        camera = pylon.InstantCamera()
        
        camera.Attach(tl_factory.CreateFirstDevice(device))
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        camera.Open()
        camera.StartGrabbing(2)
        grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
        if grab.GrabSucceeded():
            image = converter.Convert(grab)
            im = image.GetArray()

            width = 680
            height = 500
            dim = (width, height)
            

            im = cv2.resize(im, dim, cv2.INTER_AREA)
            imagelist.append(im)
    if len(imagelist)==4:
        return imagelist
        


def final_unwrap(imagelist,two_points):

        
    f_img=[]
    threshs=[]
    news=[]
    crp=0.004470
    plan_rod=False
    r_h,r_w=200,320


    a=time.time()
    final,thresh,new=unrap(imagelist[0].copy(),plan_rod=plan_rod,two_points=two_points)
    final=cv2.resize(final,(r_w,r_h))
    f_img.append(final)

  

    final1,thresh1,new1=unrap(imagelist[1].copy(),plan_rod=plan_rod,two_points=two_points)
    final1=cv2.resize(final1,(r_w,r_h))

    f_img.append(final1)

    final2,thresh2,new2=unrap(imagelist[2].copy(),plan_rod=plan_rod,two_points=two_points)
    final2=cv2.resize(final2,(r_w,r_h))

    f_img.append(final2)

    final3,thresh3,new3=unrap(imagelist[3].copy(),plan_rod=plan_rod,two_points=two_points)


    final3=cv2.resize(final3,(r_w,r_h))

    f_img.append(final3)
    for i,j in zip(f_img,range(4)):
        # cv2.imshow(f'image2{j}', imagelist[j])
    

        cv2.imwrite(f'final2{j}.jpg', i)
    print(time.time()-a)
    
    return f_img
    




def calibratedimage(ori,images):

    img1,img2,img3,img4=images
    bg= cv2.copyMakeBorder(np.zeros((ori["bg_shape"][1],ori["bg_shape"][2],3),np.uint8), ori["bg_shape"][0], ori["bg_shape"][0], ori["bg_shape"][0], ori["bg_shape"][0], cv2.BORDER_CONSTANT, None, value = 0)

    bg[ori["img1"][0] :ori["img1"][1],ori["img1"][2]:ori["img1"][3]]=img1
    bg[ori["img2"][0] :ori["img2"][1],ori["img2"][2]:ori["img2"][3]]=img2
    bg[ori["img3"][0] :ori["img3"][1],ori["img3"][2]:ori["img3"][3]]=img3
    bg[ori["img4"][0] :ori["img4"][1],ori["img4"][2]:ori["img4"][3]]=img4


    final_img=bg[max(ori["img1"][0],ori["img2"][0],ori["img3"][0],ori["img4"][0]):
                min(ori["img1"][1],ori["img2"][1],ori["img3"][1],ori["img4"][1]),
                ori["img1"][2]:ori["img4"][3]]

    return final_img


images=[]
Shape=[]
pad=80
 
n=1
ver=0
hori=0
 
n2=1
ver2=0
hori2=0

n3=1
ver3=0
hori3=0

n4=1
ver4=0
hori4=0


img_no=1
images.append(cv2.imread("final20.jpg"))
images.append(cv2.imread("final21.jpg"))
images.append(cv2.imread("final22.jpg"))
images.append(cv2.imread("final23.jpg"))

Shape.append(images[0].shape[:2])
Shape.append(images[1].shape[:2])
Shape.append(images[2].shape[:2])
Shape.append(images[3].shape[:2])

print(Shape)
H,W=0,0

for shape in Shape:
    
    sh,sw =shape
    H=sh
    W+=sw
bg_img= cv2.copyMakeBorder(np.zeros((H,W,3),np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, value = 0)

pre=2
f_=0
ori={"bg_shape":[pad,H,W]}
def clear(images):
    img1,img2,img3,img4=images

    global ori,ver,hori,ver2,hori2,ver3,hori3,ver4,hori4
    bg= cv2.copyMakeBorder(np.zeros((H,W,3),np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, value = 0)
    if len(ori)>=2:
        bg[ori["img1"][0] :ori["img1"][1],ori["img1"][2]:ori["img1"][3]]=img1
    if len(ori)>=3:
        bg[ori["img2"][0] :ori["img2"][1],ori["img2"][2]:ori["img2"][3]]=img2
    if len(ori)>=4:
         bg[ori["img3"][0] :ori["img3"][1],ori["img3"][2]:ori["img3"][3]]=img3
    if len(ori)==5:
         bg[ori["img4"][0] :ori["img4"][1],ori["img4"][2]:ori["img4"][3]]=img4
    return bg
def mover_image(images,dia):
    global  ori,ver,hori,ver2,hori2,ver3,hori3,ver4,hori4,pad,bg_img,img_no
   
    while True:
         
        img1,img2,img3,img4=images
        h1,w1=img1.shape[:2]
        h2,w2=img2.shape[:2]
        h3,w3=img3.shape[:2]
        h4,w4=img4.shape[:2]
        bg=bg_img.copy()

        k=cv2.waitKey(1)
        # print(k)
        if k==ord('1'):
            img_no=1
        if k==ord('2'):
            img_no=2
        if k==ord('3'):
            img_no=3
        if k==ord('4'):
            img_no=4

        if img_no==1:
            # bg_img[0:h1,0:w1]=img1

            bg_img[pad+ver:h1+ver+pad,pad+hori:w1+hori+pad]=img1
            
            ###  vertical move
            if k==ord("w"):
                ver-=pre
            if k==ord('s'):
                ver+=pre 


        ###  horizontal move
            if k==ord("a"):
                hori-=pre
            if k==ord('d'):
                hori+=pre


            ori["img1"]=[pad+ver,h1+ver+pad,pad+hori,w1+hori+pad]
            bg_img=clear(images)
            

            


        if img_no==2:
            bg_img[pad+ver2:h1+ver2+pad,pad+hori2+w1:w1+w2+hori2+pad]=img2

                ###  vertical move
            if k==ord("w"):
                ver2-=pre
            if k==ord('s'):
                ver2+=pre


        ###  horizontal move
            if k==ord("a"):
                hori2-=pre
            if k==ord('d'):
                hori2+=pre
            

            ori["img2"]=[pad+ver2,h1+ver2+pad,pad+hori2+w1,w1+w2+hori2+pad]
            bg_img=clear(images)
            
        if img_no==3:
            # bg_img[0:h3,w1+w2:w1+w2+w3]=img3
            bg_img[pad+ver3:h1+ver3+pad,pad+hori3+w1+w2:w1+w2+w3+hori3+pad]=img3

                ###  vertical move
            if k==ord("w"):
                ver3-=pre
            if k==ord('s'):
                ver3+=pre


        ###  horizontal move
            if k==ord("a"):
                hori3-=pre
            if k==ord('d'):
                hori3+=pre

            ori["img3"]=[pad+ver3,h1+ver3+pad,pad+hori3+w1+w2,w1+w2+w3+hori3+pad]
            bg_img=clear(images)
            
        if img_no==4:
            # bg_img[0:h4,w1+w2+w3:w1+w2+w3+w4]=img4
            bg_img[pad+ver4:h1+ver4+pad,pad+hori4+w1+w2+w3:w1+w2+w3+w4+hori4+pad]=img4

                ###  vertical move
            if k==ord("w"):
                ver4-=pre
            if k==ord('s'):
                ver4+=pre


        ###  horizontal move
            if k==ord("a"):
                hori4-=pre
            if k==ord('d'):
                hori4+=pre

            ori["img4"]=[pad+ver4,h1+ver4+pad,pad+hori4+w1+w2+w3,w1+w2+w3+w4+hori4+pad]
            bg_img=clear(images)

        if k==ord('f'):
            cor=json.dumps(ori)
            with open(f"stich{dia}.json", "w") as dic:
                dic.write(cor)

            out=calibratedimage(ori,images)

            cv2.imwrite("output.jpg",out)
            break
        
        # print(ori)
        k=0
        cv2.namedWindow("calibrate",cv2.WINDOW_NORMAL)

        cv2.imshow("calibrate",bg_img)
        
        
    cv2.destroyAllWindows()

def calibrated_image(dia,images):
    with open(f'stich{dia}.json', 'r') as openfile:
        ori= json.load(openfile)
    img1,img2,img3,img4=images
    bg= cv2.copyMakeBorder(np.zeros((ori["bg_shape"][1],ori["bg_shape"][2],3),np.uint8), ori["bg_shape"][0], ori["bg_shape"][0], ori["bg_shape"][0], ori["bg_shape"][0], cv2.BORDER_CONSTANT, None, value = 0)

    bg[ori["img1"][0] :ori["img1"][1],ori["img1"][2]:ori["img1"][3]]=img1
    bg[ori["img2"][0] :ori["img2"][1],ori["img2"][2]:ori["img2"][3]]=img2
    bg[ori["img3"][0] :ori["img3"][1],ori["img3"][2]:ori["img3"][3]]=img3
    bg[ori["img4"][0] :ori["img4"][1],ori["img4"][2]:ori["img4"][3]]=img4


    final_img=bg[max(ori["img1"][0],ori["img2"][0],ori["img3"][0],ori["img4"][0]):
                min(ori["img1"][1],ori["img2"][1],ori["img3"][1],ori["img4"][1]),
                ori["img1"][2]:ori["img4"][3]]
    return final_img

print(str(platform.uname()[0:]))

if str(platform.uname()[0:])=="('Linux', 'oem', '5.4.0-152-generic', '#169~18.04.1-Ubuntu SMP Wed Jun 7 22:22:24 UTC 2023', 'x86_64', 'x86_64')":



    final_img=0
    good_img,bad_img = 0,0
    dia_lbl = None
    # lock = threading.Lock()
    # lock =Lock()
    image_label8 = None

    # label2 =None
    # lbl1 = None
    # label1 = None

    # cap = cv2.VideoCapture(0)



    def preview_camera():
        print("working.......")
        while True:
            imagelist = camera_opner()
            partition=np.ones((imagelist[0].shape[0],20,3),np.uint8)*255
            concat_image=hconcat_resize([imagelist[0],partition,imagelist[1],partition,imagelist[2],partition,imagelist[3]])
            
            cv2.namedWindow("Press-Q for Exit",cv2.WINDOW_NORMAL)

            cv2.imshow("Press-Q for Exit",concat_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()  
    


    def load_image(path, width, height):
        image = Image.open(path)
        image = image.resize((width,height))  # Resize the image if needed
        image_tk = ImageTk.PhotoImage(image)
        return image_tk



    def logo(x):
        heading = Label(x, text="Compaq work", font=("times new roman", 30, "bold"), bg=bg_color,
                    fg="black", relief=GROOVE)
        image_tk = load_image("logo.jpeg",150,50)
        image_label2 = Label(heading, image=image_tk)
        image_label2.image = image_tk  # Store a reference to the image to prevent garbage collection
        image_label2.pack(side=LEFT)
        heading.pack(fill=X)

    calib_flage = False

    #
    popup = None

    def popup_window():
        popup = Toplevel(root)
        popup.title("INTERFACE")
        popup.overrideredirect(True)
        popup.geometry("400x200")

        label3 = Label(popup, text="working...........",font=("times new roman", 25, "bold"),fg="green")
        label3.pack(padx=20,pady=20) 

        open_another_button2 = Button(popup, text="Exit",font=("times new roman",15, "bold"),fg="white", bg="red", height=1, width=10,command=lambda: popup.destroy())
        open_another_button2.place(x=200, y=150)  

        return popup


    def destroy_popup():
        global popup
        if popup:
            popup.destroy()
            print("destroy")



    # directory_path(1)

    def last_window():
        # global popup
        global calib_flage
        if calib_flage:
            pass
        else:
            calib_flage = True
            last_window = Toplevel(root)
            last_window.title("INTERFACE")
            # last_window.wm_attributes('-fullscreen', 'True')
            # last_window.overrideredirect(True)
            last_window.geometry("1500x800")
            # popup_window()

            logo(last_window)


            def read_options(configuration_root_directory = None):
                filnames = os.listdir(configuration_root_directory)
                dia_list = [filname.split(".")[0][2:]+'.'+filname.split(".")[1] for filname in filnames]
                return dia_list
            
            def label2_work():
                global label2
                label2 = Label(last_window,text="working.....",font=("times new roman", 15, "bold"),padx=10,pady=10)
                label2.pack()

            def wait():
                open_another_button.config(text="please wait....")
                print("hellio")
            # thread1 = threading.Thread(target=update_frame1,args=(10,lock))
            
            

            def update_frame1():
            
                # thread1.join()
            
                global final_img,popup
                print("update frame working")
                
                # popup = popup_window()
                # queue.put("hello")
                # lock.acquire()
                
                dia=clicked.get()
                # ret1, frame1 = cap1.read()
                imagelist=camera_opner()
                partition=np.ones((imagelist[0].shape[0],20,3),np.uint8)*255
                image=hconcat_resize([imagelist[0],partition,imagelist[1],partition,imagelist[2],partition,imagelist[3]])
                # dia=52.66
                with open(f"setting/xy{dia}.json","r") as openfile:
                    points=json.load(openfile)
                f_img=final_unwrap(imagelist,two_points=points)
                final_img=calibrated_image(dia,f_img)
                # cv2.imshow("img",final_img)
                # cv2.waitKey(30)
                print("xyz")
                parth=np.ones((20,image.shape[1],3),np.uint8)*255
                frame=vconcat_resize([image,parth,final_img])
                # if ret1:
                frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame1 = cv2.resize(frame1, (1000, 400))
                img1 = Image.fromarray(frame1)
                img_tk = ImageTk.PhotoImage(image=img1)
                # if label1:
                #     label1.destroy()
                label1.config(image=img_tk)
            
                label1.img1 = img_tk
                # label1.after(10, update_frame1)
                label1.pack()
                # time.sleep(2)
                # destroy_popup()
                # popup.destroy()
                # thread1.join()
                print("popup done")
                # queue.put("work done")
                # thread1.join()
                # lock.release()
                # thread1.join()

            def preview_plt():
                global final_img
                img = cv2.cvtColor(final_img,cv2.COLOR_BGR2RGB)
                plt.figure(num="Image Preview")
                plt.imshow(img)
                plt.show()
                    

            def work():
                print("work start")

                update_frame1()
                print("work done")
            # p1 = multiprocessing.Process(target=update_frame1,args=(2,))

            def directory_path(dia):
                print("1")
                base_dir = "SmartViz_Image_Manual_Inspection"
                print("@")
                counter = 1
                dir_nm = f"dia_{dia}_{counter}"
                print("w")

                dir_path = os.path.join(base_dir,dir_nm)
                
                while os.path.exists(dir_path):
                    print("#")
                    counter += 1
                    dir_nm = f"dia_{dia}_{counter}"
                    print("$")
                    dir_path = os.path.join(base_dir,dir_nm)


                os.mkdir(dir_path) 
                print("d")   
                return dir_path

            def wait():
                open_another_button.config(text="please wait....")
                print("hellio")
            # thread1 = threading.Thread(target=update_frame1,args=(10,lock))
            def selected(event):
                global popup, dia
            

                
                # thread1 = threading.Thread(target=update_frame1)
                dia = clicked.get()
                dia_lbl.config(text=f"dia_{dia}")
                dia_lbl.pack()
                print("123")

                
                # print("work before....")
                print("popup start")

                # popup = popup_window()
                # p2 = multiprocessing.Process(target=update_frame1)
                # directory()
                # p2.run()
                print("update completed")
                # directory_path(dia)
                

                
                update_frame1()
                # thread1.start()
                # queue = multiprocessing.Queue()
                # print("processing start...")

                
            
                
                # p1.join()
                # update_frame1()
                
                print("done....")
                # update_frame1()
                print("process")
                # new_w.destroy()
                
                    
        
            

            def save_good():
                global final_img,good_img,dia
                print(dia)
                dir_path = directory_path(dia)
                
                print("123")
                good_img += 1
                print(good_img)
                good_dir=  os.path.join(dir_path,"OK")  
                # last = os.path.join(good_dir,final_img)  
                isExist = os.path.exists(good_dir)
                print(isExist)
                if not isExist:
                    os.makedirs(good_dir)
                
                cv2.imwrite(f"{good_dir}/{good_img}.jpg",final_img) 

            def save_bad():
                global final_img,bad_img,dia
                dir_path = directory_path(dia)
                bad_dir = os.path.join(dir_path,"NG")
                bad_img += 1
                # filname="bad/"    
                isExist = os.path.exists(bad_dir)
                if not isExist:
                    os.makedirs(bad_dir)
                cv2.imwrite(f"{bad_dir}/{bad_img}.jpg",final_img)


            options = read_options(configuration_root_directory="setting")
            clicked = StringVar()
            clicked.set(options[0])

            drop = OptionMenu(last_window, clicked, *options, command=selected)

            drop.pack(pady=5)

            diameter = LabelFrame(last_window,text="Serial No : ",font=("times new roman", 15, "bold"))
            dia_lbl = Label(diameter,text=f"dia",font=("times new roman", 10, "bold"), padx=10, pady=10)
            # dia_lbl.pack()
            diameter.place(x=80, y=70)

            lbl1 = LabelFrame(last_window, text="Results",font=("times new roman", 15, "bold"), padx=10, pady=10)
            # lbl1.pack(side=LEFT)
            lbl1.place(x=200, y=150)
            label1 = Label(lbl1)
            # label1.pack()

            def destroy_last_window():
                global calib_flage
                calib_flage = False
                last_window.destroy()

            def on_close():
                global calib_flage
                calib_flage = False
                print(calib_flage)
                last_window.destroy()

            open_another_button = Button(last_window, text="Next",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=15,command=work)
            open_another_button.place(x=500,y=650)

            open_another_button4 = Button(last_window, text="Preview",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=preview_plt)
            open_another_button4.place(x=800,y=650)

            open_another_button1 = Button(last_window, text="Exit",font=("times new roman",15, "bold"),fg="white", bg="red", height=1, width=10,command=destroy_last_window)
            open_another_button1.place(x=1320,y=750)

            open_another_button2 = Button(last_window, text="OK",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=10,command=lambda:save_good())
            open_another_button2.place(x=10, y=750)
            
            open_another_button3 = Button(last_window, text="NG",font=("times new roman",15, "bold"),fg="white", bg="red", height=1, width=10,command=lambda:save_bad())
            open_another_button3.place(x=200, y=750)

            last_window.protocol("Another Window",on_close)
        

    def open_new_window():
        
        global calib_flage
        if calib_flage:
            pass
        else:   
            calib_flage = True
            global input_value
            new_window = Toplevel(root)
            new_window.title("CALIBRATION")
            # new_window.overrideredirect(True)
            new_window.geometry('2000x700')
            
            try: os.remove("output.jpg")
            except:pass
            
            logo(new_window)

            main_frame = LabelFrame(new_window, text="For Selection Of Images", font=("times new roman", 30, "bold"),fg="black", relief=GROOVE)
            text1 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
            mini_text = Label(text1, text="Image 1 - Press 1 ", font=("times new roman", 15, "bold"))
            mini_text.grid(row=0,column=0,padx=5,pady=5)
            text1.grid(row=0,column=0,padx=10, pady=10)
            text2 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
            mini_text = Label(text2,text="Image 2 - Press 2 ", font=("times new roman", 15, "bold"))
            mini_text.grid(padx=5,pady=5)
            text2.grid(row=0, column=1, padx=10,pady=10)
            text3 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
            mini_text = Label(text3,text="Image 3 - Press 3 ", font=("times new roman", 15, "bold"))
            mini_text.grid(padx=5,pady=5)
            text3.grid(row=0, column=2, padx=10,pady=10)
            text4 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
            mini_text = Label(text4,text="Image 4 - Press 4 ", font=("times new roman", 15, "bold"))
            mini_text.grid(padx=5,pady=5)
            text4.grid(row=0, column=3, padx=10,pady=10)
            text5 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
            mini_text = Label(text5,text="a -> Left ", font=("times new roman", 15, "bold"))
            mini_text.grid(row=0, column=1)
            mini_text = Label(text5,text="s -> Down ", font=("times new roman", 15, "bold"))
            mini_text.grid(row=1, column=0)
            mini_text = Label(text5,text="w -> Up ", font=("times new roman", 15, "bold"))
            mini_text.grid(row=0,column=0 )
            mini_text = Label(text5,text="d -> Right ", font=("times new roman", 15, "bold"))
            mini_text.grid(row=1,column=1)
            text5.grid(row=0, column=4, padx=10,pady=10)
            text6 = Label(main_frame, text="",font=("times new roman", 20, "bold"),fg="black", relief=GROOVE)
            mini_text = Label(text6,text="Press F after Calibration is done", font=("times new roman", 15, "bold"))
            mini_text.grid(padx=5,pady=5)
            text6.grid(row=0, column=5, padx=10,pady=10)
            main_frame.pack(pady=10,padx=15)

            input_label = Label(new_window, text="Diameter:",font=("times new roman",20, "bold"),fg="white", bg="grey", height=1, width=10)
            input_label.pack(pady=10)
            input_entry = Entry(new_window)
            input_entry.pack()
            lbl1 = LabelFrame(new_window, text="work",font=("times new roman", 15, "bold"), padx=10, pady=10)
            # label1.pack()
            # lbl1.pack(side=LEFT)
            def confirm():
                print("xyz...........")
                ans = askyesno(title="exit",message="do you want to exit")
                if ans:
                    new_window.destroy()

            def img_show():
                global image_label8
                image = Image.open("output.jpg")
                image = image.resize((1000,300))  
                image_tk = ImageTk.PhotoImage(image)
                if image_label8:
                    image_label8.destroy()
                # image_tk8 = load_image("images/spring.jpg",130,130)
                image_label8 = Label(lbl1, image=image_tk)
                image_label8.image = image_tk
                image_label8.pack()
                lbl1.pack()

            def pass_input_value():
                global input_value, dropdown_values
            
                input_value = input_entry.get()
                print("Input Value:", input_value)
                x1,y2=predict_points(float(input_value))
                points=[[x1,  0.15525],[ 0.5,y2]]
                cor=json.dumps(points)
                with open(f"setting/xy{input_value}.json", "w") as dic:
                            dic.write(cor)
                imagelist=camera_opner()
                
                f_img=final_unwrap(imagelist,two_points=points)
                
                
                
                mover_image(f_img,input_value)
                img_show()

            def destroy_fun():
                global calib_flage
                calib_flage = False
                new_window.destroy()
                
            
            open_another_button = Button(new_window, text="Calibrate",font=("times new roman",15, "bold"),fg="white", bg="green", height=1, width=20, command=pass_input_value)
            open_another_button.pack(pady=20)

            
            
            exit_button = Button(new_window,text='Exit',font=("times new roman", 20, "bold"),command=destroy_fun,bg="red",fg="white")
            exit_button.place(x=1765,y=655)
            new_window.protocol("NW_Delete_Window",confirm)



    root = Tk()
    root.title("INTERFACE")
    root.geometry("700x500")
    # root.overrideredirect(True)
    bg_color = "white"

    logo(root)

    # update_video()
    # main_frame = Label(root, text="", font=("times new roman", 30, "bold"),fg="white", relief=GROOVE)
    new_window_button = Button(root, text="Calibrate",font=("times new roman",20, "bold"), command=open_new_window,fg="white", bg="Blue", height=1, width=10)
    new_window_button1 = Button(root,text="Start",font=("times new roman", 20, "bold"),command=last_window,width=10, height=1, fg="white", bg="green")
    # dropdown_button = Button(root, text="Select", command=show_dropdown)
    exit_button = Button(root,text='Exit',font=("times new roman", 20, "bold"),command=lambda: root.quit(),bg="red",fg="white")


    new_window_button.pack(expand=True )
    new_window_button1.pack(expand=True)
    exit_button.pack(padx=5, pady=5, side=RIGHT)

    new_window_button2 = Button(root, text="Preview",font=("times new roman",15, "bold"), command=preview_camera,fg="white", bg="Blue", height=1, width=10)
    new_window_button2.pack(side=LEFT)

    root.mainloop()
else: 
    print("not the right system")

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

from pypylon import pylon
import numpy as np
import cv2 as cv
import random

class camera_streams:
    capture = None
    mode = None
    IP = None
    path = None

    def __init__(self, mode, IP=None, path=None):
        self.IP = IP
        self.mode = mode
        self.path = path
        if mode == 'Industrial':

            self.capture = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.capture.Open()
            self.capture.ExposureTime.SetValue(50000)
            self.capture.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_RGB16packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
            self.bgr_img = self.frame = np.ndarray(shape=(self.capture.Height.Value, self.capture.Width.Value, 3),
                                                   dtype=np.uint8)
        elif mode == "IP cam":
            self.capture = cv.VideoCapture(self.IP)
            if self.capture.isOpened():
                print("IP Camera Connected.")
            else:
                print("Please check the host address/network connectivity")

        elif mode == "fromfile":
            self.capture = cv.VideoCapture(self.path)
            if self.capture.isOpened():
                print("Video File Ready to Read.")
            else:
                print("Path may be incorrect. Please check again.")

        else:
            for cameraID in range(0, 200):
                print("Looking for open camera device.")
                self.capture = cv.VideoCapture(cameraID)
                if self.capture.isOpened():
                    print("Working camID: ", cameraID)
                    break
                if cameraID == 200:
                    print("No camera ID is found working")

    def get_frame(self, size):
        if self.mode == "Industrial":
            grabResult = self.capture.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                frame = np.ndarray(buffer=image.GetBuffer(), shape=(image.GetHeight(), image.GetWidth(), 3),
                                   dtype=np.uint16)
                self.bgr_img[:, :, 0] = frame[:, :, 2]
                self.bgr_img[:, :, 1] = frame[:, :, 1]
                self.bgr_img[:, :, 2] = frame[:, :, 0]
                # print(self.bgr_img.shape, "----")
                self.frame = self.bgr_img.copy()
                self.frame = cv.resize(self.frame, size)
            else:
                print("Error in frame grabbing.")
            grabResult.Release()
        else:
            ret, self.frame = self.capture.read()
            if ret:
                self.frame = cv.resize(self.frame, size)
            else:
                print("Error in frame grabbing.")
        return self.frame

    def __del__(self):
        self.capture.close()
    
cap=camera_streams(mode="Industrial")
def get_relative_coordinates(event):

    x, y = event.x, event.y

    relative_x = x - (width // 2)
    relative_y = (height // 2) - y  

    print(f"Relative Coordinates from Center: ({relative_x}, {relative_y})")

def draw_center_lines(frame):
    
    cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 0, 255), 1)

    cv2.line(frame, (0, height // 2), (width, height // 2), (0, 0, 255), 1)
    return frame

def update_frame():
    frame = cap.get_frame(size=(777, 582))

    if frame is not None:
        frame = draw_center_lines(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", frame)
        cv2.waitKey(30)
        # frame = cv2.resize(frame, (width, height))
        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        label.config(image=photo)
        label.image = photo

    root.after(10, update_frame)

root = tk.Tk()
root.title("Video Frame with Center Lines")
root.geometry("800x600")

width, height = 777, 582
label = ttk.Label(root)
label.pack()

# cap = cv2.VideoCapture(0)  

label.bind("<Button-1>", get_relative_coordinates)

update_frame()

root.mainloop()
cap.release()

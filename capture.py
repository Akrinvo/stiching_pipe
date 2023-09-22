from pypylon import pylon
from pypylon import genicam
import cv2
import random

import matplotlib.pyplot as plt


class Camera(object):
    def __init__(self, countOfImagesToGrab=2, maxCamerasToUse=1, scale_percent=30, live=False):
        if (countOfImagesToGrab == 0):
            self.countOfImagesToGrab = 1
        else:
            self.countOfImagesToGrab = countOfImagesToGrab

        self.maxCamerasToUse = maxCamerasToUse
        self.exitCode = 0
        self.scale_percent = scale_percent
        self.c = 0
        self.live = live

    def capture(self):
        try:
            C1=0
            C2=0
            C3=0
            C4=0
            CamList = []


            # Get the transport layer factory
            tlFactory = pylon.TlFactory.GetInstance()

            # Get all attached devices and exit application if no device is found.
            devices = tlFactory.EnumerateDevices()
            if len(devices) == 0:
                raise pylon.RuntimeException("No camera present.")

            # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
            cameras = pylon.InstantCameraArray(
                min(len(devices), self.maxCamerasToUse))

            l = cameras.GetSize()

            # Create and attach all Pylon Devices.
            for i, cam in enumerate(cameras):
                cam.Attach(tlFactory.CreateDevice(devices[i]))

                # Print the model name of the camera.
                # print("Using device ", cam.GetDeviceInfo().GetModelName())

            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            cameras.StartGrabbing()
            # Grab c_countOfImagesToGrab from the cameras.
            for i in range(self.countOfImagesToGrab):
                if not cameras.IsGrabbing():
                    break

                grabResult = cameras.RetrieveResult(
                    5000, pylon.TimeoutHandling_ThrowException)
                cameraContextValue = grabResult.GetCameraContext()

                print("Camera ", cameraContextValue, ": ", cameras[cameraContextValue].GetDeviceInfo().GetModelName())
            # print("GrabSucceeded: ", grabResult.GrabSucceeded())
            # print("SizeX: ", grabResult.GetWidth())
            # print("SizeY: ", grabResult.GetHeight())
            # print("Gray value of first pixel: ", img[0, 0])

                img = grabResult.GetArray()

                if grabResult.GrabSucceeded():
                    image = converter.Convert(grabResult)

                    if (cameraContextValue == 0):
                        Cam0 = image.GetArray()
                        C1=1
                        width = int(Cam0.shape[1] * self.scale_percent / 100)
                        height = int(Cam0.shape[0] * self.scale_percent / 100)
                        dim = (width, height)

                        Cam0 = cv2.resize(Cam0, dim, cv2.INTER_AREA)
                        
                        Cam = Cam0.copy()
                        CamList.append(Cam0)
                # Cam0 = Cam0[50:Cam0.shape[0]-50 ,150:Cam0.shape[1]-150]
                        cv2.line(Cam, (int(
                            Cam0.shape[1]/2), 0), (int(Cam0.shape[1]/2), Cam0.shape[0]), (0, 255, 0), 1)
                        cv2.line(Cam, (0, int(
                            Cam0.shape[0]/2)), (int(Cam0.shape[1]), int(Cam0.shape[0]/2)), (0, 255, 0), 1)
                        cv2.line(Cam, (int(Cam0.shape[1]/2 - 200), 0), (int(
                            Cam0.shape[1]*0.5 - 200), int(Cam0.shape[0])), (0, 255, 0), 1)
                        cv2.line(Cam, (int(Cam0.shape[1]/2 + 200), 0), (int(
                            Cam0.shape[1]*0.5 + 200), int(Cam0.shape[0])), (0, 255, 0), 1)
                        if (self.live):
                            cv2.imshow('title0', Cam)
                            # pass
                        
                            
                    if (cameraContextValue == 1):
                        C2=1
                        Cam1 = image.GetArray()
                        Cam1 = cv2.resize(Cam1, dim, cv2.INTER_AREA)
                        CamList.append(Cam1)

                        Cam = Cam1.copy()
                        cv2.line(Cam, (int(
                            Cam0.shape[1]/2), 0), (int(Cam0.shape[1]/2), Cam0.shape[0]), (0, 255, 0), 1)
                        cv2.line(Cam, (0, int(
                            Cam0.shape[0]/2)), (int(Cam0.shape[1]), int(Cam0.shape[0]/2)), (0, 255, 0), 1)
                        cv2.line(Cam, (int(Cam0.shape[1]/2 - 200), 0), (int(
                            Cam0.shape[1]*0.5 - 200), int(Cam0.shape[0])), (0, 255, 0), 1)
                        cv2.line(Cam, (int(Cam0.shape[1]/2 + 200), 0), (int(
                            Cam0.shape[1]*0.5 + 200), int(Cam0.shape[0])), (0, 255, 0), 1)
                        if (self.live):
                            cv2.imshow('title1', Cam)
                            # pass
                    if (cameraContextValue == 2):
                        C3=1
                        Cam2 = image.GetArray()
                        Cam2 = cv2.resize(Cam2, dim, cv2.INTER_AREA)
                        CamList.append(Cam2)

                        Cam = Cam2.copy()
                        cv2.line(Cam, (int(
                            Cam0.shape[1]/2), 0), (int(Cam0.shape[1]/2), Cam0.shape[0]), (0, 255, 0), 1)
                        cv2.line(Cam, (0, int(
                            Cam0.shape[0]/2)), (int(Cam0.shape[1]), int(Cam0.shape[0]/2)), (0, 255, 0), 1)
                        cv2.line(Cam, (int(Cam0.shape[1]/2 - 200), 0), (int(
                            Cam0.shape[1]*0.5 - 200), int(Cam0.shape[0])), (0, 255, 0), 1)
                        cv2.line(Cam, (int(Cam0.shape[1]/2 + 200), 0), (int(
                            Cam0.shape[1]*0.5 + 200), int(Cam0.shape[0])), (0, 255, 0), 1)
                        if (self.live):
                            cv2.imshow('title2', Cam)
                            # pass
                    if (cameraContextValue == 3):
                        C4=1
                        Cam3 = image.GetArray()
                        Cam3 = cv2.resize(Cam3, dim, cv2.INTER_AREA)
                        CamList.append(Cam3)

                        Cam = Cam3.copy()
                        cv2.line(Cam, (int(
                            Cam0.shape[1]/2), 0), (int(Cam0.shape[1]/2), Cam0.shape[0]), (0, 255, 0), 1)
                        cv2.line(Cam, (0, int(
                            Cam0.shape[0]/2)), (int(Cam0.shape[1]), int(Cam0.shape[0]/2)), (0, 255, 0), 1)
                        cv2.line(Cam, (int(Cam0.shape[1]/2 - 200), 0), (int(
                            Cam0.shape[1]*0.5 - 200), int(Cam0.shape[0])), (0, 255, 0), 1)
                        cv2.line(Cam, (int(Cam0.shape[1]/2 + 200), 0), (int(
                            Cam0.shape[1]*0.5 + 200), int(Cam0.shape[0])), (0, 255, 0), 1)
                        if (self.live):
                            cv2.imshow('title3', Cam)
                            # pass

                    if (self.live):
                        k = cv2.waitKey(10)
                        if k == ord('q'):
                            break
                            pass

                grabResult.Release()

        except genicam.GenericException as e:
            # Error handling
            print("An exception occurred.", e.GetDescription())
            exitCode = 1

        
        # if (self.maxCamerasToUse==0):
        #     print(1)
        #     CamList.append(Cam0)
        # if (self.maxCamerasToUse > 1):
        #     print(2)
        #     CamList.append(Cam1)
        # if (self.maxCamerasToUse > 2):
        #     CamList.append(Cam2)
        # if (self.maxCamerasToUse > 3):
        #     CamList.append(Cam3)
        # # cv2.destroyAllWindows()
        
        return CamList,C1+C2+C3+C4
cap=Camera(live=True)
cap1=Camera(live=True)
cap2=Camera(live=True)
cap3=Camera(live=True)

while True:
       
    a,total=cap.capture()
    b,total=cap1.capture()
    c,total=cap2.capture()
    d,total=cap3.capture()
    
    
    print(len(a),total)
    try:
        cv2.imshow("a",a[-1])
        cv2.imshow("a1",b[-1])
        cv2.imshow("a2",c[-1])
        cv2.imshow("a3",d[-1])
    except:pass
    if cv2.waitKey(1)==ord("q"):break
cv2.destroyAllWindows()

    
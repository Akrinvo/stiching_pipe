from pypylon import pylon

from pypylon import genicam
import cv2

def getimage( countOfImagesToGrab=4, maxCamerasToUse=4, scale_percent=30, live=False):
        if (countOfImagesToGrab == 0):
            countOfImagesToGrab = 10000000
        else:
            countOfImagesToGrab = countOfImagesToGrab

        maxCamerasToUse = maxCamerasToUse
        exitCode = 0
        scale_percent = scale_percent
        c = 0
  
        Cam0 = 0
        Cam1 = 0
        Cam2 = 0
        Cam3 = 0




    
        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()
        print(len(devices))
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")

        # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        cameras = pylon.InstantCameraArray(
            min(len(devices), maxCamerasToUse))

        l = cameras.GetSize()
        # print(l)
        # Create and attach all Pylon Devices.
        for i, cam in enumerate(cameras):
            cam.Attach(tlFactory.CreateDevice(devices[i]))
            print(cam)

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        cameras.StartGrabbing()
        # Grab c_countOfImagesToGrab from the cameras.
        for i in range(countOfImagesToGrab):
            if not cameras.IsGrabbing():
                break

            grabResult = cameras.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException)
            cameraContextValue = grabResult.GetCameraContext()

            # print("Camera ", cameraContextValue, ": ", cameras[cameraContextValue].GetDeviceInfo().GetModelName())
            # print("GrabSucceeded: ", grabResult.GrabSucceeded())
            # print("SizeX: ", grabResult.GetWidth())
            # print("SizeY: ", grabResult.GetHeight())
            # print("Gray value of first pixel: ", img[0, 0])

            img = grabResult.GetArray()
            print(img.shape)
            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)

                if (cameraContextValue == 0):
                    img = image.GetArray()
                    # percent of original size
                    width = int(img.shape[1] * scale_percent / 100)
                    height = int(img.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    img = cv2.resize(img, dim, cv2.INTER_AREA)
                    Cam0 = img  # [0:img.shape[0] ,50:470]
                    # Cam1 = img 
                    # Cam2 = img 
                    # Cam3 = img 
                    cv2.namedWindow('title0', cv2.WINDOW_AUTOSIZE)
                    cv2.line(Cam0, (int(Cam0.shape[1]/2), 0), (int(Cam0.shape[1]/2), Cam0.shape[0]), (0, 255, 0), 1)
                    cv2.line(Cam0, (int(Cam0.shape[1]*0.37), int(Cam0.shape[0]/2)), (int(Cam0.shape[1]*0.64), int(Cam0.shape[0]/2)), (0, 255, 0), 1)
                    cv2.imshow('title0', Cam0)
                if (cameraContextValue == 1):
                    img = image.GetArray()
                    # percent of original size
                    img = cv2.resize(img, dim, cv2.INTER_AREA)
                    Cam1 = img  # [0:img.shape[0] ,50:470]
                    cv2.namedWindow('title0', cv2.WINDOW_AUTOSIZE)
                    cv2.line(Cam0, (int(Cam0.shape[1]/2), 0), (int(Cam0.shape[1]/2), Cam0.shape[0]), (0, 255, 0), 1)
                    cv2.line(Cam0, (int(Cam0.shape[1]*0.37), int(Cam0.shape[0]/2)), (int(Cam0.shape[1]*0.64), int(Cam0.shape[0]/2)), (0, 255, 0), 1)
                    cv2.imshow('title1', Cam1)

                if (cameraContextValue == 2):
                    img = image.GetArray()
                    # percent of original size
                    img = cv2.resize(img, dim, cv2.INTER_AREA)
                    # [50:img.shape[0]-50 ,150:img.shape[1]-150]
                    Cam2 = img
                    cv2.namedWindow('title0', cv2.WINDOW_AUTOSIZE)
                    cv2.line(Cam0, (int(Cam0.shape[1]/2), 0), (int(Cam0.shape[1]/2), Cam0.shape[0]), (0, 255, 0), 1)
                    cv2.line(Cam0, (int(Cam0.shape[1]*0.37), int(Cam0.shape[0]/2)), (int(Cam0.shape[1]*0.64), int(Cam0.shape[0]/2)), (0, 255, 0), 1)
                    cv2.imshow('title2', Cam2)

                if (cameraContextValue == 3):
                    img = image.GetArray()
                    # percent of original size
                    img = cv2.resize(img, dim, cv2.INTER_AREA)
                    # [50:img.shape[0]-50 ,150:img.shape[1]-150]
                    Cam3 = img
                    cv2.namedWindow('title0', cv2.WINDOW_AUTOSIZE)
                    cv2.line(Cam0, (int(Cam0.shape[1]/2), 0), (int(Cam0.shape[1]/2), Cam0.shape[0]), (0, 255, 0), 1)
                    cv2.line(Cam0, (int(Cam0.shape[1]*0.37), int(Cam0.shape[0]/2)), (int(Cam0.shape[1]*0.64), int(Cam0.shape[0]/2)), (0, 255, 0), 1)
                    cv2.imshow('title3', Cam3)

                k = cv2.waitKey(10)
                if k == ord('q'):
                    break

            grabResult.Release()

            # cv2.destroyAllWindows()
while(True):getimage(live=True)
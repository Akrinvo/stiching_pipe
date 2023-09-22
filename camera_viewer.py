from pypylon import pylon
import time
import cv2


tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
cv2.namedWindow("imag1",cv2.WINDOW_NORMAL)
# cv2.namedWindow("thresh",cv2.WINDOW_NORMAL)
# cv2.namedWindow("new",cv2.WINDOW_NORMAL)
cv2.namedWindow("imag2",cv2.WINDOW_NORMAL)
cv2.namedWindow("imag3",cv2.WINDOW_NORMAL)
cv2.namedWindow("imag4",cv2.WINDOW_NORMAL)

# cv2.namedWindow("stich",cv2.WINDOW_NORMAL)

while(True):

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

            width = 600
            height = 680
            dim = (width, height)

            im = cv2.resize(im, dim, cv2.INTER_AREA)
            imagelist.append(im)
    # image=hconcat_resize(imagelist)
    
    cv2.imshow('imag1', imagelist[0])
    cv2.imshow('imag2', imagelist[1])

    cv2.imshow('imag3', imagelist[2])
    cv2.imshow('imag4', imagelist[3])
    if cv2.waitKey(1)==ord("q"):break

cv2.destroyAllWindows()

from pypylon import pylon
from pypylon import genicam
import cv2
import numpy as np
import sys
class camera_streams:

    def __init__(self):

        self.capture = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.capture.Open()
        self.capture.ExposureTime.SetValue(50000)
        print("Using device ", self.capture.GetDeviceInfo().GetModelName())
        self.capture.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_RGB16packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
        self.bgr_img = self.frame = np.ndarray(shape=(self.capture.Height.Value, self.capture.Width.Value, 3),
                                               dtype=np.uint8)

    def get_frame(self, size):

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
            self.frame = cv2.resize(self.frame, size)
        else:
            print("Error in frame grabbing.")
        grabResult.Release()

        return self.frame

    def __del__(self):
        self.capture.close()


cap = camera_streams()
frame = cap.get_frame((680, 500))
cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
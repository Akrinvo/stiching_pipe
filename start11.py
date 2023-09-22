










from pypylon import pylon
import time
import cv2
from stichlib import *
import json
def camera_opner():
    scale_percent=30
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    cv2.namedWindow("imagee",cv2.WINDOW_NORMAL)
    # cv2.namedWindow("thresh",cv2.WINDOW_NORMAL)
    # cv2.namedWindow("new",cv2.WINDOW_NORMAL)
    cv2.namedWindow("allimg",cv2.WINDOW_NORMAL)
    cv2.namedWindow("allimgc",cv2.WINDOW_NORMAL)
    cv2.namedWindow("himagwe",cv2.WINDOW_NORMAL)

    # cv2.namedWindow("stich",cv2.WINDOW_NORMAL)
    with open("setting/xy40.36.json","r") as openfile:
        Shape=json.load(openfile)
    print(Shape)
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

                width = int(im.shape[1] * scale_percent / 100)
                height = int(im.shape[0] * scale_percent / 100)
                dim = (width, height)

                im = cv2.resize(im, dim, cv2.INTER_AREA)
                imagelist.append(im)
        image=hconcat_resize(imagelist)
        
        cv2.imshow('imagee', image)
    
            
        f_img=[]
        threshs=[]
        news=[]
        crp=0.004470
        plan_rod=False
        r_h,r_w=420,650


        if len(imagelist)==4:
            final,thresh,new=unrap(imagelist[0].copy(),plan_rod=plan_rod,Shape=Shape)
            h,w=final.shape[:2]
            final = final[100:h-100,int(w*crp):w-int(w*crp)]
            final=cv2.resize(final,(r_w,r_h))
        
            f_img.append(final)

            threshs.append(thresh)
            news.append(new)

            final1,thresh1,new1=unrap(imagelist[1].copy(),plan_rod=plan_rod,Shape=Shape)
            h,w=final1.shape[:2]
            final1 = final1[100:h-100,int(w*crp):w-int(w*crp)]
            final1=cv2.resize(final1,(r_w,r_h))

            f_img.append(final1)
            threshs.append(thresh1)
            news.append(new1)
            final2,thresh2,new2=unrap(imagelist[2].copy(),plan_rod=plan_rod,Shape=Shape)
            h,w=final2.shape[:2]
            final2 = final2[100:h-100,int(w*crp):w-int(w*crp)]
            final2=cv2.resize(final2,(r_w,r_h))

            f_img.append(final2)
            threshs.append(thresh2)
            news.append(new2)
            final3,thresh3,new3=unrap(imagelist[3].copy(),plan_rod=plan_rod,Shape=Shape)

            h,w=final3.shape[:2]
            final3= final3[100:h-100,int(w*crp):w-int(w*crp)]
            final3=cv2.resize(final3,(r_w,r_h))

            f_img.append(final3)
            threshs.append(thresh3)
            news.append(new3)
            print(11111111111111111111111111)
        
        # cv2.imshow('image', imagelist[0])
        # cv2.imshow('final', f_img[0])
        # cv2.imshow('image1', imagelist[1])
        # cv2.imshow('final1', f_img[1])
        # cv2.imshow('image2', imagelist[2])
        # cv2.imshow('final2', f_img[2])
        # cv2.imshow('image3', imagelist[3])
        # cv2.imshow('final3', f_img[3])
        # if cv2.waitKey(1000)==ord('q'):break

        for i,j in zip(f_img,range(4)):
            # cv2.imshow(f'image2{j}', imagelist[j])
        

            cv2.imwrite(f'final2{j}.jpg', i)
            
            # cv2.waitKey(100)

        
        fimage=hconcat_resize(f_img)
        himage=fimage.copy()
        for points in range(0,himage.shape[1],20):
            cv2.line(himage,(0,points),(100000,points),(0,255,0),1)
        threshh=hconcat_resize(threshs)
        neww=hconcat_resize(news)
        allimg=vconcat_resize([threshh,neww])
        allimgc=vconcat_resize([image,fimage])



        cv2.imshow("himagwe",himage)
        cv2.imshow("allimg",allimg)
        cv2.imshow("allimgc",allimgc)

        # cv2.imshow('stich', fimage)


        k=cv2.waitKey(1)
        if k==ord('s'):
            cv2.imwrite('stich.jpg', fimage)

        # except Exception as e:
        #     print(e)

        if k ==ord('q'):break
    cv2.destroyAllWindows()
    camera.Close()

    # breakqq

camera_opner()




from pypylon import pylon
import time
import cv2
from stichlib import *
import json
from calibrated import *



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
    # image=hconcat_resize(imagelist)
    
    # cv2.imshow('imagee', image)

        
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
    
    # cv2.imshow('image', imagelist[0])
    # cv2.imshow('final', f_img[0])
    # cv2.imshow('image1', imagelist[1])
    # cv2.imshow('final1', f_img[1])
    # cv2.imshow('image2', imagelist[2])
    # cv2.imshow('final2', f_img[2])
    # cv2.imshow('image3', imagelist[3])
    # cv2.imshow('final3', f_img[3])
    # if cv2.waitKey(1000)==ord('q'):break

  
        
        # cv2.waitKey(100)

    
    





if __name__=="__main__":
    for iii in range(11):
        cv2.namedWindow("imagee",cv2.WINDOW_NORMAL)
        # cv2.namedWindow("thresh",cv2.WINDOW_NORMAL)
        # cv2.namedWindow("new",cv2.WINDOW_NORMAL)
        cv2.namedWindow("allimgc",cv2.WINDOW_NORMAL)
        cv2.namedWindow("himagwe",cv2.WINDOW_NORMAL)

        # cv2.namedWindow("stich",cv2.WINDOW_NORMAL)
    
        dia=52.66
        with open(f"setting/xy{dia}.json","r") as openfile:
            points=json.load(openfile)


                
        imagelist= camera_opner()   
        f_img=final_unwrap(imagelist,two_points=points) 

        image=hconcat_resize(imagelist)
        fimage=hconcat_resize(f_img)
        himage=fimage.copy()

    
        allimgc=vconcat_resize([image,fimage])



        cv2.imshow("himagwe",himage)
        cv2.imshow("allimgc",allimgc)

        # cv2.imshow('stich', fimage)


        if cv2.waitKey(1)==ord("q"):break
      
    cv2.destroyAllWindows()
    # camera.Close()


    


   

    # breakqq


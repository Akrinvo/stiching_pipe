import math
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import multiprocessing
# def task(arg):
#     return sum([math.sqrt(i) for i in range(1, arg)])
 
# # protect the entry point
# if __name__ == '__main__':
#     # report a message
#     print('Starting task...')
#     # perform calculations
#     a=time.time()
#     results = [task(i) for i in range(1,5000)]
#     results = [task(i) for i in range(1,5000)]

#     # report a message
#     print('Done. ',time.time()-a)





 

# def task(arg):
#     return sum([math.sqrt(i)*22222222222 for i in range(1, arg)])
 
# def multi(arg):
#         with ProcessPoolExecutor(2) as exe:

#            return exe.map(task, range(1,arg))
# if __name__ == '__main__':
 
#     print('Starting task...')
#     a=time.time()
#     multi(5000)
#     multi(5000)



 
#     print('Done. ',time.time()-a)
















from pypylon import pylon
import time
import cv2
from stichlib import *
import json
from calibrated import *
import os

crp=0.004470
plan_rod=False
r_h,r_w=1400,1900
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()


def camera1(device=None,name=None):
    global tl_factory
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------")

    camera = pylon.InstantCamera()

    # while os.path.isfile(name)==False:
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

        width = 2590
        height = 1942
    
        

        cv2.imwrite(f"{name}",im)
def m_camera1(device=None,name=None):
  

    with Pool(8) as pool:
     
        pool.apply(camera1, args=(device,name))

def camera_opner():
    global tl_factory,devices
    tl_factory = pylon.TlFactory.GetInstance()
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

            width = 2590
            height = 1942
     
            

            imagelist.append(im)
    if len(imagelist)==4:
        return imagelist
        
def m_unwrap(image=None,two_points=None,name=None):
    global plan_rod,r_w,r_h

    final,out,g=unrap(image.copy(),plan_rod=plan_rod,two_points=two_points)
    final=cv2.resize(final,(r_w,r_h))
    cv2.imwrite(f"{name}",final)

    



def f_unwrap(image=None,two_points=None,name=None):
    with Pool(8) as pool:
     
        pool.apply(m_unwrap, args=(image,two_points,name))

def final_unwrap(imagelist,two_points):

    global crp,plan_rod,r_h,r_w




    a=time.time()
    process1 = multiprocessing.Process(target=f_unwrap,args=(imagelist[0],two_points,"final20.jpg"))
    process1.start()
    process2 = multiprocessing.Process(target=f_unwrap,args=(imagelist[1],two_points,"final21.jpg"))
    process2.start()
    process3 = multiprocessing.Process(target=f_unwrap,args=(imagelist[2],two_points,"final22.jpg"))
    process3.start()
    process4 = multiprocessing.Process(target=f_unwrap,args=(imagelist[3],two_points,"final23.jpg"))
    process4.start()
    process1.join()
    process2.join()
    process3.join()
    process4.join()






    # final,thresh,new=unrap(imagelist[0].copy(),plan_rod=plan_rod,two_points=two_points)
    # final=cv2.resize(final,(r_w,r_h))
    # f_img.append(final)

  

    # final1,thresh1,new1=unrap(imagelist[1].copy(),plan_rod=plan_rod,two_points=two_points)
    # final1=cv2.resize(final1,(r_w,r_h))

    # f_img.append(final1)

    # final2,thresh2,new2=unrap(imagelist[2].copy(),plan_rod=plan_rod,two_points=two_points)
    # final2=cv2.resize(final2,(r_w,r_h))

    # f_img.append(final2)

    # final3,thresh3,new3=unrap(imagelist[3].copy(),plan_rod=plan_rod,two_points=two_points)


    # final3=cv2.resize(final3,(r_w,r_h))

    # f_img.append(final3)
    print("unwrap time = ",time.time()-a)

    f_img=[cv2.imread("final20.jpg"),cv2.imread("final21.jpg"),cv2.imread("final22.jpg"),cv2.imread("final23.jpg")]
    if len(f_img)==4:
   
        print(time.time()-a)
        
        return f_img
    

    
    





if __name__=="__main__":
    for iii in range(11):
        a=time.time()
        
        cv2.namedWindow("allimgc",cv2.WINDOW_NORMAL)
        cv2.namedWindow("himagwe",cv2.WINDOW_NORMAL)

        dia=52.66
        x1,y2=predict_points(dia)
        points=[[x1,  0.16525],[ 0.5,y2]]
        print(points)


        b=time.time()
        
        imagelist= camera_opner()   
        print("camera opning time = ",time.time()-b)
        F_img=final_unwrap(imagelist,two_points=points) 
        if len(F_img)==4:
            image=hconcat_resize(imagelist)
            fimage=hconcat_resize(F_img)
            himage=fimage.copy()

        
            allimgc=vconcat_resize([image,fimage])



            cv2.imshow("himagwe",himage)
            cv2.imshow("allimgc",allimgc)

            # cv2.imshow('stich', fimage)


            if cv2.waitKey(1)==ord("q"):break
            print("total time  ",time.time()-a)
      
    cv2.destroyAllWindows()
    # camera.Close()


    


   

    # breakqq



# s=0
 
# # define a cpu-intensive task
# def task(arg):
#     global s
#     s+=sum([math.sqrt(i)*22222222222 for i in range(1, arg)])
    
 

# def poolmulti(arg,d):
#     with Pool(8) as pool:
#         # perform calculations
#         print(d)
#         return pool.apply(task,args=(arg,))


# if __name__ == '__main__':
#     print('Starting task...')
#     a=time.time()
#     process1 = multiprocessing.Process(target=poolmulti,args=(5000,10))
#     process2 = multiprocessing.Process(target=poolmulti,args=(5000,10))
#     process1.start()
#     process2.start()
#     process1.join()
#     process2.join()
#     # poolmulti(5000)
#     # poolmulti(5000)
#     print(s)
#     print('Done. ',time.time()-a)






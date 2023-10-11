import cv2
import numpy as np
import json
from stichlib import hconcat_resize,image_loc
import os

def calibrated_Himage(ori=None,path=None):
    images=sorted(image_loc(path))
    no_ofimg=len(images)
    bg_h,bg_w=ori['bg_shape'][1][0],ori["bg_shape"][1][1]*no_ofimg
    pad=ori["bg_shape"][0]
    if no_ofimg >=2:
        bg= cv2.copyMakeBorder(np.zeros((bg_h,bg_w,3),np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, value = 0)
        img1=cv2.rotate(cv2.imread(images[0]), cv2.ROTATE_90_CLOCKWISE)
        bg[pad+ori["img1"][0]:bg_h+ori["img1"][0]+pad,pad+ori["img1"][1]:ori["bg_shape"][1][1]+ori["img1"][1]+pad]=img1
        
        for himg,no in zip(images[1:],range(1,no_ofimg)):
            img=cv2.rotate(cv2.imread(himg), cv2.ROTATE_90_CLOCKWISE)
            bg[pad+ori["img2"][0]*no:bg_h+ori["img2"][0]*no+pad ,pad + ori["img2"][1]*no + ori["bg_shape"][1][1]*no  : ori["bg_shape"][1][1] + ori["bg_shape"][1][1]*no + ori["img2"][1]*no + pad ]=img
            

                                                                                                                                        
    return bg




def hori_oneByone(ori=None,path=None):
    images=sorted(image_loc(path))
    dia=path.split("/")[-1].split("_")[1]
    hpath=path+"__"+"full"
    if os.path.exists(hpath):
        no_ofimg=len(images)
        if no_ofimg >=2:
            prev_img=cv2.imread(f"{hpath}/full_{dia}_.jpg")
            current_nofimg=prev_img.shape[1]//ori["bg_shape"][1][1]
            no_ofaddimg=no_ofimg-current_nofimg
            print(current_nofimg,no_ofimg)

            bg_h,bg_w=ori['bg_shape'][1][0],prev_img.shape[1]+ori["bg_shape"][1][1]*no_ofaddimg
            pad=ori["bg_shape"][0]
            bg= cv2.copyMakeBorder(np.zeros((bg_h,bg_w,3),np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, value = 0)
            print(bg.shape,prev_img.shape)
            bg[pad:pad+prev_img.shape[0],pad:pad+prev_img.shape[1]]=prev_img
            img=cv2.rotate(cv2.imread(images[-1]), cv2.ROTATE_90_CLOCKWISE)
            for i in range(no_ofaddimg):
                no=current_nofimg+i
                ww=pad + ori["img2"][1]*no + ori["bg_shape"][1][1]*no 
                w1=ori["bg_shape"][1][1] + ori["bg_shape"][1][1]*no + ori["img2"][1]*no + pad
                hh=pad+ori["img2"][0]*no
                h1=bg_h+ori["img2"][0]*no+pad 
   
            bg[hh:h1,ww:w1]=img
            bg=bg[pad:h1,pad:w1]
            cv2.imwrite(f"{hpath}/full_{dia}_.jpg",bg)
            return bg

        
           
    else:
        no_ofimg=len(images)
        if no_ofimg>=2:
            os.makedirs(hpath)

            bg_h,bg_w=ori['bg_shape'][1][0],ori["bg_shape"][1][1]*no_ofimg
            pad=ori["bg_shape"][0]
            img1=cv2.rotate(cv2.imread(images[0]), cv2.ROTATE_90_CLOCKWISE)
            bg= cv2.copyMakeBorder(np.zeros((bg_h,bg_w,3),np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, value = 0)
            bg[pad+ori["img1"][0]:bg_h+ori["img1"][0]+pad,pad+ori["img1"][1]:ori["bg_shape"][1][1]+ori["img1"][1]+pad]=img1
            min_w=pad+ori["img1"][1]
            max_w=ori["bg_shape"][1][1]+ori["img1"][1]+pad
            min_h=pad+ori["img1"][0]
            max_h=bg_h+ori["img1"][0]+pad

            for himg,no in zip(images[1:],range(1,no_ofimg)):
                img=cv2.rotate(cv2.imread(himg), cv2.ROTATE_90_CLOCKWISE)
                ww=pad + ori["img2"][1]*no + ori["bg_shape"][1][1]*no 
                w1=ori["bg_shape"][1][1] + ori["bg_shape"][1][1]*no + ori["img2"][1]*no + pad
                hh=pad+ori["img2"][0]*no
                h1=bg_h+ori["img2"][0]*no+pad 
                bg[hh:h1,ww:w1]=img
                if min_w>ww or min_w==0:
                    min_w=ww 
                if max_w<w1:
                    max_w=w1
                if min_h>hh or min_h==0:
                    min_h=hh
                if max_h<h1:
                    max_h=h1
  
            bg=bg[min_h:max_h,min_w:max_w]
            cv2.imwrite(f"{hpath}/full_{dia}_.jpg",bg)
            return bg





                                                                                                                                        
    

# twoimage=[]
hor_pad=200
hori_pre=1

hori_ver=0
Hori=0
H,W=0,0

hori_ver2=0
Hori2=0
hor_ori={}
img_no=0
def hor_clear(images):
    img1,img2=images

    global hor_ori,hori_ver,Hori,hori_ver2,Hori2,hor_pad,img_no
    bg= cv2.copyMakeBorder(np.zeros((H,W,3),np.uint8), hor_pad, hor_pad, hor_pad, hor_pad, cv2.BORDER_CONSTANT, None, value = 0)
    if img_no==1:
        if len(hor_ori)>=3:
            bg[hor_pad+hor_ori["img2"][0]:hor_ori["bg_shape"][1][0]+hor_ori["img2"][0]+hor_pad , hor_pad+hor_ori["img2"][1] + hor_ori["bg_shape"][1][1]:hor_ori["bg_shape"][1][1]*2+hor_ori["img2"][1]+hor_pad]=img2

        if len(hor_ori)>=2:
            bg[hor_pad+hor_ori["img1"][0]:hor_ori["bg_shape"][1][0]+hor_ori["img1"][0]+hor_pad , hor_pad+hor_ori["img1"][1]:hor_ori["bg_shape"][1][1]+hor_ori["img1"][1]+hor_pad]=img1
    if img_no==2:

        if len(hor_ori)>=2:
            bg[hor_pad+hor_ori["img1"][0]:hor_ori["bg_shape"][1][0]+hor_ori["img1"][0]+hor_pad , hor_pad+hor_ori["img1"][1]:hor_ori["bg_shape"][1][1]+hor_ori["img1"][1]+hor_pad]=img1

        if len(hor_ori)>=3:
            bg[hor_pad+hor_ori["img2"][0]:hor_ori["bg_shape"][1][0]+hor_ori["img2"][0]+hor_pad , hor_pad+hor_ori["img2"][1] + hor_ori["bg_shape"][1][1]:hor_ori["bg_shape"][1][1]*2+hor_ori["img2"][1]+hor_pad]=img2



    return bg

def hori_move(path=None,dia=None):
    global hor_ori,hori_ver,Hori,hori_ver2,Hori2,img_no,H,W
    twoimage=[]
    images=sorted(image_loc(path))
    if len(images)>=2:
        twoimage.append(cv2.rotate(cv2.imread(images[0]), cv2.ROTATE_90_CLOCKWISE))
        twoimage.append(cv2.rotate(cv2.imread(images[1]), cv2.ROTATE_90_CLOCKWISE))
        H=twoimage[0].shape[0]
        W=twoimage[0].shape[1]*2
        bg_img= cv2.copyMakeBorder(np.zeros((H,W,3),np.uint8), hor_pad, hor_pad, hor_pad, hor_pad, cv2.BORDER_CONSTANT, None, value = 0)
        hor_ori={"bg_shape":[hor_pad,twoimage[0].shape[:2]]}

    

        while True:
            
            img1,img2=twoimage
            h1,w1=img1.shape[:2]
            h2,w2=img2.shape[:2]
    
            bg=bg_img.copy()
          
            
            k=cv2.waitKey(1)
            # print(k)
            if k==ord('1'):
                img_no=1
            if k==ord('2'):
                img_no=2
        

            if img_no==1:
                # bg_img[0:h1,0:w1]=img1

                bg_img[hor_pad+hori_ver:h1+hori_ver+hor_pad,hor_pad+Hori:w1+Hori+hor_pad]=img1
                
                ###  hori_vertical move
                if k==ord("w"):
                    hori_ver-=hori_pre
                if k==ord('s'):
                    hori_ver+=hori_pre 


            ###  horizontal move
                if k==ord("a"):
                    Hori-=hori_pre
                if k==ord('d'):
                    Hori+=hori_pre


                hor_ori["img1"]=[hori_ver,Hori]
                
                bg_img=hor_clear(images=twoimage)

                


            if img_no==2:
                bg_img[hor_pad+hori_ver2:h1+hori_ver2+hor_pad,         hor_pad+Hori2+w1    :w1+w2+Hori2+hor_pad]=img2

                    ###  vertical move
                if k==ord("w"):
                    hori_ver2-=hori_pre
                if k==ord('s'):
                    hori_ver2+=hori_pre


            ###  horizontal move
                if k==ord("a"):
                    Hori2-=hori_pre
                if k==ord('d'):
                    Hori2+=hori_pre
                

                hor_ori["img2"]=[hori_ver2,Hori2]
                bg_img=hor_clear(images=twoimage)

                









        
            cv2.namedWindow("calibrate",cv2.WINDOW_NORMAL)

            cv2.imshow("calibrate",bg)


            if k==ord('f'):
                cor=json.dumps(hor_ori)
                with open(f"horiz{dia}.json", "w") as dic:
                    dic.write(cor)
                break
            
        
        cv2.destroyAllWindows()
    # print(images)
    




if __name__=="__main__":
    path='SmartViz_Image_Manual_Inspection/dia_40.00_2'
    dia=path.split("/")[-1].split("_")[1]
    print(dia)
    # hori_move(path=path,dia=dia)
    with open(f"horiz{dia}.json","r") as openfile:
                    cal=json.load(openfile)
    bg=hori_oneByone(ori=cal,path=path)
    # bg=calibrated_Himage(ori=cal,path=path)
    cv2.namedWindow("overallbg",cv2.WINDOW_NORMAL)
    cv2.imshow("overallbg",bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
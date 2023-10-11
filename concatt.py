import cv2
import numpy as np
import json
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

def calibrated_image(ori,images):

    img1,img2,img3,img4=images
    bg= cv2.copyMakeBorder(np.zeros((ori["bg_shape"][1],ori["bg_shape"][2],3),np.uint8), ori["bg_shape"][0], ori["bg_shape"][0], ori["bg_shape"][0], ori["bg_shape"][0], cv2.BORDER_CONSTANT, None, value = 0)

    bg[ori["img1"][0] :ori["img1"][1],ori["img1"][2]:ori["img1"][3]]=img1
    bg[ori["img2"][0] :ori["img2"][1],ori["img2"][2]:ori["img2"][3]]=img2
    bg[ori["img3"][0] :ori["img3"][1],ori["img3"][2]:ori["img3"][3]]=img3
    bg[ori["img4"][0] :ori["img4"][1],ori["img4"][2]:ori["img4"][3]]=img4


    final_img=bg[max(ori["img1"][0],ori["img2"][0],ori["img3"][0],ori["img4"][0]):
                min(ori["img1"][1],ori["img2"][1],ori["img3"][1],ori["img4"][1]),
                ori["img1"][2]:ori["img4"][3]]

    # cv2.namedWindow("final",cv2.WINDOW_NORMAL)

    # cv2.imshow("final",final_img)
    return final_img
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


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

pre=1
f_=0
ori={"bg_shape":[pad,H,W]}
def clear(images):
    img1,img2,img3,img4=images

    global ori,ver,hori,ver2,hori2,ver3,hori3,ver4,hori4
    bg= cv2.copyMakeBorder(np.zeros((H,W,3),np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, value = 0)
    if img_no==1:

        if len(ori)>=3:
            bg[ori["img2"][0] :ori["img2"][1],ori["img2"][2]:ori["img2"][3]]=img2
        if len(ori)>=4:
            bg[ori["img3"][0] :ori["img3"][1],ori["img3"][2]:ori["img3"][3]]=img3
        if len(ori)==5:
            bg[ori["img4"][0] :ori["img4"][1],ori["img4"][2]:ori["img4"][3]]=img4
        if len(ori)>=2:
            bg[ori["img1"][0] :ori["img1"][1],ori["img1"][2]:ori["img1"][3]]=img1
    if img_no==2:

        if len(ori)>=4:
            bg[ori["img3"][0] :ori["img3"][1],ori["img3"][2]:ori["img3"][3]]=img3
        if len(ori)==5:
            bg[ori["img4"][0] :ori["img4"][1],ori["img4"][2]:ori["img4"][3]]=img4
        if len(ori)>=2:
            bg[ori["img1"][0] :ori["img1"][1],ori["img1"][2]:ori["img1"][3]]=img1
        if len(ori)>=3:
            bg[ori["img2"][0] :ori["img2"][1],ori["img2"][2]:ori["img2"][3]]=img2

    if img_no==3:


        if len(ori)==5:
            bg[ori["img4"][0] :ori["img4"][1],ori["img4"][2]:ori["img4"][3]]=img4
        if len(ori)>=2:
            bg[ori["img1"][0] :ori["img1"][1],ori["img1"][2]:ori["img1"][3]]=img1
        if len(ori)>=3:
            bg[ori["img2"][0] :ori["img2"][1],ori["img2"][2]:ori["img2"][3]]=img2
        if len(ori)>=4:
            bg[ori["img3"][0] :ori["img3"][1],ori["img3"][2]:ori["img3"][3]]=img3
    if img_no==4:



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
    control=np.ones((600,bg_img.shape[1],3),np.uint8)*255
    text1=   "  For The Movement of Image      Press 'w' For UP           Press 's' For DOWN         Press 'a' For LEFT           Press 'd' for RIGHT  "

    text2=   "  For Selection of Image         For 1st Image press '1'    For 2nd Image press '2'    For 3rd Image press '3'    For 4th Image press '4'"
    text3=   "                                          After the calibration is done press 'f' to save the calibration                                                 "           

    control=cv2.putText(control,text=text1,org=(400,100),  fontFace = cv2.FONT_HERSHEY_DUPLEX,  fontScale = 3.0,  color = 0,  thickness = 6)
    control=cv2.putText(control,text=text2,org=(400,300),  fontFace = cv2.FONT_HERSHEY_DUPLEX,  fontScale = 3.0,  color = 0,  thickness = 6)
    control=cv2.putText(control,text=text3,org=(400,500),  fontFace = cv2.FONT_HERSHEY_DUPLEX,  fontScale = 3.0,  color = 0,  thickness = 6)

    while True:
         
        img1,img2,img3,img4=images
        h1,w1=img1.shape[:2]
        h2,w2=img2.shape[:2]
        h3,w3=img3.shape[:2]
        h4,w4=img4.shape[:2]
        bg=bg_img.copy()
        # conimg= move_ver_image([img1,img2,img3,img4],bg_img=bg_img,img_no=img_no)
        # bg_h,bg_w=bg_img.shape[:2]
        # cv2.imshow("image",images[1])

        # cv2.imshow("zoomed",zoom_at(image,zoom=n))
        overall_bg=vconcat_resize([control,bg_img])
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
            bg_img[pad+ver3:h1+ver3+pad,  pad+hori3+w1+w2:w1+w2+w3+hori3+pad]=img3

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

            out=calibrated_image(ori,images)

            cv2.imwrite("output.jpg",out)
            break
        
        # print(ori)
        k=0
        cv2.namedWindow("calibrate",cv2.WINDOW_NORMAL)

        cv2.imshow("calibrate",overall_bg)
        
        
    cv2.destroyAllWindows()

if __name__=="__main__":mover_image(images,dia=52.66)
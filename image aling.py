import cv2
import numpy as np
def zoom_at(img, zoom, coord=None):
   
    h, w, _ = [ zoom * i for i in img.shape ]
    
    if coord is None: cx, cy = w/2, h/2
    else: cx, cy = [ zoom*c for c in coord ]
    if zoom<1:
        img = cv2.resize( img, (int(img.shape[1]*zoom),int(img.shape[0]*zoom)))

    else:
        img = cv2.resize( img, (0, 0), fx=zoom, fy=zoom)
        img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
                int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
                : ]
    
    return img
def move_ver_image(img,ver=0,hori=0,height=None,width=None):
    pad=50
    if height==None and width==None:
        height,width=img.shape[:2]
    h,w=img.shape[:2]
    bg_img= cv2.copyMakeBorder(np.zeros((height,width,3),np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, value = 0)
    # if ver>pad:ver=pad
    # if ver<-pad:ver=-pad
    # if hori>pad:hori=pad
    # if hori<-pad:hori=-pad
    if ver<0 or hori<0:
      bg_img[0-ver:h-ver,0-hori:w-hori]=img
    else:
      if ver>=0 and hori>=0:
         bg_img[0+ver:h+ver,0+hori:w+hori]=img


    return bg_img
image=cv2.imread("final23.jpg")
h,w=image.shape[:2]
n=1
ver=0
hori=0
pad=50
while True:
    try:
        bg_img= move_ver_image(zoom_at(image,zoom=n),ver=ver,hori=hori,height=h,width=w)
        bg_h,bg_w=bg_img.shape[:2]
        cv2.imshow("image",image)
        cv2.imshow("vertical_move",bg_img)

        # cv2.imshow("zoomed",zoom_at(image,zoom=n))
        k=cv2.waitKey(1)


        ###  vertical move
        if k==ord("w"):
            if ver<0: ver+=10
        if k==ord('s'):
            ver-=10


    ###  horizontal move
        if k==ord("a"):
            if hori<0:hori+=10
        if k==ord('d'):
            hori-=10
    

        ## zoom

        if k==ord('i'):
            n+=0.1
        if k==ord('o'):
            n-=0.1
        if n<0:
            n=0.1
    except:
        n=1
        ver=0
        hori=0


    if k==ord("q"):break
cv2.destroyAllWindows()
import cv2 
import numpy as np
import json



def calibrated_image(dia,images):
    with open(f'stich{dia}.json', 'r') as openfile:
        ori= json.load(openfile)
    img1,img2,img3,img4=images
    bg= cv2.copyMakeBorder(np.zeros((ori["bg_shape"][1],ori["bg_shape"][2],3),np.uint8), ori["bg_shape"][0], ori["bg_shape"][0], ori["bg_shape"][0], ori["bg_shape"][0], cv2.BORDER_CONSTANT, None, value = 0)

    bg[ori["img1"][0] :ori["img1"][1],ori["img1"][2]:ori["img1"][3]]=img1
    bg[ori["img2"][0] :ori["img2"][1],ori["img2"][2]:ori["img2"][3]]=img2
    bg[ori["img3"][0] :ori["img3"][1],ori["img3"][2]:ori["img3"][3]]=img3
    bg[ori["img4"][0] :ori["img4"][1],ori["img4"][2]:ori["img4"][3]]=img4


    final_img=bg[max(ori["img1"][0],ori["img2"][0],ori["img3"][0],ori["img4"][0]):
                min(ori["img1"][1],ori["img2"][1],ori["img3"][1],ori["img4"][1]),
                ori["img1"][2]:ori["img4"][3]]
    return final_img
   

if __name__=="__main__":
    dia=52.66
    images=[]

    images.append(cv2.imread("final20.jpg"))
    images.append(cv2.imread("final21.jpg"))
    images.append(cv2.imread("final22.jpg"))
    images.append(cv2.imread("final23.jpg"))
    final_img=calibrated_image(dia,images)
    cv2.namedWindow("final",cv2.WINDOW_NORMAL)

    cv2.imshow("final",final_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

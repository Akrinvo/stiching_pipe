import cv2
import numpy as np
img1=cv2.imread("final22.jpg")
img2=cv2.imread("final23.jpg")
img2=cv2.resize(img2,(img1.shape[1],img1.shape[0]))
img12=np.hstack((img1,img2))

print(img1.shape)
print(img2.shape)
# generate Gaussian pyramid for img1
img1_copy = img1.copy()
gp_img1 = [img1_copy]
for i in range(6):
    img1_copy = cv2.pyrDown(img1_copy)
    gp_img1.append(img1_copy)

# generate Gaussian pyramid for img2
img2_copy = img2.copy()
gp_img2 = [img2_copy]
for i in range(6):
    img2_copy = cv2.pyrDown(img2_copy)
    gp_img2.append(img2_copy)

# generate Laplacian Pyramid for img1
img1_copy = gp_img1[5]
lp_img1 = [img1_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_img1[i])
  
    Gp_img=cv2.resize(gp_img1[i-1],(gaussian_expanded.shape[1],gaussian_expanded.shape[0]))
    laplacian = cv2.subtract(Gp_img, gaussian_expanded)
    lp_img1.append(laplacian)

# generate Laplacian Pyramid for img2
img2_copy = gp_img2[5]
lp_img2 = [img2_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_img2[i])
    Gp_img=cv2.resize(gp_img2[i-1],(gaussian_expanded.shape[1],gaussian_expanded.shape[0]))
    laplacian = cv2.subtract(Gp_img, gaussian_expanded)
    lp_img2.append(laplacian)



# Now add left and right halves of images in each level
img1_img2_pyramid = []
n = 0
for img1_lap, img2_lap in zip(lp_img1, lp_img2):
    n += 1
    cols, rows, ch = img1_lap.shape
    laplacian = np.hstack((img1_lap, img2_lap))
    img1_img2_pyramid.append(laplacian)
# now reconstruct
img1_img2_reconstruct = img1_img2_pyramid[0]
for i in range(1, 6):
    img1_img2_reconstruct = cv2.pyrUp(img1_img2_reconstruct)
    Gp_img=cv2.resize(img1_img2_pyramid[i],(img1_img2_reconstruct.shape[1],img1_img2_reconstruct.shape[0]))

    img1_img2_reconstruct = cv2.add(Gp_img, img1_img2_reconstruct)


cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow("img1_img2_reconstruct", img1_img2_reconstruct)

cv2.imshow("imag12",img12)
cv2.waitKey(0)
cv2.destroyAllWindows()
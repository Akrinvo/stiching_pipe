import cv2

# define a function to display the coordinates of

# of the points clicked on the image
points=[]
def click_event(event, x, y, flags, params):
   global points
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      if len(points)<6:
           points.append([x,y])
      
      # put coordinates as text on the image
      cv2.putText(img, f'({x},{y})',(x,y),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
      # draw point on the image
      cv2.circle(img, (x,y), 3, (0,255,255), -1)
 
# read the input image
img = cv2.imread('final22.jpg')

# create a window
cv2.namedWindow('Point Coordinates')

# bind the callback function to window
cv2.setMouseCallback('Point Coordinates', click_event)

# display the image
while True:
   print(points)
   if len(points)<6:
     cv2.imshow('Point Coordinates',img)
   else: break
   k = cv2.waitKey(1) & 0xFF
   if k == ord("q"):
            break
cv2.destroyAllWindows()
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('./data1/samples/dog.jpg')
#cv2.circle(img,(129, 222),3,(0,0,255),4)
#cv2.circle(img,(314, 522),3,(0,0,255),4)
cropped = img[222:522,129:314]
plt.figure("img")
plt.imshow(cropped)
plt.show()
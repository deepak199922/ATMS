
import pytesseract
import cv2 
import imutils
import numpy as np
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
image=cv2.imread('asa.jpeg')  #read image
plt.imshow(image)
image=imutils.resize(image,width=500)  #resize

plt.imshow(image)

while True: #display original image
    cv2.imshow('created',image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()

gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)


gray = cv2.bilateralFilter(gray, 11, 17, 17) #noise removal while percieving edge

#find edges of grayscale

edge = cv2.Canny(gray,170,200)
plt.imshow(edge)

#contours - continuous shapes in image

cnts, new = cv2.findContours(edge.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
img1=image.copy()
cv2.drawContours(img1, cnts, -1,(0,255,0),3)

#sort contours based on their area min area=30
cnts=sorted(cnts,key=cv2.contourArea, reverse=True)[:30]
NumberPlateCnt=None
# Top 30 cnts
img2=image.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3)
plt.imshow(img2)
#best possible approximate contours for number plate

count=0
i=7

for c in cnts:
  peri=cv2.arcLength(c,True)
  approx=cv2.approxPolyDP(c,0.02*peri,True)  #number of edges
  if len(approx)==4:           #number palte has 4 corners
    NumberPlateCnt=approx

    #crop the contours
    x,y,w,h=cv2.boundingRect(c)  #find rect for plate
    new_img=image[y:y+h,x:x+w] #create new image
    cv2.imwrite('/'+str(i) + '.jpg',new_img)#store new image
    i+=1

    break  
#draw selected contour on car
cv2.drawContours(image,[NumberPlateCnt],-1,(0,255,0),3)
plt.imshow(new_img)



while True: #display original image
    cv2.imshow('created',im2)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()



dst = cv2.fastNlMeansDenoisingColored(new_img,None,10,10,7,21)


plt.imshow(dst)

text=pytesseract.image_to_string(dst,lang='eng')
print('Number is : ',text)


im=cv2.imread('ad.PNG')  #read image



text=pytesseract.image_to_string(im,lang='eng')
print('Number is : ',text)







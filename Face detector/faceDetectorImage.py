import cv2
from random import randrange

trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('grpPeople.jpg')
img = cv2.resize(img,(900,600))

greyScaledImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faceCoordinates = trainedFaceData.detectMultiScale(greyScaledImg)

for (x,y,w,h) in faceCoordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(128,256), randrange(128,256), randrange(128,256)), 2)

cv2.imshow('Face Detector', img)
cv2.waitKey()

print('Code Completed!')
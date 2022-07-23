import cv2
from random import randrange

# Machine learning data for face detection downloaded from openCV
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# input of the image used for detection
img = cv2.imread('Me2.jpg')
img = cv2.resize(img,(950,650))

greyScaledImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# getting face coordinates
faceCoordinates = trainedFaceData.detectMultiScale(greyScaledImg)

# drawing the square
for (x,y,w,h) in faceCoordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(128,256), randrange(128,256), randrange(128,256)), 2)

cv2.imshow('Face Detector', img)
cv2.waitKey()

print('Code Completed!')
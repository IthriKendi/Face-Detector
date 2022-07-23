import cv2

# Machine learning data for face detection downloaded from openCV
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# input of the video used for detection
webcam = cv2.VideoCapture(0)

while True:
    
    successfulFrameRead, frame = webcam.read()
    
    greyScaledImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # getting face coordinates
    faceCoordinates = trainedFaceData.detectMultiScale(greyScaledImg)
    
    # drawing of the square
    for (x,y,w,h) in faceCoordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)
    
    # click the key 'q' or 'Q' to quit the program
    if key == 81 or key == 113:
        break

webcam.release()
print('Code Completed!')

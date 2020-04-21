from keras.models import load_model
import operator
import cv2
import sys, os

model=load_model("model.h5")

cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (128, 128)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 225, 127,cv2.THRESH_TOZERO_INV)
    cv2.imshow("test", test_image)
    
    result = model.predict(test_image.reshape(1, 128, 128, 1))
    prediction = {'ZERO RIGHT': result[0][0], 
                  'ONE RIGHT': result[0][1], 
                  'TWO RIGHT': result[0][2],
                  'THREE RIGHT': result[0][3],
                  'FOUR RIGHT': result[0][4],
                  'FIVE RIGHT': result[0][5],
                  'ZERO LEFT':result[0][6],
                  'ONE LEFT':result[0][7],
                  'TWO LEFT':result[0][8],
                  'THREE LEFT':result[0][9],
                  'FOUR LEFT':result[0][10],
                  'FIVE LEFT':result[0][11]}
    
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()
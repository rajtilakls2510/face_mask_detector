import cv2
from tensorflow.keras.models import load_model
import numpy as np


model = load_model("D:\CampusX Mentorship Programme\Face Mask Detector\Models\mobilefacenet_99.61.h5")
class_labels={0:'with_mask', 1:'without_mask'}

def predict(images):
    prediction = model.predict(images)
    return prediction



cap = cv2.VideoCapture(0)
face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
i=0


while True:
    ret, frame = cap.read()

    i+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    detected_faces=[]

    for (x,y,w,h) in faces:

        detected_face = frame[y:y+h, x:x+w]
        detected_face = cv2.resize(detected_face, (224,224))
        detected_faces.append(detected_face)

    detected_faces = np.array(detected_faces)

    if len(detected_faces)>0:


        predictions = predict(detected_faces)

        for (x,y,w,h), prediction_precentage in zip(faces, predictions):

            prediction= np.argmax(prediction_precentage)
            colour = (255,0,0)
            label = class_labels[prediction] + " " + str(np.round(prediction_precentage[prediction]*100,2)) + " %"
            if prediction==1:
                # Without mask
                colour= (0,0,255)
            else:
                # With mask
                colour = (0,255,0)
            frame = cv2.rectangle(frame,(x,y), (x+w,y+h),colour, 2)
            frame = cv2.putText(frame,label,(x-2,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.7, colour, thickness=2)



    cv2.imshow("window", frame)


    if cv2.waitKey(1) & 0xFF==ord('x'):
        break

cv2.destroyAllWindows()
cap.release()

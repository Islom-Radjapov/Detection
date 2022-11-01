import time

from keras import models
from keras.utils import img_to_array
import numpy as np
import cv2
import cvlib as cv


# load model
model = models.load_model('gender_detection.model')
# open webcam
webcam = cv2.VideoCapture(0)

classes = ['man','woman']

# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()  # frame bu videoni numpy array shakli,  status bu agar camera ishlavotgan bosa True boladi

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):
        # time.sleep(.5)

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle (frame, (startX, startY), (endX, endY), (0,0,0), 2)

        # crop the detected face region
        face_crop = np.copy( frame[startY:endY,startX:endX] )

        # if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
        #     continue

        # preprocessing for gender detection model
        face_crop = cv2.resize( face_crop, (96, 96) )
        print(face_crop)
        face_crop = face_crop.astype("float") / 255.0
        print("----------")
        print(face_crop)
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) == ord('q'):
        break
    break

# release resources
webcam.release()
cv2.destroyAllWindows()


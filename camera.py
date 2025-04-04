import cv2
import numpy
from model import predict_emotion

cam = cv2.VideoCapture(0)

window_name = "test"
cv2.namedWindow(window_name)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        img_name = "stn_visualization_{}.png".format(img_counter)
        emotion, stn_visualization = predict_emotion(frame)

        stn_visualization_to_save = numpy.array(
            stn_visualization)[:, :, ::-1].copy()
        cv2.imwrite(img_name, stn_visualization_to_save)
        print("Emotion detected: #%d %s" % (img_counter, emotion))
        img_counter += 1

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray_image, 1.1, 5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow(window_name, frame)

cam.release()

cv2.destroyAllWindows()

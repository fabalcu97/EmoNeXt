import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import set_target_layers, predict_emotion, target_layers_map

cam = cv2.VideoCapture(0)

window_name = "EmoNeXt + CBAM + GradCam - Emotion Recognition"
cv2.namedWindow(window_name)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Closing...")
        break

    if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        layer_num = str(int(chr(key)) - 1)
        print(f"Toggling target layer number {layer_num}")
        set_target_layers(layer_num)
        print(
            f"Layers {layer_num} is " +
            f"{target_layers_map[layer_num]['isEnabled'] and 'enabled' or 'disabled'}")

    ret, video_frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray_image, 1.5, 5, minSize=(40, 40))

    overlay_frame = video_frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        crop_img = overlay_frame[y: y + h, x: x + w]
        emotion, grad_cam_mask = predict_emotion(
            crop_img)

        if grad_cam_mask is not None:
            grad_cam_mask = np.uint8(255 * grad_cam_mask)
            heatmap = cv2.applyColorMap(grad_cam_mask, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (w, h))

            face_with_heatmap = cv2.addWeighted(crop_img, 0.5, heatmap, 0.5, 0)
            overlay_frame[y: y + h, x: x + w] = face_with_heatmap

        cv2.putText(overlay_frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow(window_name, overlay_frame)

cam.release()

cv2.destroyAllWindows()

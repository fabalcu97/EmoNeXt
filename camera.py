import cv2
import numpy as np

from trained_model import set_target_layers, predict_emotion, target_layers_map

cam = cv2.VideoCapture(0)

window_name = "EmoNeXt + CBAM + GradCam - Emotion Recognition"
cv2.namedWindow(window_name)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

emotion_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def draw_emotion_probabilities(frame, probabilities, x, y, w, h):
    """
    Draw emotion probability bars on the frame.

    Args:
        frame: The frame to draw on
        probabilities: Tensor of emotion probabilities
        x, y, w, h: Face bounding box coordinates
    """
    if probabilities is None:
        return

    # Convert probabilities to percentages
    probs_percent = probabilities * 100

    # Configuration for the probability display
    bar_width = 150
    bar_height = 15
    bar_spacing = 20
    text_offset_x = 5
    text_offset_y = 12

    # Starting position for the probability bars (to the right of the face)
    start_x = x + w + 10
    start_y = y

    # Color map for different emotions (BGR format)
    emotion_colors = {
        "Angry": (0, 0, 255),        # Red
        "Disgust": (0, 128, 0),      # Dark Green
        "Fear": (128, 0, 128),       # Purple
        "Happy": (0, 255, 255),      # Yellow
        "Neutral": (128, 128, 128),  # Gray
        "Sad": (255, 0, 0),          # Blue
        "Surprise": (0, 165, 255)    # Orange
    }

    for i, (emotion_name, prob) in enumerate(zip(emotion_names, probs_percent)):
        # Calculate bar position
        bar_y = start_y + (i * bar_spacing)

        # Draw background bar (dark gray)
        cv2.rectangle(frame,
                      (start_x, bar_y),
                      (start_x + bar_width, bar_y + bar_height),
                      (50, 50, 50), -1)

        # Draw filled bar based on probability
        fill_width = int((prob / 100.0) * bar_width)
        if fill_width > 0:
            color = emotion_colors.get(emotion_name, (255, 255, 255))
            cv2.rectangle(frame,
                          (start_x, bar_y),
                          (start_x + fill_width, bar_y + bar_height),
                          color, -1)

        # Draw text with emotion name and percentage
        text = f"{emotion_name}: {prob:.1f}%"
        cv2.putText(frame, text,
                    (start_x + text_offset_x, bar_y + text_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Closing...")
        break

    if key in [ord("1"), ord("2"), ord("3"), ord("4")]:
        layer_num = str(int(chr(key)) - 1)
        print(f"Toggling target layer number {layer_num}")
        set_target_layers(layer_num)
        print(
            f"Layers {layer_num} is "
            + f"{target_layers_map[layer_num]['isEnabled'] and 'enabled' or 'disabled'}"
        )

    ret, video_frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.5, 5, minSize=(40, 40))

    overlay_frame = video_frame.copy()
    for x, y, w, h in faces:
        cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        crop_img = overlay_frame[y:y + h, x:x + w]
        emotion, grad_cam_mask, probabilities = predict_emotion(crop_img)

        if grad_cam_mask is not None:
            grad_cam_mask = np.uint8(255 * grad_cam_mask)
            heatmap = cv2.applyColorMap(grad_cam_mask, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (w, h))

            face_with_heatmap = cv2.addWeighted(crop_img, 0.5, heatmap, 0.5, 0)
            overlay_frame[y:y + h, x:x + w] = face_with_heatmap

        # Display main emotion
        cv2.putText(
            overlay_frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        # Display probability bars for all emotions
        draw_emotion_probabilities(overlay_frame, probabilities, x, y, w, h)

    cv2.imshow(window_name, overlay_frame)

cam.release()

cv2.destroyAllWindows()


import os

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotions_count = {
    "Angry": 0,
    "Disgust": 0,
    "Fear": 0,
    "Happy": 0,
    "Sad": 0,
    "Surprise": 0,
    "Neutral": 0
}
datasets_path = os.path.join(os.path.dirname(__file__), "../datasets")
datasets = os.listdir(datasets_path)

for dataset in datasets:
    dataset_path = os.path.join(datasets_path, dataset)
    if not os.path.isdir(dataset_path):
        continue

    stages = ["train", "valid", "test"]
    total_count = 0
    print(f"Counting images in dataset: {dataset}")
    for stage in stages:
        stage_path = os.path.join(dataset_path, stage)
        images_count = 0
        for emotion in emotions:
            emotion_path = os.path.join(stage_path, emotion)
            count = len(os.listdir(emotion_path)) if os.path.exists(emotion_path) else 0
            emotions_count[emotion] += count
            images_count += count
        total_count += images_count
        print(f"{stage}: {images_count} images")
    print(emotions_count)
    print(f"Total: {total_count} images")
    print("-" * 40)

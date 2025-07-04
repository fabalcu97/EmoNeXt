import os
import random
import matplotlib.pyplot as plt
from PIL import Image

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# Spanish labels for plotting
spanish_emotions = ["Enojo", "Disgusto", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Neutral"]
datasets_path = os.path.join(os.path.dirname(__file__), "../datasets")
datasets = os.listdir(datasets_path)

for dataset in datasets:
    if dataset.startswith('.'):
        continue
    dataset_train_path = os.path.join(datasets_path, dataset, "train")
    # Pick 2 random samples from each emotion folder and generate a sample grid image
    samples = []
    found_emotions = []
    for emotion in emotions:
        emotion_dir = os.path.join(dataset_train_path, emotion)
        if not os.path.isdir(emotion_dir):
            print(f"Warning: emotion folder {emotion_dir} not found.")
            continue
        found_emotions.append(emotion)
        imgs = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not imgs:
            print(f"No images found in {emotion_dir}.")
            continue
        selected = random.sample(imgs, min(2, len(imgs)))
        for img_name in selected:
            samples.append((emotion, os.path.join(emotion_dir, img_name)))

    # Create grid: 2 rows (samples) and one column per emotion
    n_rows = 2
    n_cols = len(found_emotions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    for col, emotion in enumerate(found_emotions):
        emo_imgs = [p for emo, p in samples if emo == emotion]
        for row in range(n_rows):
            ax = axes[row, col]
            ax.axis('off')
            if row < len(emo_imgs):
                img = Image.open(emo_imgs[row]).convert('RGB')
                ax.imshow(img)
        # Label each column with the emotion name in Spanish
        axes[0, col].set_title(spanish_emotions[col])

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "images", f"{dataset}_sample.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved sample grid for {dataset} to {output_path}")

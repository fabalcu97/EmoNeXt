import os
import random
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision import transforms

from train.utils import get_device
from trained_model import get_model

device = get_device()

# spanish_emotions = ["Enojo", "Disgusto", "Miedo", "Felicidad", "Neutral", "Tristeza", "Sorpresa"]
# emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Gimefive dataset has only 6 emotions, so we adjust the list accordingly
spanish_emotions = ["Enojo", "Disgusto", "Miedo", "Felicidad", "Tristeza", "Sorpresa"]
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

spanish_emotions_map = {
    "Angry": "Enojo",
    "Disgust": "Disgusto",
    "Fear": "Miedo",
    "Happy": "Felicidad",
    "Neutral": "Neutral",
    "Sad": "Tristeza",
    "Surprise": "Sorpresa"
}

test_transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize(236),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        lambda x: x.repeat(3, 1, 1)
    ]
)

configurations = [
    # {"model_configs": {"use_cbam": False, }, "dataset": "fer2013", "model_name": "base_baseline_fer2013.pt"},
    # {"model_configs": {"use_cbam": True, }, "dataset": "fer2013", "model_name": "base_cbam_fer2013.pt"},
    # {"model_configs": {"use_cbam": False, }, "dataset": "rafdb", "model_name": "base_baseline_rafdb.pt"},
    # {"model_configs": {"use_cbam": True, }, "dataset": "rafdb", "model_name": "base_cbam_rafdb.pt"},
    # {"model_configs": {"use_cbam": False, }, "dataset": "ckplus", "model_name": "base_baseline_fer2013.pt"},
    # {"model_configs": {"use_cbam": True, }, "dataset": "ckplus", "model_name": "base_cbam_fer2013.pt"},
    # {"model_configs": {"use_cbam": False, }, "dataset": "gimefive", "model_name": "base_baseline_gimefive.pt"},
    {"model_configs": {"use_cbam": True, }, "dataset": "gimefive", "model_name": "base_cbam_gimefive.pt"},
]

amount_of_emotions = 10

for config in configurations:
    model_name = config["model_name"]
    dataset = config["dataset"]
    model_configs = config["model_configs"]

    test_dataset_path = os.path.join("datasets", dataset + "/test")
    test_dataset = datasets.ImageFolder(test_dataset_path, test_transform,)

    emotions_map = {
        "Angry": random.sample(list(filter(lambda x: x[1] == 0, test_dataset.samples)), amount_of_emotions),
        "Disgust": random.sample(list(filter(lambda x: x[1] == 1, test_dataset.samples)), amount_of_emotions),
        "Fear": random.sample(list(filter(lambda x: x[1] == 2, test_dataset.samples)), amount_of_emotions),
        "Happy": random.sample(list(filter(lambda x: x[1] == 3, test_dataset.samples)), amount_of_emotions),
        # "Neutral": random.sample(list(filter(lambda x: x[1] == 4, test_dataset.samples)), amount_of_emotions),
        # "Sad": random.sample(list(filter(lambda x: x[1] == 5, test_dataset.samples)), amount_of_emotions),
        # "Surprise": random.sample(list(filter(lambda x: x[1] == 6, test_dataset.samples)), amount_of_emotions)
        # Gimefive
        "Sad": random.sample(list(filter(lambda x: x[1] == 4, test_dataset.samples)), amount_of_emotions),
        "Surprise": random.sample(list(filter(lambda x: x[1] == 5, test_dataset.samples)), amount_of_emotions)
    }

    if not model_name.endswith(".pt"):
        continue
    print(f"Testing model: {model_name} for dataset: {dataset}")
    model_path = os.path.join("latest", model_name)
    model = get_model(model_path, configs=model_configs)
    model.eval()

    for emotion, images in emotions_map.items():
        file_name = f"{'_'.join([dataset, model_name.split('.')[0].split('_')[1], spanish_emotions_map.get(emotion, emotion)])}.png"
        fig, axes = plt.subplots(1, amount_of_emotions, figsize=((amount_of_emotions*1.3), 2))
        plt.suptitle(f"Emoción real: {spanish_emotions_map.get(emotion, emotion)}", y=.95, fontsize=12)
        for idx, img_info in enumerate(images):
            img = test_dataset.loader(img_info[0])
            axes[idx].imshow(img)
            img = test_transform(img).unsqueeze(0).to(device)
            emotion_index, probs = model(img)
            emotion_name = spanish_emotions[emotion_index]
            # emotion_name = spanish_emotions_map.get(emotion, emotion)

            axes[idx].set_title(f"{emotion_name}", fontsize=12, y=-0.3)
            axes[idx].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 1.15])
        plt.savefig(f"images/{file_name}")

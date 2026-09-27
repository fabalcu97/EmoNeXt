import os
from matplotlib.pylab import f
import pandas as pd
from torchvision import datasets
from torchvision import transforms
from tqdm.contrib import tenumerate

from train.utils import get_device
from trained_model import get_model

device = get_device()
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
# emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
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
    {"model_configs": {"use_cbam": False, }, "dataset": "fer2013", "model_name": "base_baseline_fer2013.pt"},
    {"model_configs": {"use_cbam": True, }, "dataset": "fer2013", "model_name": "base_cbam_fer2013.pt"},
    # {"model_configs": {"use_cbam": False, }, "dataset": "rafdb", "model_name": "base_baseline_rafdb.pt"},
    # {"model_configs": {"use_cbam": True, }, "dataset": "rafdb", "model_name": "base_cbam_rafdb.pt"},
    # {"model_configs": {"use_cbam": False, }, "dataset": "ckplus", "model_name": "base_baseline_fer2013.pt"},
    # {"model_configs": {"use_cbam": True, }, "dataset": "ckplus", "model_name": "base_cbam_fer2013.pt"},
    # {"model_configs": {"use_cbam": False, }, "dataset": "gimefive", "model_name": "base_baseline_gimefive.pt"},
    # {"model_configs": {"use_cbam": True, }, "dataset": "gimefive", "model_name": "base_cbam_gimefive.pt"},
]


for config in configurations:
    model_name = config["model_name"]
    dataset = config["dataset"]
    model_configs = config["model_configs"]

    test_dataset_path = os.path.join("datasets", dataset + "/test")
    test_dataset = datasets.ImageFolder(test_dataset_path, test_transform)

    if not model_name.endswith(".pt"):
        continue
    print(f"Testing model: {model_name}")
    model_path = os.path.join("latest", model_name)
    model = get_model(model_path, configs=model_configs)
    model.eval()

    predicted_emotions = []
    actual_emotions = []
    for idx, (img, label) in tenumerate(test_dataset, unit="image", desc="Testing: "):
        emotion_index, probs = model(img.unsqueeze(0).to(device))
        predicted_emotions.append(emotions[emotion_index])
        actual_emotions.append(emotions[label])

    file_name = f"{'_'.join([model_name.split('.')[0].split('_')[1], dataset])}.csv"
    # f"{'_'.join(model_name.split('.')[0].split('_')[1:])}.csv"
    pd.DataFrame({
        "actual": actual_emotions,
        "predicted": predicted_emotions
    }).to_csv(f"metrics/{file_name}", index=False)

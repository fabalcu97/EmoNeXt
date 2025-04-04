import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np

from models import EmoNeXt

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")


# def get_model(num_classes, model_size="tiny",):
#     if model_size == "tiny":
#         depths = [3, 3, 9, 3]
#         dims = [96, 192, 384, 768]

#     elif model_size == "small":
#         depths = [3, 3, 27, 3]
#         dims = [96, 192, 384, 768]

#     elif model_size == "base":
#         depths = [3, 3, 27, 3]
#         dims = [128, 256, 512, 1024]

#     elif model_size == "large":
#         depths = [3, 3, 27, 3]
#         dims = [192, 384, 768, 1536]

#     else:
#         depths = [3, 3, 27, 3]
#         dims = [256, 512, 1024, 2048]

depths = [3, 3, 9, 3]
dims = [96, 192, 384, 768]
emoxnet = EmoNeXt(
    depths=depths, dims=dims, num_classes=6, drop_path_rate=0.1
)

data = torch.load("./outputs/2025-03-29_ACC_81_5.pt",
                  weights_only=True, map_location=device)
emoxnet.load_state_dict(data["model"])


def repeat_tensor(x):
    return x.repeat(3, 1, 1)


transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize(236),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        repeat_tensor,
    ]
)


base_path = '/Users/fabalcu97/Programming/University/EmoNeXt/custom_images/'

sadness = os.path.join(base_path, 'sadness.jpg')
happiness = os.path.join(base_path, 'happiness.jpg')
anger = os.path.join(base_path, 'anger.jpg')
disgust = os.path.join(base_path, 'disgust.jpg')
fear = os.path.join(base_path, 'fear.jpg')
surprise = os.path.join(base_path, 'surprise.jpg')

image_path = happiness


# emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
emotions = ['anger', ]
emoxnet.eval()

to_pil = transforms.ToPILImage()

for emotion in emotions:

    image = Image.open(os.path.join(
        base_path, emotion + '.jpg')).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    image_array = np.array(image)
    emotion_index, probabilities = emoxnet(image_tensor)

    stn_result = emoxnet.stn(image_tensor)
    stn_batch = to_pil(utils.make_grid(
        stn_result, nrow=16, padding=4))
    Image.Image.save(stn_batch, "stn.jpg")

    print("Emotion: %s | Predicted emotion: %s" %
          (emotion, emotions[emotion_index]))

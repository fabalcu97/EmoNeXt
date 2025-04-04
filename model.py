import torch
from torchvision import transforms, utils
from PIL import Image

from models import EmoNeXt

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']


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
to_pil = transforms.ToPILImage()


def get_model():

    depths = [3, 3, 9, 3]
    dims = [96, 192, 384, 768]
    emoxnet = EmoNeXt(
        depths=depths, dims=dims, num_classes=6, drop_path_rate=0.1
    )

    data = torch.load("./outputs/2025-03-29_ACC_81_5.pt",
                      weights_only=True, map_location=device)
    emoxnet.load_state_dict(data["model"])

    emoxnet.eval()
    return emoxnet


emoxnet = get_model()


def predict_emotion(video_frame):

    image = Image.fromarray(video_frame)
    frame_tensor = transform(image).unsqueeze(0).to(device)
    emotion_index, _ = emoxnet(frame_tensor)

    stn_result = emoxnet.stn(frame_tensor)
    stn_visualization = to_pil(utils.make_grid(
        stn_result, nrow=16, padding=4)).convert("RGB")

    return emotions[emotion_index], stn_visualization

import os
import cv2
import numpy
import torch
from torchvision import transforms, utils
from PIL import Image
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
    ClassifierOutputSoftmaxTarget,
)
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.image import show_cam_on_image

from models import EmoNeXt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")

# emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
# emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]  # all models
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"] # gimefive 🤦
num_classes = len(emotions)


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


def get_model(model_path, configs={}):
    # Base
    depths = [3, 3, 27, 3]
    dims = [128, 256, 512, 1024]
    emoxnet = EmoNeXt(
        depths=depths,
        dims=dims,
        num_classes=num_classes,
        drop_path_rate=0.1,
        use_cbam=configs.get("use_cbam", False),
    ).to(device)

    data = torch.load(
        os.path.join(os.path.dirname(__file__), 'out', model_path),
        weights_only=True,
        map_location=device,
    )
    emoxnet.load_state_dict(data["model"])

    emoxnet.eval()
    return emoxnet


# emonext = get_model("latest/base_cbam_p15_dpr.pt")

# targets = [ClassifierOutputSoftmaxTarget(i) for i in range(num_classes)]
# cam = GradCAM(model=emonext, target_layers=target_layers)

# targets = [ClassifierOutputTarget(i) for i in range(num_classes)] <-- uncomment this!

# target_layers_map = {
#     "0": {"isEnabled": False, "layer": emonext.stages[0]},
#     "1": {"isEnabled": False, "layer": emonext.stages[1]},
#     "2": {"isEnabled": False, "layer": emonext.stages[2]},
#     "3": {"isEnabled": False, "layer": emonext.stages[3]},
# }


# def set_target_layers(idx):
#     """Get the layers to be used for Grad-CAM by toggling the index of the layer to be enabled."""
#     target_layers_map[idx]["isEnabled"] = not target_layers_map[idx]["isEnabled"]


# def predict_emotion(video_frame):
#     target_layers = []
#     get_grad_cam_mask = False
#     for layer_index, layer in target_layers_map.items():
#         if layer["isEnabled"]:
#             target_layers.append(layer["layer"])
#     if len(target_layers):
#         get_grad_cam_mask = True
#     with GradCAM(model=emonext, target_layers=target_layers) as cam:
#         image = Image.fromarray(video_frame)
#         frame_tensor = transform(image).unsqueeze(0).to(device)
#         emotion_index, probs = emonext(frame_tensor)

#         grad_cam_mask = None
#         if get_grad_cam_mask:
#             grad_cam_mask = cam(input_tensor=frame_tensor, targets=targets)
#             grad_cam_mask = grad_cam_mask[0, :]

#         # rgb_img = frame_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
#         # cam_image = show_cam_on_image(
#         #     rgb_img, grad_cam_mask, use_rgb=True)
#         # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
#         # cv2.imwrite("images/grad_cam.png", cam_image)

#         normalized_probs = torch.nn.functional.softmax(probs, dim=1)
#         normalized_probs = normalized_probs.cpu().detach().numpy()[0]

#         return emotions[emotion_index], grad_cam_mask, normalized_probs

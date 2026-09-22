import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from .config import settings


def resolve_device() -> str:
    """Choose the configured device, or the safest available default."""

    if settings.device != "auto":
        return settings.device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


device = resolve_device()

model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(
    torch.load(
        settings.model_checkpoint,
        map_location=device
    )
)

model = model.to(device)
model.eval()


def predict_mask(img):

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img_resized = cv2.resize(
        img_rgb,
        (512,512)
    )

    x = img_resized.astype(np.float32) / 255.0

    x = torch.tensor(
        x.transpose(2,0,1)
    ).unsqueeze(0)

    x = x.to(device)

    with torch.no_grad():

        pred = model(x)

        pred = torch.sigmoid(pred)

    pred = pred.squeeze().cpu().numpy()

    pred_binary = (
        pred > settings.prediction_threshold
    ).astype(np.uint8)

    return pred_binary

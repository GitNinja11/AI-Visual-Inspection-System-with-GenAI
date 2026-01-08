import torch
from torchvision import models, transforms
from PIL import Image

# Load pretrained ResNet
model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_defect(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    confidence = torch.softmax(outputs, dim=1).max().item()

    # Fake defect logic for MVP
    if confidence < 0.55:
        return "Defective", confidence
    else:
        return "Normal", confidence

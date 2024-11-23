import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained model
model = models.resnet50(pretrained=True)
model = model.to('cuda')  # Move model to GPU
model.eval()  # Set to evaluation mode

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to('cuda')  # Add batch dimension and move to GPU
    return image

def classify_image(image_path):
    image = process_image(image_path)
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(image)  # Get model output
    return output

def decode_output(output):
    _, predicted_class = torch.max(output, 1)
    class_idx = predicted_class.item()
    return class_idx  # Return the predicted class index

def main(image_path):
    output = classify_image(image_path)
    class_idx = decode_output(output)
    print(f"Predicted Class Index: {class_idx}")

if __name__ == '__main__':
    image_path = 'image.jpg'  # Make sure to change this to your image's path
    main(image_path)

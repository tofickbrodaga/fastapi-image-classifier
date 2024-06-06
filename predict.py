import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.models import resnet101
import os
import pandas as pd

images_dir = 'archive_food/images'

dataframe = []
for root, _, files in os.walk(images_dir):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            file_path = os.path.join(root, file)
            label = os.path.basename(root)
            formatted_path = os.path.join(label, file)
            dataframe.append({'path': formatted_path, 'label': label})


class Food101(Dataset):
    def __init__(self, dataframe, base_dir, transform=None):
        self.dataframe = dataframe
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.dataframe.path.iloc[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.label.iloc[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


dataframe = pd.DataFrame(dataframe)
food_dataset = Food101(dataframe, images_dir, transform=transform)
num_classes = len(food_dataset.dataframe['label'].unique())


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet101 = resnet101(pretrained=True)
        for param in self.resnet101.parameters():
            param.requires_grad = False
        self.resnet101.fc = nn.Linear(self.resnet101.fc.in_features,
                                      num_classes)

    def forward(self, x):
        x = self.resnet101(x)
        return x


model = ResNetClassifier(
    num_classes=len(food_dataset.dataframe['label'].unique()))


def classify_image(image_path, model, label_encoder, device):
    # Load and preprocess the input image
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        model.eval()
        output = model(image_tensor)

    # Get predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()

    # Map index to class name
    predicted_label = label_encoder.get_label(predicted_idx)

    return predicted_label


class Label_encoder:
    def __init__(self, labels):
        labels = list(set(labels))
        self.labels = {label: idx for idx, label in enumerate(classes)}

    def get_label(self, idx):
        return list(self.labels.keys())[idx]

    def get_idx(self, label):
        return self.labels[label]


state_dict = torch.load('test_model.pth')
model.load_state_dict(state_dict)
classes = open("archive_food/classes.txt", 'r').read().splitlines()
class_names = sorted((food_dataset.dataframe['label'].unique()))
label_encoder = Label_encoder(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "downloads/applepie.jpg"


def predict_label(image_path: str):
    predicted_label = classify_image(image_path, model, label_encoder, device)
    return predicted_label

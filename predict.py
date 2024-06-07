import contextlib
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet101

images_dir = 'archive_food/images'


dataframe = []
for root, _, files in os.walk(images_dir):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            file_path = os.path.join(root, file)
            label = os.path.basename(root)
            formatted_path = os.path.join(label, file)
            dataframe.append({'path': formatted_path, 'label': label})

dataframe = pd.DataFrame(dataframe)


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

food_dataset = Food101(dataframe, images_dir, transform=transform)
num_classes = len(food_dataset.dataframe['label'].unique())


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        with suppress_stdout():
            self.resnet101 = resnet101(pretrained=True)
        for param in self.resnet101.parameters():
            param.requires_grad = False
        self.resnet101.fc = nn.Linear(self.resnet101.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet101(x)
        return x


class LabelEncoder:
    def __init__(self, labels):
        labels = list(set(labels))
        self.labels = {label: idx for idx, label in enumerate(labels)}
        self.inverse_labels = {idx: label for label, idx in self.labels.items()}

    def get_label(self, idx):
        return self.inverse_labels.get(idx)

    def get_idx(self, label):
        return self.labels.get(label)


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


model = ResNetClassifier(num_classes=num_classes)


state_dict = torch.load('accuracy52.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

classes = food_dataset.dataframe['label'].unique()
label_encoder = LabelEncoder(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classify_image(image_path, model, label_encoder, device):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        output = model(image_tensor)

    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()
    predicted_label = label_encoder.get_label(predicted_idx)

    return predicted_label


def predict_label(image_path: str):
    return classify_image(image_path, model, label_encoder, device)

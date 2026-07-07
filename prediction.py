
print("Program started")

import os
import torch
import torch.nn as nn
import pyttsx3

from PIL import Image
from torchvision import transforms

from grammar import improve_sentence


# CNN Architecture

class Cnn(nn.Module):

    def __init__(self, input_channels, num_classes=29):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128,num_classes)

        )


    def forward(self,x):

        x = self.features(x)
        x = self.classifier(x)

        return x



# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



# Load Model

model = Cnn(
    input_channels=3,
    num_classes=29
)


checkpoint = torch.load(
    "sign_model.pth",
    map_location=device
)


# because your file contains only weights

model.load_state_dict(checkpoint)


model = model.to(device)

model.eval()



# Classes

classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]



# Image preprocessing

transform = transforms.Compose([

    transforms.Resize((128,128)),
    transforms.ToTensor()

])



# Text to Speech

engine = pyttsx3.init()

engine.setProperty(
    "rate",
    150
)

engine.setProperty(
    "volume",
    1.0
)



# Input folder

folder_path = r"C:\Users\Vedika\PROJECTS\input_images"



predicted_text = ""



# Predict every image

for image_name in sorted(os.listdir(folder_path)):


    if image_name.lower().endswith(
        (".jpg",".jpeg",".png")
    ):


        image_path = os.path.join(
            folder_path,
            image_name
        )


        image = Image.open(
            image_path
        ).convert("RGB")


        image = transform(image)

        image = image.unsqueeze(0)

        image = image.to(device)



        with torch.no_grad():

            outputs = model(image)

            predicted_class = outputs.argmax(
                dim=1
            ).item()



        predicted_letter = classes[predicted_class]


        print(
            image_name,
            "->",
            predicted_letter
        )



        if predicted_letter == "space":

            predicted_text += " "


        elif predicted_letter == "del":

            predicted_text = predicted_text[:-1]


        elif predicted_letter == "nothing":

            continue


        else:

            predicted_text += predicted_letter



print("\nCNN Output:")
print(predicted_text)



# Hugging Face grammar correction

final_sentence = improve_sentence(
    predicted_text
)



print("\nLLM Output:")
print(final_sentence)



# Speak

engine.say(
    final_sentence
)

engine.runAndWait()
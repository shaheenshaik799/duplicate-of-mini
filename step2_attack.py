import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder("dataset/test", transform=transform)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64*16*16,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


model = CNNModel().to(device)

model.load_state_dict(torch.load("baseline_model.pth"))

model.eval()

criterion = nn.CrossEntropyLoss()

epsilon = 0.1

correct = 0
total = 0


def fgsm_attack(image, epsilon, data_grad):

    sign = data_grad.sign()

    perturbed = image + epsilon * sign

    perturbed = torch.clamp(perturbed,0,1)

    return perturbed


for image, label in test_loader:

    image = image.to(device)
    label = label.to(device)

    image.requires_grad = True

    output = model(image)

    loss = criterion(output,label)

    model.zero_grad()

    loss.backward()

    data_grad = image.grad.data

    perturbed_image = fgsm_attack(image,epsilon,data_grad)

    output = model(perturbed_image)

    _,pred = torch.max(output,1)

    total += 1

    if pred.item() == label.item():

        correct += 1

accuracy = 100*correct/total

print("Accuracy under FGSM attack:",accuracy,"%")
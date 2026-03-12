import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("dataset/train", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel,self).__init__()

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

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x


model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)


def fgsm_attack(image, epsilon, grad):

    return torch.clamp(image + epsilon*grad.sign(),0,1)


for epoch in range(5):

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        images.requires_grad = True

        outputs = model(images)

        loss = criterion(outputs,labels)

        model.zero_grad()

        loss.backward()

        grad = images.grad.data

        adv_images = fgsm_attack(images,0.1,grad)

        outputs = model(adv_images)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print("Epoch",epoch+1)

torch.save(model.state_dict(),"defense_model.pth")

print("Defense model saved")
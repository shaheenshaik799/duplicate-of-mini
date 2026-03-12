import torch

from models.cnn_model import CNNModel
from utils import get_data_loaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, test_loader = get_data_loaders("dataset")

model = CNNModel().to(device)

model.load_state_dict(torch.load("baseline_model.pth"))

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("Baseline Model Accuracy:", accuracy, "%")
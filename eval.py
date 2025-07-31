import torch
from model.model import CNNModel
from data_loader import get_loader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, test_loader = get_loader()

model = CNNModel().to(device)
model.load_state_dict(torch.load('weights/mnist_cnn.pth'))
model.eval()

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')


images, labels = next(iter(test_loader))
img = images[0]
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Label: {labels[0].item()}")
plt.show()


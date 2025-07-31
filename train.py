import torch 
import torch.nn as nn
import torch.optim as optim

from model import CNNModel
from data_loader import get_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = get_loader()

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 


epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')


torch.save(model.state_dict(), 'mnist_cnn.pth')


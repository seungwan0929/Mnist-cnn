import torch 
import torch.nn as nn
import torch.optim as optim

from model import CNNModel
from data_loader import train_loader
from data_loader import test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))  # 한 배치 가져오기
    img = images[0]  # 첫 이미지
    label = labels[0]
    output = model(img.unsqueeze(0).to(device))
    pred = torch.argmax(output, 1)
    print("Predicted:", pred.item())
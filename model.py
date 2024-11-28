import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score

torch.manual_seed(42)

# Step 1: Load the Fashion-MNIST dataset
transform = transforms.Compose(
    [transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize( [0.5], [0.5])]
    )

# Load training and test datasets
dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

train_size = int(0.8 *len(dataset))
test_size = len(dataset)- train_size

test_set, train_set = torch.utils.data.random_split(dataset, [train_size,test_size ])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)


dataiter = iter(trainloader)
images, labels = next(dataiter)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)  
        self.fc1 = nn.Linear(256 * 1 * 1, 256)  
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128,10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  
        x = self.pool(torch.relu(self.conv2(x)))  
        x = self.pool(torch.relu(self.conv3(x)))  
        x = self.pool(torch.relu(self.conv4(x)))  
        x = x.view(-1, 256 * 1 * 1)  
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x)) 
        x = self.fc3(x)  
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model = CNN().to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs=10
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  
        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader)}")

print("Finished Training")


ground_truth = []
prediction = []
model.eval()

with torch.no_grad():  
    for inputs, labels in testloader:
        inputs = inputs.to(device)

        ground_truth += labels.tolist()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1) 
        prediction+= predicted.cpu().tolist()

accuracy = accuracy_score(ground_truth, prediction)
recall = recall_score(ground_truth, prediction, average='weighted')
precision = precision_score(ground_truth, prediction, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')



dataiter = iter(testloader)
images, labels = next(dataiter)


outputs = model(images)
_, predicted = torch.max(outputs, 1)





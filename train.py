import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from utils import json_loader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, images, keypoints):
        self.images = images
        self.keypoints = keypoints

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.keypoints[index]

class model_resnet50(nn.Module):
    def __init__(self, num_keypoint=133, pretrained=False):
        super(model_resnet50, self).__init__()
        self.encoder = models.resnet50(pretrained=pretrained)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(7 * 7 * 512 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(7 * 7 * 512, 512)
        # self.fc4 = nn.Linear(512, 512)
        self.outlayer1 = nn.Linear(1024, num_keypoint*3)
        # self.outlayer2 = nn.Linear(512, num_keypoint*2)

    def forward(self, x):
        x = (x - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)  # 100352 = 28 x 28 x 128
        x = self.encoder.layer3(x)  # 50176 = 14 x 14 x 256
        x = self.encoder.layer4(x)  # 25088 = 7 x 7 x 512 # 100352
        x = x.reshape(x.shape[0], -1)
        x1 = self.relu(self.fc1(x))
        x1 = self.relu(self.fc2(x1))
        x1 = (self.outlayer1(x1))
        # x2 = self.relu(self.fc3(x))
        # x2 = self.relu(self.fc4(x2))
        # x2 = (self.outlayer2(x2))
        return x1 #, x2
    
    
imgpath = "data/h3wb/images/"
imgresizepath = "data/h3wb/reimages/"


input_list, target_list, _ = json_loader("data/h3wb/annotations",3,'train')

input_list = input_list[:len(input_list)//2]
target_list = target_list[:len(target_list)//2]

num_data = len(input_list)
img_list = []

# Making an actual torch.tensor out of images
transform = transforms.Compose([transforms.ToTensor()])
i = 0
for input in input_list:
    if(i % 200 == 0):
        print(i,'/',len(input_list))
    image_dir = '../data/h3wb/images'
    sample_img = Image.open(os.path.join(imgresizepath, input))
    torch_img = transform(sample_img)
    img_list.append(torch_img)
    i+=1
    
img_list = torch.stack(img_list) # [100, 3, 244, 244]

# Making an actual torch.tensor out of target_list
target_list = torch.stack(target_list)
prefix_size = target_list.size()[:-2]
target_list = target_list.view(*prefix_size,399).squeeze(dim=1) # [100, 399]

###########################################################

# 1. Creating DataLoader

batch_size = 10
dataset = CustomDataset(img_list, target_list)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. Define the model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model_resnet50(pretrained=False).to(device)

if os.path.exists('./net.pth'):
    net.load_state_dict(torch.load('./net.pth', map_location=device))
    print('load pretrained weight')

# 3. Define Loss function 

loss_fn = nn.MSELoss()

# 4. Define Optimizer 

optimizer = optim.Adam(net.parameters(), lr=0.005)

# 5. Training loop

num_epochs = 10  # Adjust as needed
print_interval = 100  # Adjust as needed

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for i, (images, keypoints) in enumerate(dataloader, 1):
        images, keypoints = images.to(device), keypoints.to(device)

        # Forward pass
        outputs = net(images)

        # Compute loss
        loss = loss_fn(outputs, keypoints)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print statistics
        if i % print_interval == 0:
            average_loss = running_loss / print_interval
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i}/{len(dataloader)}], Loss: {average_loss}")
            running_loss = 0.0

    

# 7. Save the model
torch.save(net.state_dict(), "trained_model.pth")



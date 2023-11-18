import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
import torch.optim as optim
from models.Resnet50 import model_resnet50

from utils import json_loader
from PIL import Image
from tqdm.auto import tqdm
import argparse
from utils.dataset import CustomDataset

from utils.device import get_device



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


imgresizepath = "data/h3wb/reimages/"


def main(args):
    set_seed(args.seed)
    
    input_list, target_list, _ = json_loader("data/h3wb/annotations/train.json", 3, "train")
    print(f"json loaded")

    input_list = input_list[: len(input_list) // 8]
    target_list = target_list[: len(target_list) // 8]

    img_list = []

    # Making an actual torch.tensor out of images
    transform = transforms.Compose([transforms.ToTensor()])
    for input in tqdm(input_list):
        sample_img = Image.open(os.path.join(imgresizepath, input))
        torch_img = transform(sample_img)
        img_list.append(torch_img)

    img_list = torch.stack(img_list)  # [100, 3, 244, 244]

    # Making an actual torch.tensor out of target_list
    target_list = torch.stack(target_list)
    prefix_size = target_list.size()[:-2]
    target_list = target_list.view(*prefix_size, 399).squeeze(dim=1)  # [100, 399]

    ###########################################################

    # 1. Creating DataLoader    
    batch_size = args.batch_size
    dataset = CustomDataset(img_list, target_list)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Define the model
    device = get_device()
    weights = ResNet50_Weights.DEFAULT if args.use_pretrained else None
    net = model_resnet50(weights=weights).to(device)

    if os.path.exists("./net.pth"):
        net.load_state_dict(torch.load("./net.pth", map_location=device))
        print("load pretrained weight")

    # 3. Define Loss function and optmizer
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)

    # 4. Training loop
    num_epochs = args.num_epochs
    print_interval = args.print_interval

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for i, (images, keypoints) in enumerate(tqdm(dataloader)):
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
            if (i + 1) % print_interval == 0:
                average_loss = round(running_loss / print_interval, 4)
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {average_loss}"
                )
                running_loss = 0.0

    # 6. Save the model
    torch.save(net.state_dict(), args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--print_interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_pretrained", default=True)
    parser.add_argument("--save_path", type=str, default="./trained_model.pth")
    args = parser.parse_args()

    main(args)

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import ResNet50_Weights
import torch.optim as optim
from models.Resnet50 import model_resnet50

from tqdm.auto import tqdm
import argparse
from utils.dataset import prepare_dataloader, prepare_lazy_dataloader

from utils.device import get_device


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(net, dataloader, device):
    net.eval()
    loss_fn = nn.L1Loss()
    running_loss = 0.0
    for images, keypoints in tqdm(dataloader):
        images, keypoints = images.to(device), keypoints.to(device)
        outputs = net(images)
        loss = loss_fn(outputs, keypoints)
        running_loss += loss.item()
    average_loss = round(running_loss / len(dataloader), 4)
    return average_loss


def main(args):
    set_seed(args.seed)

    if args.lazy:
        train_dataloader = prepare_lazy_dataloader(args, "train")
        eval_dataloader = prepare_lazy_dataloader(args, "dev")
    else:
        train_dataloader = prepare_dataloader(args, "train")
        eval_dataloader = prepare_dataloader(args, "dev")

    # 2. Define the model
    device = get_device()
    weights = ResNet50_Weights.DEFAULT if args.use_pretrained else None
    net = model_resnet50(weights=weights).to(device)

    if os.path.exists("./net.pth"):
        net.load_state_dict(torch.load("./net.pth", map_location=device))
        print("load pretrained weight")

    # 3. Define Loss function and optmizer
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    # 4. Training loop
    num_epochs = args.num_epochs
    print_interval = args.print_interval
    best_eval_loss = float("inf")

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for i, (images, keypoints) in enumerate(tqdm(train_dataloader)):
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
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i+1}/{len(train_dataloader)}], Loss: {average_loss}"
                )
                running_loss = 0.0

        with torch.no_grad():
            eval_loss = validate(net, eval_dataloader, device)
            print(f"Validation Loss at Epoch [{epoch + 1}/{num_epochs}]: {eval_loss}")
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(net.state_dict(), args.save_path)
                print("Saved best model at epoch ", epoch + 1)

    print("Finished Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--print_interval", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_pretrained", default=True)
    parser.add_argument("--save_path", type=str, default="./trained_model.pth")
    parser.add_argument("--image_path", type=str, default="./data/h3wb/reimages/")
    parser.add_argument(
        "--annotation_path", type=str, default="./data/h3wb/annotations"
    )
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    main(args)

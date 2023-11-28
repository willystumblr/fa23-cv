import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
from tqdm.auto import tqdm
import argparse
from models.CombinedModel import  model_resnet18_4_with_sobel, model_resnet50_4_with_sobel, model_resnet50_with_sobel, model_resnet50_with_sift, model_resnet50_5_with_sobel_superpixel, model_resnet18_5_with_sobel_superpixel, model_resnet50_4_with_superpixel, model_resnet18_4_with_superpixel, model_resnet18_with_sift
from models.Resnet50 import model_resnet18, model_resnet50
from utils.dataset import prepare_dataloader, prepare_lazy_dataloader, prepare_sift_dataloader, prepare_superpixel_dataloader

from utils.device import get_device


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(args, net, dataloader, device):
    net.eval()
    loss_fn = nn.L1Loss()
    running_loss = 0.0
    for components in tqdm(dataloader):
        images, keypoints = components[0].float().to(device), components[1].float().to(device)
        if 'sift' in args.model_name:
            descriptors = components[2].float().to(device)
            outputs = net(images, descriptors)
        elif 'superpixel' in args.model_name:
            label = components[2].float().to(device)
            outputs = net(images, label)    
        else:
            outputs = net(images)
        loss = loss_fn(outputs, keypoints)
        running_loss += loss.item()
    average_loss = round(running_loss / len(dataloader), 4)
    return average_loss


def main(args):
    set_seed(args.seed)

    if 'sift' in args.model_name:
        train_dataloader = prepare_sift_dataloader(args, "train")
        eval_dataloader = prepare_sift_dataloader(args, "dev")
    elif 'superpixel' in args.model_name:
        train_dataloader = prepare_superpixel_dataloader(args, "train")
        eval_dataloader = prepare_superpixel_dataloader(args, "dev")   
    elif args.lazy:
        train_dataloader = prepare_lazy_dataloader(args, "train")
        eval_dataloader = prepare_lazy_dataloader(args, "dev")
    else:
        train_dataloader = prepare_dataloader(args, "train")
        eval_dataloader = prepare_dataloader(args, "dev")

    # 2. Define the model
    device = get_device()
    print(f"Using {args.model_name}")
    if args.model_name == "resnet50_with_sobel":
        if args.use_pretrained:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            print("Using pretrained weights")
        else:
            weights = None
            print("Training from scratch")
        net = model_resnet50_with_sobel(weights=weights).to(device)
    
    elif args.model_name == "resnet50_with_sift":
        if args.use_pretrained:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            print("Using pretrained weights")
        else:
            weights = None
            print("Training from scratch")
        net = model_resnet50_with_sift(weights=weights).to(device) 
    
    elif args.model_name == "resnet50_4_with_sobel":
        if args.use_pretrained:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            print("Using pretrained weights")
        else:
            weights = None
            print("Training from scratch")
        net = model_resnet50_4_with_sobel(weights=weights).to(device)
    
    elif args.model_name == "resnet18_4_with_sobel":
        if args.use_pretrained:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            print("Using pretrained weights")
        else:
            weights = None
            print("Training from scratch")
        net = model_resnet18_4_with_sobel(weights=weights).to(device)
    
    elif args.model_name == "resnet50_4_with_superpixel":
        if args.use_pretrained:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            print("Using pretrained weights")
        else:
            weights = None
            print("Training from scratch")
        net = model_resnet50_4_with_superpixel(weights=weights).to(device)
        
    elif args.model_name == "resnet18_4_with_superpixel":
        if args.use_pretrained:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            print("Using pretrained weights")
        else:
            weights = None
            print("Training from scratch")
        net = model_resnet18_4_with_superpixel(weights=weights).to(device) 
    
    elif args.model_name == "resnet50_5_with_sobel_superpixel":
        if args.use_pretrained:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            print("Using pretrained weights")
        else:
            weights = None
            print("Training from scratch")
        net = model_resnet50_5_with_sobel_superpixel(weights=weights).to(device)
        
    elif args.model_name == "resnet18_5_with_sobel_superpixel":
        if args.use_pretrained:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            print("Using pretrained weights")
        else:
            weights = None
            print("Training from scratch")
        net = model_resnet18_5_with_sobel_superpixel(weights=weights).to(device)    
     
    elif args.model_name == "resnet18":
        if args.use_pretrained:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            print("Using pretrained weights")
        else:
            weights = None
            print("Training from scratch")
        net = model_resnet18(weights=weights).to(device)
    
    elif args.model_name == "resnet18_with_sift":
        if args.use_pretrained:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            print("Using pretrained weights")
        else:
            weights = None
            print("Training from scratch")
        net = model_resnet18_with_sift(weights=weights).to(device)
       
    else:
        if args.use_pretrained:
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            print("Using pretrained weights")
        else:
            weights = None
            print("Training from scratch")
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

        for i, components in enumerate(tqdm(train_dataloader)):
            # print(len(components))
            # try:
            images, keypoints = components[0].float().to(device), components[1].float().to(device)
            if 'sift' in args.model_name:
                descriptors = components[2].float().to(device)
                # print(descriptors.shape)
                outputs = net(images, descriptors)
            if 'superpixel' in args.model_name:
                labels = components[2].float().to(device)
                outputs = net(images, labels)
            # Forward pass
            else:
                outputs = net(images)
            # except: ### NOTE: Use the line below (the original) if .float() doesn't work ###
            #     images, keypoints = components[0].to(device), components[1].float().to(device)
            #     if 'sift' in args.model_name:
            #         descriptors = components[2].to(device)
            #         outputs = net(images, descriptors)
            #     # Forward pass
            #     else:
            #         outputs = net(images)

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
            eval_loss = validate(args, net, eval_dataloader, device)
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
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--save_path", type=str, default="./trained_model.pth")
    parser.add_argument("--image_path", type=str, default="./data/h3wb/reimages/")
    parser.add_argument(
        "--annotation_path", type=str, default="./data/h3wb/annotations"
    )
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    main(args)

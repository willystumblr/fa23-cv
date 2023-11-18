import os
import torch
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from models.Resnet50 import model_resnet50

from utils import json_loader
from PIL import Image
from tqdm.auto import tqdm
import argparse
from utils.dataset import CustomEvalDataset

from utils.device import get_device

imgresizepath = "data/h3wb/reimages/"


def test_score(predict_list, target_list):
    predict_list = torch.cat(predict_list, dim=0)

    count = [0, 0, 0, 0, 0, 0]
    diff = predict_list - target_list
    diff = diff - (diff[:, 11:12, :] + diff[:, 12:13, :]) / 2  # pelvis align
    diff1 = (diff - diff[:, 0:1, :])[:, 23:91, :]  # nose align face
    diff21 = (diff - diff[:, 91:92, :])[:, 91:112, :]  # wrist aligned left hand
    diff22 = (diff - diff[:, 112:113, :])[:, 112:, :]  # wrist aligned right hand

    diff = torch.sqrt(torch.sum(torch.square(diff), dim=-1))

    count[0] = torch.mean(diff).item()
    count[1] = torch.mean(diff[:, :23]).item()
    count[2] = torch.mean(diff[:, 23:91]).item()
    count[3] = torch.mean(diff[:, 91:]).item()
    count[4] = torch.mean(torch.sqrt(torch.sum(torch.square(diff1), dim=-1))).item()
    count[5] = (
        torch.mean(torch.sqrt(torch.sum(torch.square(diff21), dim=-1))).item()
        + torch.mean(torch.sqrt(torch.sum(torch.square(diff22), dim=-1))).item()
    )
    
    for i in range(6):
        count[i] = round(count[i], 1)

    print("Pelvis aligned MPJPE is " + str(count[0]) + " mm")
    print("Pelvis aligned MPJPE on body is " + str(count[1]) + " mm")
    print("Pelvis aligned MPJPE on face is " + str(count[2]) + " mm")
    print("Nose aligned MPJPE on face is " + str(count[4]) + " mm")
    print("Pelvis aligned MPJPE on hands is " + str(count[3]) + " mm")
    print("Wrist aligned MPJPE on hands is " + str(count[5] / 2) + " mm")


def main(args):
    print(f"evaluating {args.model_path} on dev set")

    split = "dev" if args.mode == "dev" else "test"
    input_list, target_list, _ = json_loader(f"data/h3wb/annotations/{split}.json", 3, "train")
    print(f"json loaded")

    input_list = input_list[: len(input_list) // 800]
    target_list = target_list[: len(target_list) // 800]

    img_list = []

    # Making an actual torch.tensor out of images
    transform = transforms.Compose([transforms.ToTensor()])
    for input in tqdm(input_list):
        sample_img = Image.open(os.path.join(imgresizepath, input))
        torch_img = transform(sample_img)
        img_list.append(torch_img)

    img_list = torch.stack(img_list)  # [100, 3, 244, 244]

    # Making an actual torch.tensor out of target_list
    target_list = torch.stack(target_list).reshape(-1, 133, 3)

    batch_size = args.batch_size
    dataset = CustomEvalDataset(img_list)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size)

    device = get_device()
    weights = ResNet50_Weights.DEFAULT
    net = model_resnet50(weights=weights).to(device)

    # net.load_state_dict(torch.load(args.model_path, map_location=device))
    # print("load trained weight")

    net.eval()
    predict_list = []
    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)
            outputs = net(images)  # [batch_size, 399]
            outputs = torch.reshape(outputs, (-1, 133, 3))
            predict_list.append(outputs)

    test_score(predict_list, target_list.to(device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mode", type=str, default="dev")
    args = parser.parse_args()
    
    assert args.mode in ["dev", "test"], "mode should be either dev or test"

    main(args)

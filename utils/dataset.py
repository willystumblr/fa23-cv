import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from utils import json_loader
from PIL import Image
from tqdm.auto import tqdm
from utils.dataset import CustomDataset


class CustomDataset(Dataset):
    def __init__(self, images, keypoints):
        self.images = images
        self.keypoints = keypoints

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.keypoints[index]


class CustomEvalDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


imgresizepath = "data/h3wb/reimages/"


def prepare_dataloader(batch_size, set_type):
    assert set_type in ["train", "dev"], "set_type must be either train or dev"
    input_list, target_list, _ = json_loader(
        f"data/h3wb/annotations/{set_type}.json", 3, "train"
    )
    print(f"json loaded")

    input_list = input_list[: len(input_list)]
    target_list = target_list[: len(target_list)]

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

    batch_size = batch_size
    dataset = CustomDataset(img_list, target_list)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True if set_type == "train" else False
    )
    return dataloader

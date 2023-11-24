import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from utils import json_loader
from PIL import Image
from tqdm.auto import tqdm
import cv2
import numpy as np

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


def prepare_dataloader(
    args,
    set_type,
):
    assert set_type in ["train", "dev"], "set_type must be either train or dev"
    input_list, target_list, _ = json_loader(
        f"{args.annotation_path}/{set_type}.json", 3, "train"
    )
    print(f"json loaded")

    img_list = []
    # Making an actual torch.tensor out of images
    transform = transforms.Compose([transforms.ToTensor()])
    for input in tqdm(input_list):
        sample_img = Image.open(os.path.join(args.image_path, input))
        torch_img = transform(sample_img)
        img_list.append(torch_img)

    img_list = torch.stack(img_list)  # [100, 3, 244, 244]

    # Making an actual torch.tensor out of target_list
    target_list = torch.stack(target_list)
    prefix_size = target_list.size()[:-2]
    target_list = target_list.view(*prefix_size, 399).squeeze(dim=1)  # [100, 399]

    dataset = CustomDataset(img_list, target_list)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if set_type == "train" else False,
        num_workers=args.num_workers,
    )
    return dataloader


class CustomLazyDataset(Dataset):
    def __init__(self, image_paths, keypoints):
        self.image_paths = image_paths
        self.keypoints = keypoints
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = self.transforms(image)
        return image, self.keypoints[index]


def prepare_lazy_dataloader(
    args,
    set_type,
):
    assert set_type in ["train", "dev"], "set_type must be either train or dev"
    input_list, target_list, _ = json_loader(
        f"{args.annotation_path}/{set_type}.json", 3, "train"
    )
    print(f"json loaded")

    img_paths = [os.path.join(args.image_path, input) for input in input_list]
    target_list = torch.stack(target_list)
    prefix_size = target_list.size()[:-2]
    target_list = target_list.view(*prefix_size, 399).squeeze(dim=1)  # [100, 399]

    lazy_dataset = CustomLazyDataset(img_paths, target_list)
    print(f"Dataset size: {len(lazy_dataset)}")
    lazy_dataloader = DataLoader(
        lazy_dataset,
        batch_size=args.batch_size,
        shuffle=True if set_type == "train" else False,
        num_workers=args.num_workers,
    )
    return lazy_dataloader

class CustomSIFTDataset(Dataset):
    def __init__(self, images, keypoints, sift_descriptors):
        self.images = images
        self.sift_descriptors = sift_descriptors
        self.keypoints = keypoints

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        sift_descriptor = self.sift_descriptors[index]
        keypoint = self.keypoints[index]
        return image, keypoint, sift_descriptor

class CustomSIFTEvalDataset(Dataset):
    def __init__(self, images, sift_descriptors):
        self.images = images
        self.sift_descriptors = sift_descriptors
        #self.keypoints = keypoints

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        sift_descriptor = self.sift_descriptors[index]
        #keypoint = self.keypoints[index]
        return image, sift_descriptor


def compute_sift_descriptors(images, num_keypoints=133):
    print("Extracting SIFT descriptors...")
    sift = cv2.SIFT_create(num_keypoints)
    sift_descriptors = []
    for img in tqdm(images):
        gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        _, descriptor = sift.detectAndCompute(gray_img, None)
        # If no descriptors were found or there are fewer than num_keypoints, pad with zeros
        if descriptor is None or descriptor.shape[0] < num_keypoints:
            if descriptor is None:
                descriptor = np.zeros((num_keypoints, 128))
            else:
                descriptor = np.vstack([descriptor, np.zeros((num_keypoints - descriptor.shape[0], 128))])
        # If more descriptors are found, select the first num_keypoints ones
        elif descriptor.shape[0] > num_keypoints:
            descriptor = descriptor[:num_keypoints, :]
        sift_descriptors.append(descriptor.flatten())
    descriptors = torch.stack([torch.from_numpy(d) for d in sift_descriptors])
    # print(descriptors.shape)
    return descriptors

def prepare_sift_dataloader(
    args,
    set_type,
):
    assert set_type in ["train", "dev"], "set_type must be either train or dev"
    input_list, target_list, _ = json_loader(
        f"{args.annotation_path}/{set_type}.json", 3, "train"
    )
    print(f"json loaded")

    img_list = []
    # Making an actual torch.tensor out of images
    transform = transforms.Compose([transforms.ToTensor()])
    imgs = []
    for input in tqdm(input_list):
        sample_img = Image.open(os.path.join(args.image_path, input))
        torch_img = transform(sample_img)
        imgs.append(sample_img)
        img_list.append(torch_img)

    img_list = torch.stack(img_list)  # [100, 3, 244, 244]

    # Making an actual torch.tensor out of target_list
    target_list = torch.stack(target_list)
    prefix_size = target_list.size()[:-2]
    target_list = target_list.view(*prefix_size, 399).squeeze(dim=1)  # [100, 399]

    sift_descriptors = compute_sift_descriptors(imgs)
    
    dataset = CustomSIFTDataset(img_list, target_list, sift_descriptors)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if set_type == "train" else False,
        num_workers=args.num_workers,
    )
    return dataloader
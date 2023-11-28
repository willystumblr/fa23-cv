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
    def __init__(self, image_paths, keypoints):
        self.image_paths = image_paths
        self.keypoints = keypoints
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        sift_descriptor = compute_sift_descriptors([image])[0]
        image = self.transforms(image)
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
    # print("Extracting SIFT descriptors...")
    sift = cv2.SIFT_create(num_keypoints)
    sift_descriptors = []
    for img in images:
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

    img_paths = [os.path.join(args.image_path, input) for input in input_list]
    target_list = torch.stack(target_list)
    prefix_size = target_list.size()[:-2]
    target_list = target_list.view(*prefix_size, 399).squeeze(dim=1)  # [100, 399]
    
    dataset = CustomSIFTDataset(img_paths, target_list)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if set_type == "train" else False,
        num_workers=args.num_workers,
    )
    return dataloader


class CustomSuperpixelDataset(Dataset):
    def __init__(self, images, keypoints, superpixel_labels):
        self.images = images
        self.superpixel_labels = superpixel_labels
        self.keypoints = keypoints

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        superpixel_labels = self.superpixel_labels[index]
        keypoint = self.keypoints[index]
        return image, keypoint, superpixel_labels

class CustomSuperpixelEvalDataset(Dataset):
    def __init__(self, images, superpixel_labels):
        self.images = images
        self.superpixel_labelss = superpixel_labels
        #self.keypoints = keypoints

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        superpixel_labels = self.superpixel_labels[index]
        #keypoint = self.keypoints[index]
        return image, superpixel_labels


def compute_superpixel_labels(images, num_keypoints=133):
    print("Extracting Superpixel labels...")
    superpixel_labels = []
    for img in tqdm(images):
        gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        num_superpixels = 400  # desired number of superpixels
        num_iterations = 4     # number of pixel level iterations. The higher, the better quality
        prior = 2              # for shape smoothing term. must be [0, 5]
        num_levels = 4
        num_histogram_bins = 5 # number of histogram bins
        height, width, channels = gray_img.shape

        # initialize SEEDS algorithm
        seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)
       
        # run SEEDS
        seeds.iterate(gray_img, num_iterations)
       
        # get number of superpixel
        num_of_superpixels_result = seeds.getNumberOfSuperpixels()
       
        # retrieve the segmentation result
        labels = seeds.getLabels() # height x width matrix. Each component indicates the superpixel index of the corresponding pixel position
        superpixel_labels.append(labels)
    
    descriptors = torch.stack([torch.from_numpy(d) for d in superpixel_labels])
    print(descriptors.shape)
    
    return descriptors

def prepare_superpixel_dataloader(
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

    superpixel_labels = compute_superpixel_labels(imgs)
    
    dataset = CustomSuperpixelDataset(img_list, target_list, superpixel_labels)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if set_type == "train" else False,
        num_workers=args.num_workers,
    )
    return dataloader
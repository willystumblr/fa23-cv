import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models
from utils import json_loader
import torchvision.transforms as transforms
from PIL import Image
from tqdm.auto import tqdm

if __name__ == "__main__":
    imgpath = "data/h3wb/images/"
    imgresizepath = "data/h3wb/reimages/"
    input_list, target_list, bbox_list = json_loader(
        "data/h3wb/annotations", 3, "train"
    )
    num_data = len(input_list)

    i = 0
    u = 0

    notfoundlist = []
    for input in tqdm(input_list):
        try:
            original_image = Image.open(imgpath + input)
        except:
            notfoundlist.append(input)
            u += 1

        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        resized_image = transform(original_image)
        resized_image_pil = transforms.ToPILImage()(resized_image)
        resized_image_pil.save(imgresizepath + input)
        i += 1

    with open("file_not_found_list.txt", "w") as output_file:
        for filename in tqdm(notfoundlist):
            output_file.write(filename + "\n")

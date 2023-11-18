import torchvision.models as models
from utils import json_loader
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
from PIL import Image
from tqdm.auto import tqdm
import os

if __name__ == "__main__":
    imgpath = "data/h3wb/images/"
    imgresizepath = "data/h3wb/reimages/"
    input_list, target_list, bbox_list = json_loader(
        "data/h3wb/annotations", 3, "train"
    )
    os.makedirs(imgresizepath, exist_ok=True)

    i = 0
    u = 0

    notfoundlist = []
    mean = (0.485, 0.456, 0.456)
    std = (0.229, 0.224, 0.225)
    for input in tqdm(input_list):
        try:
            original_image = Image.open(imgpath + input)
            bbox = bbox_list[i].tolist()[0]
            [left, top, right, bottom] = bbox
            cropped_image = crop(original_image, top, left, bottom - top, right - left)
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            resized_image = transform(cropped_image)
            resized_image_pil = transforms.ToPILImage()(resized_image)
            resized_image_pil.save(imgresizepath + input)
        except:
            notfoundlist.append(input)
            u += 1
        finally:
            i += 1

    with open("file_not_found_list.txt", "w") as output_file:
        print("not found: " + str(u))
        for filename in tqdm(notfoundlist):
            output_file.write(filename + "\n")

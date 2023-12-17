# Efficient 3-D Human Pose Estimation: A Synergy of Classical Computer Vision and Deep Learning

This repository is the implementation of a research project for the 2023 Fall Semester Computer Vision class by Team 16, based on the [H3WB repository](https://github.com/wholebody3d/wholebody3d).

## Install Dependencies

- We conducted all experiments with Python 3.9 with dependencies listed in `requirements.txt`.

```
conda create -n [env name] python=3.9
conda activate [env name]
pip install -r requirements.txt
```

## Data Preparation and Preprocessing

1. Run `data.sh` in a preferred directory (takes several GB and takes 30~45 minutes.)
2. Put `RGBto3D_train.json` and  `RGBto3D_test_img.json` to `./data/h3wb/annotations `
3. Run `python resize.py` to resize images to 224x224.
4. Run `python split_dataset.py` to split the data into pre-defined train, dev, and test sets.

For further details, refer to `./Readme.txt`.

## Training

We implemented our models in `models/ClassicalModel.py` and `models/CombinedModel.py`

```
output_path=/path/to/model/checkpoint
model_name=resnet50 
# one of {"resnet50", "resnet18"} for baseline models
# one of {"resnet50_4_with_sobel", "resnet18_4_with_sobel"} for sobel operator models

python train.py \
    --learning_rate 1e-5 --batch_size 16 --num_epochs 20 \
    --model_name ${model_name} --use_pretrained \
    --save_path ${output_dir}
```

## Evaluation

```
checkpoint_path=/path/to/model/checkpoint
python evaluate.py \
    --model_path ${checkpoint_path} --model_name ${model_name}
```

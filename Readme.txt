Resnet 3d wholebody, with classical cv 
---------------------
0. Set-up:
(1) Dataset
- Run data.sh in a preferred directory (takes several GB and takes 30~45 min. Beware!!!)
- Put RGBto3D_train.json and  RGBto3D_test_img.json to /data/h3wb/annotations 
(2) Environment
```
conda create -n [env name] python=3.9

pip install -r requirements.txt
```

1. About Dataset:
- It follows the COCO dataset format: https://github.com/jin-s13/COCO-WholeBody

- See h3wb.py file, (which is mmpose config file for h3wb) to see which number corresponds to which keypoint.


<json layout>

XXX.json --- sample id --- 'image_path'
                        |
                        -- 'bbox' --- 'x_min'
                        |          |- 'y_min'
                        |          |- 'x_max'
                        |          |- 'y_max'
                        |
                        |- 'keypont_2d' --- joint id --- 'x'
                        |                             |- 'y'
                        |
                        |- 'keypont_3d' --- joint id --- 'x'
                                                      |- 'y'
                                                      |- 'z'

Example:

{
    '2000':{
        'keypoints_3d': 
            {'0': 
                {'x': -41.694671630859375, 
                'y': -1502.7777099609375, 
                'z': -198.89476013183594}, 
            '1': {'x': -61.85731887817383, 
                'y': -1535.35107421875, 
                'z': -174.36801147460938}, 
            '2': {'x': -76.65953063964844, 
                'y': -1516.6678466796875, 
                'z': -227.12913513183594}, 
            '3': {'x': -138.2924346923828, 
                'y': -1512.405029296875, 
                'z': -89.14544677734375}, 
            .
            .
            .
            '131': {'x': -380.0482177734375, 
                'y': -696.9531860351562, 
                'z': -67.98588562011719}, 
            '132': {'x': -373.5930480957031, 
                'y': -692.741943359375, 
                'z': -56.026309967041016}
            }, 
        'image_path': 'S5_Waiting_1.60457274_003051.jpg', 
        'bbox': {'x_min': 371, 'y_min': 180, 'x_max': 523, 'y_max': 609}
        }
}

------------------------------------------
2. Train your model 

+ TODO: Add Training code that loads .json file and images 
+ TODO: Since we don't have valid test dataset, We have to split training dataset. 

------------------------------------------
3. Predict from test dataset

+ TODO: Add Predicting code that writes .json file of the result.

------------------------------------------
4. Get scores 

- Finally, Put your predicted data in .json file, with the same layout above, 
so that we can run it on test_leaderboard

Use test_leaderboard.py to get the score of your data. 

$python test_leaderboard.py (predected.json)

+ TODO: Add Visualization for some predicted result 
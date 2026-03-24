## 1. Download the images

Resource: https://esa-worldcover.org/en

1. Run the ```sentinel_raw_download.py``` and  the ```esa_labels_download.py``` scripts. ***Specify the parameters in download_config.py***

I would recommend to download date-by-date if you know the specific dates you need. Or you may check them in ```config.py```.

It is recommended to download the data to the root directory of the project.

2. Run the alignment.py script to align the labels from 2021 with the labels from 2020 and raw images with the labels from 2021. Specify parameters in ```alignment_config.py```.

## 2. Feature extraction

Images are being downloaded in .tif files. We do not convert them to RGB format, because we need more bands for creating tabular data. 

1. Baseline feature extraction pipeline. ```feature_engineering/build_tables.py```

Basically extracts features from the first 3 bands R (B04), G (B03), B (B02).

2. Extended feature extraction pipeline. ```feature_engineering/build_tables_extended.py```

Extracts all the features from the baseline approach + extra features employing B08 and B011 bands of satellite imagery.


As the result, for each spatial unit of the image (basically a grid cell) we get features (described in ```config.py```) and the labels distributions for 2020 and 2021 (so we can calculate the diff).

## 3. How did we choose dates?

For the first task we need to predict t+1 timestamp given data at t. So we decided to use raw data from 2020 as the input to train and labels from 2021 as the targets to train.

Since one raw in the tables in a spatial unit, not a picture itself, we're getting a lot of rows from each picture. So we limited the number of pictures. We use 3 pictures for 3 seasons (Spring, Summer, Autumn) from 2020 to train. We filter cloudness by 5 % threshold.

Due to the time bounds of the data we don't work with time-wise test splits, focusing on spatial-wise splits and testing 

```test_spatial```. From the same dates in 2020 we randomly pick a part for the test. So model sees only specific regions of train images and never sees the regions which we put in train. It indicates the robustness of models to spatial changes.

This logic is implemented in ```evaluation/generate_test.py``` and in the ```notebooks/baseline.ipynb```. Emir is working on moving the logic from .ipynb to more handy scripts.

## 4. Why 3 labels and why distributions?

We decided tp simplify the labels structure and focus on the questions of the urban/nature proportion within the city and the dynamics of this rivarly. So we only use "urban" label, which is "built_up", "water" (all water bodies) and "vegetation" (all greenlands). 

We also use the distribution to generate labels for whole spatial units. So we do not make predictions pixel-by-pixel. But take a whole spatial unit (NxN grid cell) as an input and predict the distribution of these 3 classes within the unit.

Same logic for the task2, where we predict the change in these distributions' shifts.

## 5. Modeling 

The ```models``` directory contains all the utils for reproducible and reliable experiments running. We split train dataframe to train and val parts with spatial-split approach. Then we search for the optimal HP with Optuna framework.

We experiment with a wide range of 8 different classical ML algorithms in order to define the most suitable sor this specific task and data.


## 6. Evaluation

The ```evaluation``` directory contains the function to generate spatial-wise train/val split (the same is used to cut the test part off). And script ```metrics.py``` which generates comprehensive reports from the training results.

We also simulate stress testing of the trained models putting light (0.01 from features std) and strong (0.1 from features std) gaussian noise to the features of X_test.

## 7. Conclusion

This was our first approach. Eventually we collectively agreed on switching to the different one. Nevertheless we value the work done in the this direction as well, and it motivated us to publish this code alongside the final one.
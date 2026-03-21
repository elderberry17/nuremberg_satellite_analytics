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

Extracts all the features from the baseline approach + extra features that Aryan described in Notion.

The example of usage could be seen in ```notebooks/baseline.ipynb```

As the result, for each spatial unit of the image (basically a grid cell) we get features (described in ```config.py```) and the labels distributions for 2020 and 2021 (so we can calculate the diff).

## 3. How did we choose dates?

For the first task we need to predict t+1 timestamp given data at t. So we decided to use raw data from 2020 as the input to train and labels from 2021 as the targets to train.

Since one raw in the tables in a spatial unit, not a picture itself, we're getting a lot of rows from each picture. So we limited the number of pictures. We use 3 pictures for 3 seasons (Spring, Summer, Autumn) from 2020 to train. We filter cloudness by 5 % threshold.

Then we generate 3 different test files:

1. ```test_spatial```. From the same dates in 2020 we randomly pick a part for the test. So model sees only specific regions of train images and never sees the regions which we put in train. It indicates the robustness of models to spatial changes.

2. ```test_temp```. These are the spatial units from the same geometrical regions that models see during training, but from winter 2021. Since model never see this season, it shows the robustness to the seasonal shifts.

3. ```test_spatial_temp```. 2 ideas at the same time. It must be the stress test for models.

This logic is implemented in ```evaluation/generate_test.py``` and in the ```notebooks/baseline.ipynb```. Emir is working on moving the logic from .ipynb to more handy scripts.

## 4. Why 3 labels and why distributions?

We decided tp simplify the labels structure and focus on the questions of the urban/nature proportion within the city and the dynamics of this rivarly. So we only use "urban" label, which is "built_up", "water" (all water bodies) and "vegetation" (all greenlands). 

We also use the distribution to generate labels for whole spatial units. So we do not make predictions pixel-by-pixel. But take a whole spatial unit (NxN grid cell) as an input and predict the distribution of these 3 classes within the unit.

Same logic for the task2, where we predict the change in these distributions' shifts.

## 5. Precalculated datasets

Right now we have these datasets: https://drive.google.com/drive/folders/1aRejdRZycWDcgZgjja7pNswBmq2Bma9z?usp=sharing

We also plan to split ```train_df``` into train/val parts during HPO.

We do not insist on using them as the final version (would be nice though), but you can already experiment with them.

Features of the units near boarders were calculated with reflextive padding approach. The spatial units are sliced in the way that we do not have nor any overlaps between them neither any blind spots in-between.

```kernel_size``` and ```stride``` are chosen to make spatial units visible enough and easier to percieve.

## 6. How do we see the project?


No use input. Just geo-dashboard: maps for a list of preset dates. User chooses a date, clicks on the map -> the spatial unit is identified (it gets highlighted), then the statistics/predictions about the unit are illustrated. 

Models' predictions must be also explained within the interface.

Now we are waiting for the response of the course's team if we have to serve our models for real-time inference or we can precalcuted all the predictions offline (would be good).

## 7. What else?

1. More proficient feature engineering: mix extracted features with each other

2. Merics - in progress (do not forget about stress and other types of metrics)

3. HPO and train pipeline - in progress

4. Play with the models, parameters, datasets' compositions and features - get the technical report and the dominating models.

5. Build the UI/UX part - Soban

6. Build models API's for serving and integrate in API - J/Aryan (perphaps extra)

We will probably have 4 models: 2 per each of the tasks

7. Build SHAP for tree-based models explanation - Aryan (perphaps extra)

8. Deploy the application into Docker-compose - J

9. Test it

10. ChatGPT logs, errors analysis and so forth

11. Make the final training/inference pipeline reproducible, test it. Polish other important scripts

12. Clean code (?)

13. Make the technical report (justify every decision)

I could forget smth
## 1. How to download the data?

Resource: https://esa-worldcover.org/en

1. Run the '''sentinel_raw_download.py''' and  the '''esa_labels_download.py''' scripts. ***Specify the parameters to choose file names / dirs to save the datum.*** Also adjust the downloading parameters in the sentinel_raw_download.py! 

2. Run the *_process.py scipts to compose .rgb from .tif decomposed channels.

3. Run the alignment.py script to align the raw images with the labels.

## 2. How to run the baseline?

1. There is a simple data preprocessing + feature engineering baseline in the ```feature_engineering``` directory. These functions are used in the ```notebooks/baseline.ipynb```

2. Got to ```notebooks/baseline.ipynb```. Construct your dataset with the ```kernel_size``` and ```stride``` parameters. DON'T CHANGE RANDOM_SEED. After building the train and test datasets play with some models. 

Pros:
1. It works. We know how to collect, process data, train and evaluate models in the basic set-up.

2. ```kernel_size``` allows us extracting features for the whole spatial unit. ```stride``` helps adjusting the number of rows to create (resources limits).

3. Based on the RFC classifier it's seen that a model CAN train with 2020 data and predict something adequate for 2021. The concept for the first task is proved. The second task is not started yet.

4. We are using a split in time to evaluate models. It is quite fair. Our models never see 2021, but are being evaluated on this data (real-life scenario).


Cons:
1. Weak feature preprocessing

2. Only one .ipynb now

3. Weak evaluation. We need to evaluate in terms of space as well (try to predict the parts of the city never seen during training). No CV, no confidence intervals, weak metrics. No class disbalance handling.

4. Clouds?

5. The second task is not started yet

6. HPO is missed

7. Some more, but we will handle it
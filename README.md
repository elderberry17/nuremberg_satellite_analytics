## How to download the data?

Resource: https://esa-worldcover.org/en

1. Run the '''sentinel_raw_download.py''' and  the '''esa_labels_download.py''' scripts. ***Specify the parameters to choose file names / dirs to save the datum.*** Also adjust the downloading parameters in the sentinel_raw_download.py! 

2. Run the *_process.py scipts to compose .rgb from .tif decomposed channels.

3. Run the alignment.py script to align the raw images with the labels.
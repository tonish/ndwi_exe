This repo corrsponds to home assignment doc.

0. cd into the repo folder and run:
gsutil -m rsync -r gs://sen1floods11 to download the dataset.<br />
The dataset folder (v1.1) should be inside the repo folder for paths compatability
   
1. To answer step 1: run q1.py -q <1:3> 
    you can select 1,2,3 to answer question 1,2,3 of step 1
2. To answer step 2: run q2.py for the first part of the question and run q2.ipynb for the second part of the question
    you can also run it in colab [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tonish/ndwi_exe/blob/main/q2_ploting.ipynb)
3. To answer step 3: run q3.py - a new file called images.tfrecords should appear in the main dir.
4. In step4 I compare ndwi baseline, xgboost and UNET for optimal water segmention map with q4.ipynb .<br />
   you can also run it in colab [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tonish/ndwi_exe/blob/main/q4.ipynb)

| model | IOU | 
| ------------- | ------------- |
| xgboost | 0.51 |
| UNET -13 bands | 0.47 |
| baseline_ndwi| 0.42 |

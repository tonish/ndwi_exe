import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import skimage
import os


#load images and labels
def load_imgs(path):
    """load images and labels paths from the split csv"""
    with open(path, 'r') as file:
        content = file.readlines()
    
    data_prefix = 'v1.1/data/flood_events/HandLabeled/S2Hand'
    label_prefix = 'v1.1/data/flood_events/HandLabeled/LabelHand'

    imgs = [os.path.join(data_prefix,i.split(',')[0].replace('S1Hand','S2Hand')) for i in content]
    labels = [os.path.join(label_prefix,i.split(',')[1].strip('\n')) for i in content]
    return imgs,labels

def q4_xgb(n_estimators=10, n_jobs=-1):
    #load training images and labels
    train_path = 'v1.1/splits/flood_handlabeled/flood_train_data.csv'
    train_imgs, train_labels = load_imgs(train_path)
    
    #read the images and labels - mask out no data pixels as 0
    train_imgs = np.array([np.moveaxis(skimage.io.imread(img),0,-1) for img in train_imgs])
    train_labels = np.array([(skimage.io.imread(lbl)>0).astype(int) for lbl in train_labels])

    #reshape to n*w*h,b
    train_imgs = train_imgs.reshape(-1,train_imgs.shape[3])
    train_labels = train_labels.reshape(-1)
    
    clf = XGBClassifier(booster='gbtree', gpu_id=-1, n_estimators=10, n_jobs=-1,
              num_parallel_tree=1, random_state=42 )
    
    clf.fit(train_imgs, train_labels)

    #load test split
    test_path = 'v1.1/splits/flood_handlabeled/flood_test_data.csv'
    test_imgs, test_labels = load_imgs(test_path)

    #read the images and labels - mask out no data pixels as 0
    test_imgs = np.array([np.moveaxis(skimage.io.imread(img),0,-1) for img in test_imgs])
    test_labels = np.array([(skimage.io.imread(lbl)>0).astype(int) for lbl in test_labels])
    
    #reshape to n*w*h,b
    test_imgs = test_imgs.reshape(-1,test_imgs.shape[3])

    predictions = clf.predict(test_imgs)
    predictions_reshape = predictions.reshape(len(test_labels),512,512)
    test_labels = test_labels.reshape(len(test_labels),512,512)

    IOU_list = []
    for pred_img, label in zip(predictions_reshape, test_labels):

        # calculate IOU
        intersection = np.sum(pred_img * label)
        union = np.sum(pred_img)+np.sum(label) - intersection + np.e**-10

        IOU = intersection / union
        IOU_list.append(IOU)
    return np.mean(IOU_list), predictions_reshape
import skimage
import os
import numpy as np
from UNET import config

#Thresholding baseline
def load_imgs(path):
    """load images and labels paths from the split csv"""
    with open(path, 'r') as file:
        content = file.readlines()

    data_prefix = 'v1.1/data/flood_events/HandLabeled/S2Hand'
    label_prefix = 'v1.1/data/flood_events/HandLabeled/LabelHand'

    imgs = [os.path.join(data_prefix,i.split(',')[0].replace('S1Hand','S2Hand')) for i in content]
    labels = [os.path.join(label_prefix,i.split(',')[1].strip('\n')) for i in content]
    return imgs,labels


def calc_ndwi(img):
    """function to calcutlate ndwi with bands B03 and B08"""
    return ((img[2,:,:]-img[7,:,:]) / (img[2,:,:]+img[7,:,:]+np.e**-100))


def test_thresh(thresh):
    #load test images
    test_path = config.TEST_PATH
    test_imgs, test_labels = load_imgs(test_path)

    IOU_list = []
    masked_imgs = []
    for item_img, item_label in zip(test_imgs, test_labels):

        img = calc_ndwi(skimage.io.imread(item_img))
        label = skimage.io.imread(item_label)
        
        #apply thresh
        masked_img = img>thresh
        masked_label = label>0 #because the label has -1=no_data,0=not_water,1=water

        # calculate IOU
        intersection = np.sum(masked_img * masked_label)
        union = np.sum(masked_img)+np.sum(masked_label) - intersection + np.e**-10

        IOU = intersection / union
        if IOU <0:
            IOU=0
        IOU_list.append(IOU)
        masked_imgs.append(masked_img)
    return np.mean(IOU_list), masked_imgs


import skimage
import os
import numpy as np

def calc_ndwi(img):
    """function to calcutlate ndwi with bands B03 and B08"""
    return ((img[2,:,:]-img[7,:,:]) / (img[2,:,:]+img[7,:,:]+np.e**-100))

def load_imgs(path):
    """load images and labels paths from the split csv"""
    with open(path, 'r') as file:
        content = file.readlines()
    
    data_prefix = 'v1.1/data/flood_events/HandLabeled/S2Hand'
    label_prefix = 'v1.1/data/flood_events/HandLabeled/LabelHand'

    imgs = [os.path.join(data_prefix,i.split(',')[0].replace('S1Hand','S2Hand')) for i in content]
    labels = [os.path.join(label_prefix,i.split(',')[1].strip('\n')) for i in content]
    return imgs,labels


#train split loader
train_path = 'v1.1/splits/flood_handlabeled/flood_train_data.csv'
train_imgs, train_labels = load_imgs(train_path)

ndwi_values = []
        
#load pairs of images , calculate ndwi and collect the ndwi values
for item_img, item_label in zip(train_imgs,train_labels):
    
    #read image and label
    train_img = calc_ndwi(skimage.io.imread(item_img))  
    train_label = skimage.io.imread(item_label)

    # extract values from img based on mask
    masked_img = train_img*((train_label>0).astype(np.int16))
    masked_img = masked_img[masked_img!=0]

    ndwi_values.append(masked_img)

ndwi_values = np.concatenate(ndwi_values, axis=0)

# I want to try three thresholds : 
# 5 percentile that will capture more pixels
# and the mean that is more restricting but will generalize better outside the dataset
# and 0 becuase I ploted some histograms
thresh1 = np.round(np.percentile(ndwi_values,5),3)
thresh2 = np.round(np.mean(ndwi_values),3)
thresh3 = 0

#load images based on split - test/bolivia
test_path = 'v1.1/splits/flood_handlabeled/flood_test_data.csv'
test_imgs, test_labels = load_imgs(test_path)

bolivia_path = 'v1.1/splits/flood_handlabeled/flood_bolivia_data.csv'
bolivia_imgs, bolivia_labels = load_imgs(bolivia_path)


def test_thresh(imgs_list, labels_list, thresh):
    IOU_list = []
    for item_img, item_label in zip(imgs_list,labels_list):

        img = calc_ndwi(skimage.io.imread(item_img))  
        label = skimage.io.imread(item_label)
        #apply thresh
        masked_img = img>thresh
        masked_label = label>0 #because the label has -1=no_data,0=not_water,1=water

        # calculate IOU
        intersection = np.sum(masked_img * masked_label) 
        union = np.sum(masked_img)+np.sum(masked_label) - intersection + np.e**-11

        IOU = intersection / union
        IOU_list.append(IOU)
    return np.mean(IOU_list)

test_IOU_thresh1 = test_thresh(test_imgs, test_labels,thresh1)
print (f'When thershold = {thresh1}, the mean IOU in all the test images is:',np.round(test_IOU_thresh1,3))

test_IOU_thresh2 = test_thresh(test_imgs, test_labels,thresh2)
print (f'When thershold = {thresh2}, the mean IOU in all the test images is:',np.round(test_IOU_thresh2,3))

test_IOU_thresh0 = test_thresh(test_imgs, test_labels,thresh3)
print (f'When thershold = {thresh3}, the mean IOU in all the test images is:',np.round(test_IOU_thresh0,3))

bolivia_IOU_thresh1 = test_thresh(bolivia_imgs, bolivia_labels,thresh1)
print (f'When thershold = {thresh1}, the mean IOU in all the bolivia images is:',np.round(bolivia_IOU_thresh1,3))

bolivia_IOU_thresh2 = test_thresh(bolivia_imgs, bolivia_labels,thresh2)
print (f'When thershold = {thresh2}, the mean IOU in all the bolivia images is:',np.round(bolivia_IOU_thresh2,3))

bolivia_IOU_thresh0 = test_thresh(bolivia_imgs, bolivia_labels,thresh3)
print (f'When thershold = {thresh3}, the mean IOU in all the bolivia images is:',np.round(bolivia_IOU_thresh0,3))
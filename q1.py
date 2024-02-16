from collections import Counter
import glob
import os
import argparse
import skimage
import numpy as np
#This file corresponds to step 1 - basic statistics

argParser = argparse.ArgumentParser()

argParser.add_argument("-q", "--question", type=int, default=1,
                       help="what qeustion in step 1 to anwer (default=1)")


def count_images(csv_p):
    """
    function gets file path (csv) and 
    returns number of images and number of images per region 
    """
    with open(csv_p, 'r') as file:
        content = file.readlines()
        #split each line of csv by "," delimeter and split eash string by "_"
        content = [j.split('_')[0] for i in content for j in i.split(',')]  
    return Counter(content)

def count_all(f_path):
    "count number of lines in csv"
    with open(f_path, 'r') as file:
        content = file.readlines()
    return (len(content))
            
def calculate_proba(paths):
    "calculate water proba per label-img in the directory, and return list of probability"
    probs = []
    for img_path in paths:
        img = skimage.io.imread(img_path)
        probs.append(np.sum(img==1) / (img.shape[0]*img.shape[1]))
    return probs

def load_imgs(path):
    """load images and labels paths from the split csv"""
    with open(path, 'r') as file:
        content = file.readlines()
    
    data_prefix = 'v1.1/data/flood_events/HandLabeled/S2Hand'
    label_prefix = 'v1.1/data/flood_events/HandLabeled/LabelHand'

    imgs = [os.path.join(data_prefix,i.split(',')[0].replace('S1Hand','S2Hand')) for i in content]
    labels = [os.path.join(label_prefix,i.split(',')[1].strip('\n')) for i in content]
    return imgs,labels


def main(args):
    hand_dir = 'v1.1/splits/flood_handlabeled'
    hand_dir_files = glob.glob(hand_dir+'/*.csv')
        
    if args.question == 1:
        # Number of images in each split and each region
        for file in hand_dir_files:
            print (os.path.basename(file))
            print (count_images(file))

    elif args.question == 2:
        #Per-channel mean and standard deviation
        #sanity check- the number of images in the folder is the same as csv
        counter =0
        for file in hand_dir_files:
            counter+=count_all(file)

        s2_path = 'v1.1/data/flood_events/HandLabeled/S2Hand'
        imgs = glob.glob(s2_path+'/*.tif')
        print (f'{len(imgs)} = {counter}-',len(imgs) == counter, '- Sanity check passed')

        #load all images to a stack with n*W*H*B
        imgs_stack = []
        for img in imgs:
            imgs_stack.append(skimage.io.imread(img))
        imgs_stack = np.stack(imgs_stack)
        imgs_stack = np.moveaxis(imgs_stack,1,-1) #move bands to last axis

        #calculate mean and std
        for band in range(imgs_stack.shape[-1]):
            print(f'For band number {band+1}')
            print ('The images mean is :', imgs_stack[:,:,:,band].mean(), "The images std is:",imgs_stack[:,:,:,band].std())
    
    elif args.question == 3:
        # Probability of water Per image and Per train/dev/test sets and for the held-out region (Bolivia)
        dir_labeled = 'v1.1/data/flood_events/HandLabeled/LabelHand'
        imgs_labeled = glob.glob(dir_labeled+'/*.tif')

        #sanity check - number of labeled images eq info in CSVs       
        counter =0
        for file in hand_dir_files:
            counter+=count_all(file)
        print (f'{len(imgs_labeled)} = {counter}-',len(imgs_labeled) == counter, '- Sanity check passed')

        # probability of water per image in all images
        prboa_per_img = calculate_proba(imgs_labeled)
        print('probabilty of water per image:', prboa_per_img)
        print('\n')
        
        # probabiliy per split
        split_proba = {}

        for csv_file in hand_dir_files:
            imgs,labels = load_imgs(csv_file)
            split_proba[csv_file.split("/flood_")[2]] = np.mean(calculate_proba(labels))
        print (split_proba)    
    
    else: print ('There are only 3 question') 
           
if __name__ == "__main__":
    args = argParser.parse_args()
    main(args)

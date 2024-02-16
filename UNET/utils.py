from q4_baseline import test_thresh
from q4_xgb import q4_xgb
from UNET.train import train
import torch
import numpy as np
import matplotlib.pyplot as plt
from UNET.dataset import load_imgs
import skimage
import os
import UNET.model
import segmentation_models_pytorch as smp
from collections import OrderedDict
from UNET import config

def load_lightning_checkpoint(checkpoint):
    """function to help load torch lightning checkpoint because of using smp"""
   
    model = smp.create_model('unet',
                         encoder_name = "resnet50",
                         encoder_weights = "imagenet",
                         in_channels = 23,
                         classes = 1)

    state_dict = torch.load(checkpoint)['state_dict']
    pl_state_dict = OrderedDict([(key[6:], state_dict[key]) for key in state_dict.keys()])

    model.load_state_dict(pl_state_dict)

    return UNET.model.S2Model(model, learning_rate=0.1)

def save_predictions(path,preds,iou,name):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path,name+'_predictions.npy'), 'wb') as f:
        np.save(f, preds)
    with open(os.path.join(path,name+'_iou.txt'), 'w+') as f:
        f.write(str(iou))

def calc_IOU(pred,lbl):
    intersection = (pred * lbl).sum()
    union = pred.sum() + lbl.sum() - intersection + np.e**-10
    IOU =  intersection / union
    if IOU <0:
        IOU=0
    return IOU

def check_IOU(loader, model, device="gpu"):
    """given a dataloader and and model - return IOU and prediction for future ploting"""
    
    IOU_list = 0
    UNET_predictions = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            IOU = calc_IOU(preds,y)
            IOU_list+=IOU

            UNET_predictions.append(preds.cpu().squeeze())

    print(f"IOU score: {IOU_list/len(loader)}")
    model.train()
    return IOU_list/len(loader),UNET_predictions

def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band-band_min)/((band_max - band_min)))


def normalize_img(img):
    """Normlize img of 3 bands so they can ploted as a composite"""
    red_n = normalize(img[:,:,3])
    green_n = normalize(img[:,:,2])
    blue_n = normalize(img[:,:,1])
    rgb_composite_n= np.dstack((red_n, green_n, blue_n))
    return (rgb_composite_n)


def plot_random_test(rgbs,labels ,baselines,xgbs,unets):
    # plot rgb image, true label, predicted label
    fig, axs = plt.subplots(1,5,figsize=(16,4))

    #load test split images
    rgbs = [np.moveaxis(skimage.io.imread(img),0,-1) for img in rgbs]
    labels = [skimage.io.imread(lbl) for lbl in labels]

    #select random image
    rand_int = np.random.randint(len(rgbs))

    rgb = rgbs[rand_int]
    axs[0].imshow(normalize_img(rgb))
    axs[0].title.set_text('RGB composite')
    axs[1].imshow(labels[rand_int]>0)
    axs[1].title.set_text('True labels')
    axs[2].imshow(baselines[rand_int])
    axs[2].title.set_text(("Baseline prediction" +'\n'+ str(calc_IOU(baselines[rand_int],labels[rand_int]>0))))
    axs[3].imshow(xgbs[rand_int])
    axs[3].title.set_text(('xgb prediction' + '\n' + str(calc_IOU(xgbs[rand_int],labels[rand_int]>0))))
    axs[4].imshow(unets[rand_int])
    axs[4].title.set_text(('UNET prediction' + '\n' + str( calc_IOU(unets[rand_int],labels[rand_int]>0))))
    fig.tight_layout()
import segmentation_models_pytorch as smp
from pytorch_lightning import Trainer
from lightning.pytorch import seed_everything
seed_everything(42, workers=True)
from pytorch_lightning.loggers import TensorBoardLogger
from UNET import dataset
from UNET import config
import numpy as np
import skimage
from UNET import model
from pytorch_lightning.callbacks import ModelCheckpoint


def train():
    # calculate mean and std as in q1 to normalize the data - using the training split
    train_path = config.TRAIN_PATH
    imgs,_ = dataset.load_imgs(train_path)
    imgs_stack = []
    for img in imgs:
        imgs_stack.append(skimage.io.imread(img))
    imgs_stack = np.stack(imgs_stack)
    imgs_stack = np.moveaxis(imgs_stack,0,-1) #move bands to last axis

    #calculate mean and std
    means = []
    stds = []
    for band in range(imgs_stack.shape[-1]):
        means.append(imgs_stack[:,:,:,band].mean())
        stds.append(imgs_stack[:,:,:,band].std())
    means = np.mean(means)
    stds = np.mean(stds)

    batch_size = config.BATCH_SIZE
    min_epochs = config.MIN_EPOCHS
    max_epochs = config.MAX_EPOCHS
    num_workers= config.NUM_WORKERS


    #datamodule
    s2DM = dataset.S2DataModule(config.TRAIN_PATH,config.VALID_PATH,config.TEST_PATH,
                        means,stds,batch_size,num_workers)
    
    #unet_model
    base_model = smp.Unet(
        encoder_name="resnet50",
        in_channels=23,
        classes=1,
    )
    s2_model = model.S2Model(base_model ,learning_rate=config.LEARNING_RATE )
        
    #pytorch lightning trainer class
    logger = TensorBoardLogger("tb_logs", name="S2_model_google")
    model_checkpoint = ModelCheckpoint(save_top_k=1, mode="min", monitor="valid/loss", save_last=True)
    # Initialize a trainer
    trainer = Trainer(
        accelerator=config.ACCELERATOR,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        log_every_n_steps=10,
        precision=config.PRECISION,
        enable_progress_bar=True,
        logger=logger,
        callbacks = [model_checkpoint]
    )

    # Train the model
    return trainer, s2_model, s2DM
import torch
import torch.nn as nn
import pytorch_lightning as pl

class S2Model(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.learning_rate)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        #learning rate scheduler
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor": "train/loss",
                "interval": "epoch", # step means "batch" here, default: epoch   # New!
                "frequency": 1, # default
            },
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss,on_epoch=True,on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("valid/loss", loss,prog_bar=True, on_epoch=True,on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self(batch)
    
    def forward(self, x):
        return self.model(x)

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import lightning.pytorch as pl


class ENCODER(pl.LightningModule):

    def __init__(self,
                 input_dims = 3,
                 hidden_dims = 64,
                 codebook_dim = 128,
                 num_res_blocks = 2,
                 ):
        super().__init__()
        blocks = [
            nn.Conv2d(input_dims , hidden_dims//2 , kernel_size=7 , stride=5 ,padding=2 ),
            nn.BatchNorm2d(hidden_dims//2),
            nn.Conv2d(hidden_dims//2 , hidden_dims , kernel_size=3 ,stride=1 ,padding=1),
            nn.Tanh()
        ]
        self.blocks = nn.Sequential(*blocks)
        

    def forward(self, x):
        return self.blocks(x)



import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import lightning.pytorch as pl


class DECODER(pl.LightningModule):

    def __init__(self,
                 output_dims = 3,
                 hidden_dims = 64,
                 codebook_dim = 128,
                 num_res_blocks = 2,
                 ):
        super().__init__()
        blocks = [
            nn.ConvTranspose2d(hidden_dims , hidden_dims//2 , kernel_size=7 , stride=5 ,padding=2 ),
            nn.BatchNorm2d(hidden_dims//2),
            nn.ConvTranspose2d(hidden_dims//2 , output_dims , kernel_size=3 ,stride=1 ,padding=1),
            nn.Tanh()
        ]
        self.blocks = nn.Sequential(*blocks)
        

    def forward(self, x):
        return self.blocks(x)


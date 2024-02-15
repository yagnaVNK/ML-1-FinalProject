
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import lightning.pytorch as pl


class DECODER(nn.Module):

    def __init__(self,
                 in_feat_dim=128,
                 out_feat_dim = 3,
                 hidden_dim = 64,
                 codebook_dim = 128,
                 num_res_blocks = 2,
                 ):
        super().__init__()
        blocks = [
            nn.ConvTranspose2d(in_feat_dim , hidden_dim , kernel_size=3 , stride=1 ,padding=1 ,dtype=torch.float32),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_dim , hidden_dim//2 , kernel_size=7 , stride=5 ,padding=2 ,dtype=torch.float32),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ConvTranspose2d(hidden_dim//2 , out_feat_dim , kernel_size=3 ,stride=1 ,padding=1,dtype=torch.float32),
            nn.Tanh()
        ]
        self.blocks = nn.Sequential(*blocks)
        

    def forward(self, x):
        return self.blocks(x)

class DECODER1(nn.Module):
    def __init__(
        self, in_feat_dim, out_feat_dim, hidden_dim=128, very_bottom=False,
    ):
        super().__init__()
        self.very_bottom = very_bottom
        self.out_feat_dim = out_feat_dim 
        blocks = [nn.Conv2d(in_feat_dim, hidden_dim, kernel_size=3, padding=1,dtype=torch.float32), nn.Tanh()]
        
        blocks.extend([
                Upsample(),
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1,dtype=torch.float32),
                nn.Tanh(),
                nn.Conv2d(hidden_dim // 2, out_feat_dim, kernel_size=3, padding=1,dtype=torch.float32),
        ])

        if very_bottom is True:
            blocks.append(nn.Tanh())       
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.blocks(x)
    

class Upsample(nn.Module):

    def __init__(self, scale_factor=2):

        super().__init__()

        self.scale_factor = scale_factor



    def forward(self, x):

        return F.interpolate(x, scale_factor=self.scale_factor)
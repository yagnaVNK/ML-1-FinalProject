
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import lightning.pytorch as pl


class ENCODER(nn.Module):

    def __init__(self,
                 in_feat_dim = 3,
                 hidden_dim = 64,
                 codebook_dim = 128,
                 num_res_blocks = 2,
                 ):
        super().__init__()
        blocks = [
            nn.Conv2d(in_feat_dim , hidden_dim//2 , kernel_size=7 , stride=5 ,padding=2,dtype=torch.float32 ),
            nn.BatchNorm2d(hidden_dim//2),
            nn.Conv2d(hidden_dim//2 , hidden_dim , kernel_size=3 ,stride=1 ,padding=1,dtype=torch.float32),
            nn.Tanh()
        ]
        self.blocks = nn.Sequential(*blocks)
        
        

    def forward(self, x):
        return self.blocks(x)

class ENCODER1(nn.Module):

    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=0, batch_norm=1):
        super().__init__()
        blocks = [
            nn.Conv2d(in_feat_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1,dtype=torch.float32),
            nn.Tanh(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1,dtype=torch.float32),
            nn.Tanh(),
        ]

        blocks.append(nn.Conv2d(hidden_dim, codebook_dim, kernel_size=1,dtype=torch.float32))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.float()
        return self.blocks(x)
        
class ENCODER2(nn.Module):

    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=0, batch_norm=1):
        super().__init__()
        blocks = [
            nn.Conv2d(in_feat_dim, hidden_dim // 2, kernel_size=7, stride=4, padding=2,dtype=torch.float32),
            nn.Tanh(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1,dtype=torch.float32),
            nn.Tanh(),
        ]

        blocks.append(nn.Conv2d(hidden_dim, codebook_dim, kernel_size=1,dtype=torch.float32))
        if(batch_norm):
            blocks.append(nn.BatchNorm1d(codebook_dim))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.float()
        return self.blocks(x)
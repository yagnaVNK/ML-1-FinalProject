import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
import numpy as np
import pandas as pd

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from hae_lightning import HAE
from scipy import interpolate
from scipy import signal as sp
import lightning.pytorch as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='HQA Signal Processing Model')
    parser.add_argument('--EPOCHS', type=int, default=30, help='Number of epochs')
    parser.add_argument('--num_iq_samples', type=int, default=1024, help='Number of IQ samples')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--codebook_slots', type=int, default=256, help='Number of codebook slots')
    parser.add_argument('--codebook_dim', type=int, default=64, help='each codebook dimension')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks')
    parser.add_argument('--KL_coeff', type=float, default=0.1, help='KL coefficient')
    parser.add_argument('--CL_coeff', type=float, default=0.005, help='CL coefficient')
    parser.add_argument('--Cos_coeff', type=float, default=0.7, help='Cosine coefficient')
    parser.add_argument('--batch_norm', type=int, default=1, help='Use batch normalization')
    parser.add_argument('--codebook_init', type=str, default='normal', help='Codebook initialization method')
    parser.add_argument('--reset_choice', type=int, default=1, help='Reset choice')
    parser.add_argument('--cos_reset', type=int, default=1, help='Reset cos_coeff for further layers')
    parser.add_argument('--version', type=int, default=1, help='Which version of the checkpoint to run')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    num_classes = len(classes)
    training_samples_per_class = 4000
    valid_samples_per_class = 1000
    test_samples_per_class = 1000
    num_workers=32

    EPOCHS = args.EPOCHS
    num_iq_samples = args.num_iq_samples
    layers = args.layers
    codebook_slots = args.codebook_slots
    codebook_dim = args.codebook_dim
    num_res_blocks = args.num_res_blocks
    KL_coeff = args.KL_coeff
    CL_coeff = args.CL_coeff
    Cos_coeff = args.Cos_coeff
    batch_norm = args.batch_norm
    codebook_init = args.codebook_init
    reset_choice = args.reset_choice
    cos_reset = args.cos_reset
    version = args.version
    
    
    codebook_visuals_dir = f'Codebook_visualizations/No_norm_Visuals_HQA_Sig_1D_{codebook_init}_BN{batch_norm}_reset{reset_choice}_res{num_res_blocks}_cosReset{cos_reset}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}_iq{num_iq_samples}_codebookSlots{codebook_slots}_codebookDim{codebook_dim}_{layers}layer/version_{version}'


    batch_size = 64

    print(f'EPOCHS : {EPOCHS}')
    print(f'num_iq_samples : {num_iq_samples}')
    print(f'layers : {layers}')
    print(f'codebook_slots : {codebook_slots}')
    print(f'num_res_blocks : {num_res_blocks}')
    print(f'KL_coeff : {KL_coeff}')
    print(f'CL_coeff : {CL_coeff}')
    print(f'Cos_coeff : {Cos_coeff}')
    print(f'batch_norm : {batch_norm}')
    print(f'codebook_init : {codebook_init}')
    print(f'reset_choice : {reset_choice}')
    print(f'cos_reset : {cos_reset}')
    print(f'version : {version}')
    print(f'codebook_dim : {codebook_dim}')
    

    training_dataset = MNIST(root='./data',train=True,download=True,transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(24),
                transforms.ToTensor()]))
    validation_dataset = MNIST(root='./data',train=False,download=True,transform=transforms.Compose([
                transforms.CenterCrop(24),
                transforms.ToTensor()]))

    training_dataloader = DataLoader(training_dataset,batch_size=128)
    validation_dataloader = DataLoader(validation_dataset,batch_size=128)
    
    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]

    model_save_path=os.path.join(f'Saved_models/', f"No_Norm_HQA_Sig_1D_{codebook_init}_BN{batch_norm}_reset{reset_choice}_res{num_res_blocks}_cosReset{cos_reset}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}_iq{num_iq_samples}_codebookSlots{codebook_slots}_codebookDim{codebook_dim}_{layers}layer_version_{version}.ckpt")
    hae = []
    
    for i in range(layers): 
        print(f'training Layer {i}')
        print('==============================================')
        if i == 0:
            hqa = HAE.init_bottom(
                input_feat_dim=1,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=num_res_blocks,
                Cos_coeff = Cos_coeff,
                batch_norm = batch_norm,
                reset_choice = reset_choice,
                layer = i,
                cos_reset = cos_reset
            )
            
        else:
            hqa = HAE.init_higher(
                hqa_prev,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=num_res_blocks,
                Cos_coeff = Cos_coeff,
                batch_norm = batch_norm,
                reset_choice = reset_choice,
                layer = i,
                cos_reset = cos_reset
            )
        logger = TensorBoardLogger(f"tb_logs", name=f"No_Norm_HQA_Sig_1D_{codebook_init}_BN{batch_norm}_reset{reset_choice}_res{num_res_blocks}_cosReset{cos_reset}_Cos{Cos_coeff}_KL{KL_coeff}_C{CL_coeff}_Classes6_e{EPOCHS}_iq{num_iq_samples}_codebookSlots{codebook_slots}_codebookDim{codebook_dim}_{layers}layer_version_{version}")
        
        trainer = pl.Trainer(max_epochs=EPOCHS, 
             logger=logger,  
             devices=1,
             accelerator = 'gpu',
             num_sanity_val_steps=0,
        )
        #hqa = hqa_model[i]
        trainer.fit(hqa, training_dataloader, validation_dataloader)
        hqa_prev = hqa
        hae = hqa
        torch.save(hqa, model_save_path)  
        print(f'saved the model as {model_save_path}')
        print('==========================================')

        print(hae)
    #hqa_model = torch.load(model_save_path)


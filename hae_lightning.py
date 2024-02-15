import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import Encoder as E
import Decoder as D
import GlobalNormalization as G
import FlatCA as Fl


class HAE(pl.LightningModule):
    def __init__(
        self,
        input_feat_dim,
        ze_dim = 128,
        prev_model=None,
        enc_hidden_dim=16,
        dec_hidden_dim=32,
        num_res_blocks=0,
        lr=4e-4,
        decay=True,
        clip_grads=False,
        layer = 0 ,
        Cos_coeff = 0.7,
        batch_norm = 1,
        reset_choice = 0,
        cos_reset = 1
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['prev_model'])
        self.prev_model = prev_model
        self.encoder = E.ENCODER1(input_feat_dim, codebook_dim=ze_dim, hidden_dim =enc_hidden_dim, num_res_blocks=num_res_blocks,batch_norm=1)
        self.decoder = D.DECODER1(
            in_feat_dim = ze_dim,
            hidden_dim=dec_hidden_dim,
            out_feat_dim=input_feat_dim,
            very_bottom=prev_model is None
        )
        self.normalize = G.GlobalNormalization1(ze_dim, scale=True)
        self.out_feat_dim = input_feat_dim
        self.codebook_dim = ze_dim
        self.lr = lr
        self.decay = decay
        self.clip_grads = clip_grads
        self.layer = layer
        self.Cos_coeff = torch.tensor(Cos_coeff)
        self.cos_reset = cos_reset



    def forward(self, x, soft = True):
        x = x.to(torch.float32)
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        z_e_lower_tilde = self.decoder(z_e)
        return z_e_lower_tilde, z_e_lower, z_e
    
    def on_train_start(self):
        if(self.cos_reset):
            self.Cos_coeff = self.Cos_coeff*int(self.layer == 0)
        
    
    def cos_loss(self,x):
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        #z_q, indices, kl, commit_loss = self.codebook(z_e)
        z_e_lower_tilde = self.decoder(z_e)
        cos_loss=torch.max(1-F.cosine_similarity(z_e_lower, z_e_lower_tilde, dim = 1),torch.zeros(z_e_lower.shape[0], z_e_lower.shape[2], device=self.device)).sum(dim=1).mean()
        return cos_loss

    def val_cos_loss(self,x):
        z_e_lower_tilde, z_e_lower, z_e =self(x, soft=False)
        cos_loss=torch.max(1-F.cosine_similarity(z_e_lower, z_e_lower_tilde, dim = 1),torch.zeros(z_e_lower.shape[0], z_e_lower.shape[2], device=self.device)).sum(dim=1).mean()
        return cos_loss


    def get_training_loss(self, x):
        recon, recon_test, lll = self(x)
        #import ipdb; ipdb.set_trace()
        recon_loss = self.recon_loss(self.encode_lower(x), recon)
        cos_loss = self.cos_loss(x)
        dims = np.prod(recon.shape[1:]) 
        loss = self.Cos_coeff*cos_loss/dims + recon_loss/dims 
        return cos_loss, recon_loss, loss 
    
    def get_validation_loss(self, x):
        recon, recon_test, _, = self(x, soft=False)
        recon_loss = self.recon_loss(self.encode_lower(x), recon)
        val_cos_loss = self.val_cos_loss(x)
        dims = np.prod(recon.shape[1:]) # orig_w * orig_h * num_channels
        loss = self.Cos_coeff*val_cos_loss/dims + recon_loss/dims 
        return val_cos_loss, recon_loss, loss  

    def recon_loss(self, orig, recon):
        return F.mse_loss(orig, recon, reduction='none').sum(dim=(1,2)).mean()
    
    def decay_temp_linear(self, step, total_steps, temp_base, temp_min=0.001):
        factor = 1.0 - (step/total_steps)
        return temp_min + (temp_base - temp_min) * factor

    def training_step(self,batch, batch_idx):
        x,_ = batch
        
        
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
    
        cos_loss, recon_loss, loss = self.get_training_loss(x)
    
        optimizer.zero_grad()
        
        self.manual_backward(loss)
        
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        self.log("loss", loss, prog_bar=True)
        self.log("cos_loss", cos_loss, prog_bar=True)
        self.log("recon", recon_loss, prog_bar=True)
        return loss


    def validation_step(self, val_batch, batch_idx):
        x,_ = val_batch
        cos_loss, recon_loss, loss = self.get_validation_loss(x)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)
        self.log("val_cos_loss", cos_loss, prog_bar=False,sync_dist=True)
        self.log("val_recon", recon_loss, prog_bar=False, sync_dist=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x,_ = test_batch
        cos_loss, recon_loss, loss = self.get_validation_loss(x)
        self.log("tst_loss", loss, prog_bar=False)
        self.log("tst_cos_loss", cos_loss, prog_bar=False)
        self.log("tst_recon", recon_loss, prog_bar=False)
        return loss    

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        lr_scheduler = Fl.FlatCA(optimizer, steps=1, eta_min=4e-5)
        return [optimizer], [lr_scheduler]
    
    def encode_lower(self, x):
        if self.prev_model is None:
            return x
        else:
            with torch.no_grad():
                z_e_lower = self.prev_model.encode(x)
                z_e_lower = self.normalize(z_e_lower)
            return z_e_lower
    def encode(self, x):
        with torch.no_grad():
            z_e_lower = self.encode_lower(x)
            z_e = self.encoder(z_e_lower)
        return z_e
        
    def decode_lower(self, z_q_lower):
        with torch.no_grad():
            recon = self.prev_model.decode(z_q_lower)           
        return recon

    def decode(self, z_q):
        with torch.no_grad():
            if self.prev_model is not None:
                z_e_u = self.normalize.unnorm(self.decoder(z_q))
                z_q_lower_tilde = self.prev_model.quantize(z_e_u)
                recon = self.decode_lower(z_q_lower_tilde)
            else:
                recon = self.decoder(z_q)
        return recon

    def reconstruct(self, x):
        return self.decode(self.encode(x))
    
    def reconstruct_from_z_e(self, z_e):
        return self.decode(z_e)
    
    def __len__(self):
        i = 1
        layer = self
        while layer.prev_model is not None:
            i += 1
            layer = layer.prev_model
        return i

    def __getitem__(self, idx):
        max_layer = len(self) - 1
        if idx > max_layer:
            print(idx[0])
            raise IndexError("layer does not exist")

        layer = self
        for _ in range(max_layer - idx):
            layer = layer.prev_model
        return layer

    def parameters(self, prefix="", recurse=True):
        for module in [self.encoder, self.decoder]:
            for name, param in module.named_parameters(recurse=recurse):
                yield param
    
    @classmethod
    def init_higher(cls, prev_model, **kwargs):
        model = HAE(prev_model.codebook_dim, prev_model=prev_model, **kwargs)
        model.prev_model.eval()
        return model
    
    @classmethod
    def init_bottom(cls, input_feat_dim, **kwargs):
        model = HAE(input_feat_dim,prev_model=None, **kwargs)
        return model



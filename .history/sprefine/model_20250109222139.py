import scanpy as sc
import pandas as pd
import lightning

import torch
a = torch.tensor(0).cuda()

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
import sklearn.model_selection

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

class Cell_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                nn.ReLU(), 
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(), 
                                nn.Linear(hidden_dim, hidden_dim),
                               )

    def forward(self, x):
        return self.l1(x)
    
class Gene_Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                nn.ReLU(), 
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(), 
                                nn.Linear(hidden_dim, hidden_dim),
                               )

    def forward(self, x):
        return self.l1(x)

def filter_nonzero(x):
    return F.softplus(x)

class sprefine_pretrain(L.LightningModule):
    def __init__(self, encoder1, encoder2, gene_emb, train_index, eta=1e-4, nonneg=False):
        super().__init__()
        self.cellencoder = encoder1
#         self.cellencoder.apply(init_weights)
        
        self.geneencoder = encoder2 
#         self.geneencoder.apply(init_weights)
        self.gene_emb = gene_emb
        self.train_index = train_index
        self.eta = eta
        
        if nonneg:
            self.id = nn.Softplus()
        else:
            self.id = nn.Identity()
            
        self.droprate = 0.55

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch

        batch = x.shape[0]
        ngenes = x.shape[1]
        droprate = self.droprate * 1.1
        # we model the sampling zeros (dropping 30% of the reads)
        res = torch.poisson((x * (self.droprate / 2))).int()
        # we model the technical zeros (dropping 50% of the genes)
        notdrop = (
            torch.rand((batch, ngenes), device=x.device) >= (self.droprate / 2)
        ).int()
        mat = (x - res) * notdrop
        out_data = torch.maximum(
            mat, torch.zeros((1, 1), device=x.device, dtype=torch.int)
        )
    
        z_cell = self.cellencoder(out_data)
        z_gene = self.geneencoder(self.gene_emb)
        
        out_final = torch.matmul(z_cell, z_gene.T)
        out_final = self.id(out_final)
        
        loss = F.mse_loss(out_final[:,self.train_index], y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch


        batch = x.shape[0]
        ngenes = x.shape[1]
        droprate = self.droprate * 1.1
        # we model the sampling zeros (dropping 30% of the reads)
        res = torch.poisson((x * (self.droprate / 2))).int()
        # we model the technical zeros (dropping 50% of the genes)
        notdrop = (
            torch.rand((batch, ngenes), device=x.device) >= (self.droprate / 2)
        ).int()
        mat = (x - res) * notdrop
        out_data = torch.maximum(
            mat, torch.zeros((1, 1), device=x.device, dtype=torch.int)
        )
        
        z_cell = self.cellencoder(out_data)
        z_gene = self.geneencoder(self.gene_emb)
        
        out_final = torch.matmul(z_cell, z_gene.T)
        out_final = self.id(out_final)
        
        val_loss = F.mse_loss(out_final[:,self.train_index], y)
        self.log('val_loss', val_loss, on_epoch=True, on_step=False)
        return val_loss

        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch

        out_data = x
        z_cell = self.cellencoder(out_data)
        z_gene = self.geneencoder(self.gene_emb)
        
        out_final = torch.matmul(z_cell, z_gene.T)
        out_final = self.id(out_final)
        
        test_loss = F.mse_loss(out_final[:,self.train_index], y)
        self.log('test_loss', test_loss)
        return test_loss
        
    def forward(self, x):
        z_cell = self.cellencoder(x)

        z_gene = self.geneencoder(self.gene_emb)
        
        out_final = torch.matmul(z_cell, z_gene.T)
        out_final = self.id(out_final)
        
        return out_final

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.eta)
        return optimizer
    



class sprefine(L.LightningModule):
    def __init__(self, encoder1, encoder2, gene_emb, train_index, eta=1e-4, nonneg=False):
        super().__init__()
        self.cellencoder = encoder1
        self.geneencoder = encoder2 
        self.gene_emb = gene_emb
        self.train_index = train_index
        self.eta = eta
        
        if nonneg:
            self.id = nn.Softplus()
        else:
            self.id = nn.Identity()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z_cell = self.cellencoder(x)
        z_gene = self.geneencoder(self.gene_emb)
        
        out_final = torch.matmul(z_cell, z_gene.T)
        out_final = self.id(out_final)
        
        loss = F.mse_loss(out_final[:,self.train_index], y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z_cell = self.cellencoder(x)
        z_gene = self.geneencoder(self.gene_emb)
        
        out_final = torch.matmul(z_cell, z_gene.T)
        out_final = self.id(out_final)
        
        val_loss = F.mse_loss(out_final[:,self.train_index], y)
        self.log('val_loss', val_loss, on_epoch=True, on_step=False)
        return val_loss

        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z_cell = self.cellencoder(x)
        z_gene = self.geneencoder(self.gene_emb)
        
        out_final = torch.matmul(z_cell, z_gene.T)
        out_final = self.id(out_final)
        
        test_loss = F.mse_loss(out_final[:,self.train_index], y)
        self.log('test_loss', test_loss)
        return test_loss
        
    def forward(self, x):
        z_cell = self.cellencoder(x)
        z_gene = self.geneencoder(self.gene_emb)
        
        out_final = torch.matmul(z_cell, z_gene.T)
        out_final = self.id(out_final)
        
        return out_final

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.eta)
        return optimizer
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

class LitAutoEncoder(L.LightningModule):
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
#         x = x.view(x.size(0), -1)
#         a = torch.poisson(x).to(x.device)
#         mask = torch.full((x.shape[0],x.shape[1]), ratio_mask).to(x.device) 
#         out_data = (x-a)*mask 
#         out_data = out_data * ((out_data > 0) * 1)

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
#         x = x.view(x.size(0), -1)
#         z_cell = self.cellencoder(x)
#         a = torch.poisson(x).to(x.device)
#         mask = torch.full((x.shape[0],x.shape[1]), ratio_mask).to(x.device) 
#         out_data = (x-a)*mask 
#         out_data = out_data * ((out_data > 0) * 1)

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
#         x = x.view(x.size(0), -1)
#         z_cell = self.cellencoder(x)
#         a = torch.poisson(x).to(x.device)
#         mask = torch.full((x.shape[0],x.shape[1]), ratio_mask).to(x.device) 
#         out_data = (x-a)*mask 
#         out_data = out_data * ((out_data > 0) * 1)
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
#         a = torch.poisson(x).to(x.device)
#         mask = torch.full((x.shape[0],x.shape[1]), 0.55).to(x.device) 
#         out_data = torch.max((x-a)*mask, 0).to(x.device)
#         z_cell = self.cellencoder(out_data)

#         a = torch.poisson(x).to(x.device)
#         mask = torch.full((x.shape[0],x.shape[1]), 0.55).to(x.device) 
#         out_data = (x-a)*mask 
#         out_data = out_data * ((out_data > 0) * 1)
#         z_cell = self.cellencoder(out_data)
        
        z_gene = self.geneencoder(self.gene_emb)
        
        out_final = torch.matmul(z_cell, z_gene.T)
        out_final = self.id(out_final)
        
        return out_final

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.eta)
        return optimizer
    

adata = sc.read_h5ad("/home/tl688/pitl688/ref_free_imp/adata_stimage_hest.h5ad")


# Convert the expression matrix to a pandas DataFrame
# If X is a sparse matrix, convert to a dense array first
import scipy.sparse as sp
if sp.issparse(adata.X):
    expr_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
else:
    expr_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

# Identify duplicated rows
# keep='first' ensures we mark duplicates after the first occurrence
duplicated_mask = expr_df.duplicated(keep='first')

# Subset the AnnData to keep only the non-duplicated rows
adata_no_duplicates = adata[~duplicated_mask, :].copy()


import numpy as np

# adata_dense = adata.X.toarray()

sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)


sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

adata_emb = sc.read_h5ad("/home/tl688/project/seq2cells_data/spatial_data/adata_spatial_logcounts_waitimpute.h5ad")

df_id = adata_emb.obs 


df_id = df_id[df_id['add_id'].isin(adata.var_names)]

adata_train =  adata

train_index = [int(i) for i in df_id.index]


adata_train = adata_train[:,df_id['add_id'].values]

import sklearn.model_selection
import numpy as np
import random
np.random.seed(2024)
random.seed(2024)

sorted(set(adata_train.obs['batch']))

train_name, test_name = sklearn.model_selection.train_test_split(sorted(set(adata_train.obs['batch'])))

valid_name = test_name

cell_enc = Cell_Encoder(input_dim=21648, hidden_dim=64)

gene_enc = Gene_Encoder(input_dim=3072,hidden_dim=64)

model = LitAutoEncoder(encoder1=cell_enc, encoder2=gene_enc, eta=1e-4, gene_emb=torch.FloatTensor(adata_emb.obsm['seq_embedding']).cuda(), train_index=train_index, nonneg=True)

model

X_tr, X_val, X_test, y_tr, y_val, y_test = torch.FloatTensor(adata_train[adata_train.obs['batch'].isin(train_name)].X.toarray()),torch.FloatTensor(adata_train[adata_train.obs['batch'].isin(valid_name)].X.toarray()), torch.FloatTensor(adata_train[adata_train.obs['batch'].isin(test_name)].X.toarray()),torch.FloatTensor(adata_train[adata_train.obs['batch'].isin(train_name)].X.toarray()),torch.FloatTensor(adata_train[adata_train.obs['batch'].isin(valid_name)].X.toarray()),torch.FloatTensor(adata_train[adata_train.obs['batch'].isin(test_name)].X.toarray())

train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
valid_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=512, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=512, num_workers=1)


# train with both splits
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=100)], max_epochs=1000)
trainer.fit(model, train_loader, valid_loader, )

best_checkpoint = trainer.checkpoint_callback.best_model_path
print(best_checkpoint)
model = LitAutoEncoder.load_from_checkpoint(best_checkpoint, encoder1=cell_enc, encoder2=gene_enc, eta=1e-4, gene_emb=torch.FloatTensor(adata_emb.obsm['seq_embedding']).cuda(), train_index=train_index)

adata_train.write_h5ad("/home/tl688/pitl688/ref_free_imp/adata_fulllist_imputation.h5ad")



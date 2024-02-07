import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 

import random
import numpy as np

from .layers import *


class RTransformer(nn.Module):
    def __init__(self, dim_input, dim_output, dim_embedding, n_layers, n_heads, dim_k, dim_v, dim_hidden, t0, drop=0.1, drop_pred=0.5, learned_pos=False):
        super(RTransformer, self).__init__()
        self.encoder = Encoder(dim_input=dim_input, dim_embedding=dim_embedding, n_layers=n_layers, n_heads=n_heads, dim_k=dim_k, dim_v=dim_v, dim_hidden=dim_hidden, drop=drop, learned_pos=learned_pos)
        self.decoder = Decoder(dim_output=dim_output, dim_embedding=dim_embedding, n_layers=n_layers, n_heads=n_heads, dim_k=dim_k, dim_v=dim_v, dim_hidden=dim_hidden, drop=drop, learned_pos=learned_pos)
        self.predictor = Predictor(dim_embedding * t0 * 1, drop=drop_pred)
        self.fc = nn.Linear(dim_embedding, dim_output)
        # self.domain = DomainPredictor(dim_embedding * t0)
    
    def forward(self, input_seq, output_seq, return_enc=False):
        dec_mask = mask_subsequent(output_seq)
        # if return_emb:
        #     enc_out, emb_x = self.encoder(input_seq)#, return_emb_x=return_emb)
        # else:
        #     enc_out = self.encoder(input_seq) #, return_emb_x=return_emb)
        enc_out = self.encoder(input_seq)
        dec_out = self.decoder(output_seq, enc_out, dec_mask)
        # long_out = self.predictor(torch.cat([enc_out.sum(dim=1), dec_out.sum(dim=1), input_seq[:, 0:1, -1]], dim=1))
        long_out = self.predictor(torch.cat([enc_out.view(dec_out.shape[0], -1)], dim=1))
        dec_out = self.fc(dec_out)
        if return_enc:
            return dec_out, long_out, enc_out
        return dec_out, long_out # batch_size, l, n_trg
    
    
    
class CTransformer(nn.Module):
    def __init__(self, dim_input, dim_output, dim_embedding, n_layers, n_heads, dim_k, dim_v, dim_hidden, dim_class, t0, drop=0.1, drop_pred=0.5, learned_pos=False, use_treat_in_pred=False):
        super(CTransformer, self).__init__()
        self.use_treat_in_pred = use_treat_in_pred
        self.encoder = Encoder(dim_input=dim_input, dim_embedding=dim_embedding, n_layers=n_layers, n_heads=n_heads, dim_k=dim_k, dim_v=dim_v, dim_hidden=dim_hidden, drop=drop, learned_pos=learned_pos)
        self.decoder = Decoder(dim_output=dim_output, dim_embedding=dim_embedding, n_layers=n_layers, n_heads=n_heads, dim_k=dim_k, dim_v=dim_v, dim_hidden=dim_hidden, drop=drop, learned_pos=learned_pos)
        if use_treat_in_pred:
            self.predictor = CPredictor(dim_embedding * t0 + t0, dim_class, drop=drop_pred)
        else:
            self.predictor = CPredictor(dim_embedding * t0, dim_class, drop=drop_pred)
        self.fc = nn.Linear(dim_embedding, dim_output)
    
    def forward(self, input_seq, output_seq, return_latent=False):
        dec_mask = mask_subsequent(output_seq)
        enc_out = self.encoder(input_seq)
        dec_out = self.decoder(output_seq, enc_out, dec_mask)
        if self.use_treat_in_pred:
            # long_input = torch.cat([enc_out, dec_out, input_seq[:, :, -1:]], dim=2)
            long_input = torch.cat([enc_out, input_seq[:, :, -1:]], dim=2)
        else:
            # long_input = torch.cat([enc_out, dec_out], dim=2)
            long_input = torch.cat([enc_out], dim=2)
        dec_out = self.fc(dec_out)
        
        long_out = self.predictor(long_input.view(long_input.shape[0], -1))
        if return_latent:
            return dec_out, long_out, enc_out
        return dec_out, long_out # batch_size, l, n_trg
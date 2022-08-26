import timm
import torch
from torch import nn


#2d position_embedding same as MOCOv3
def build_2d_sincos_position_embedding(model,temperature=10000.):
    h, w = model.patch_embed.grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert model.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = model.embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    assert model.num_prefix_tokens  == 1, 'Assuming one and only one token, [cls]'
    pe_token = torch.zeros([1, 1, model.embed_dim], dtype=torch.float32)
    model.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    model.pos_embed.requires_grad = False





def define_Vit(backbone,numclass,pretrain_arg=False):
    inputsizebk = {'vit_small_patch16_224': 384,'vit_base_patch16_224':768}
    model = timm.create_model(backbone, pretrained=pretrain_arg)
    build_2d_sincos_position_embedding(model)
    model.head=nn.Linear(inputsizebk[backbone],numclass)
    return model

import torch
import os
import math
import numpy as np
import random
import torch.nn.functional as F
from einops import rearrange

class Conv_Token_Emb(torch.nn.Module):
  def __init__(self, in_channel, emb_dim, k_size, stride, padding, pooling_size = 3, pooling_stride = 2, pooling_pad = 1):
    super(Conv_Token_Emb, self).__init__()
    self.conv = torch.nn.Conv2d(in_channels = in_channel, out_channels = emb_dim, kernel_size = k_size, stride = stride, padding = padding)
    self.relu = torch.nn.ReLU()
    self.pool = torch.nn.MaxPool2d(kernel_size = pooling_size, stride = pooling_stride, padding = pooling_pad)

  def forward(self, x):
    out = self.relu(self.conv(x))
    out = self.pool(out) 
    return out
  
class Conv_Block(torch.nn.Module):
  def __init__(self, num_layer, img_size, in_channel, emb_dim, k_size, stride, padding):
    super(Conv_Block, self).__init__()
    self.num_layer = num_layer
    self.img_size = img_size
    self.conv_blk = torch.nn.ModuleList(
        [Conv_Token_Emb(in_channel[i], emb_dim[i], k_size[i], stride[i], padding[i]) for i in range(num_layer)]
        )

  def seq_len(self):
    return self.forward(torch.zeros(1,3,self.img_size, self.img_size)).shape[2]

  def forward(self, x):
    out = self.conv_blk[0](x)
    for i in range(1, self.num_layer):
      out = self.conv_blk[i](out)
    return out
  
class Attention(torch.nn.Module):
  def __init__(self, emb_dim, head, dropout = 0):
    super(Attention, self).__init__()
    self.emb_dim = emb_dim
    self.head = head
    self.softmax = torch.nn.Softmax(dim = -1)
    self.dropout = torch.nn.Dropout(p = dropout)

  #sent k.T in (transpose k before sent in forward)
  def forward(self, q, k, v):
    qk = torch.matmul(q, k) / math.sqrt(self.emb_dim//self.head)
    att_w = self.dropout(self.softmax(qk))
    out = torch.matmul(att_w, v)
    return out
  
class Multi_Head_ATT(torch.nn.Module):
    def __init__(self, emb_dim, multi_head = 1, dropout = 0):
      super(Multi_Head_ATT,self).__init__()
      self.head = multi_head
      self.emb_dim = emb_dim
      self.q_att = torch.nn.Linear(emb_dim, emb_dim, bias = False) 
      self.k_att = torch.nn.Linear(emb_dim, emb_dim, bias = False) 
      self.v_att = torch.nn.Linear(emb_dim, emb_dim, bias = False) 
      self.attention = Attention(emb_dim, multi_head, dropout = dropout)
      self.LN = torch.nn.LayerNorm(emb_dim)
      self.WO = torch.nn.Linear(emb_dim, emb_dim)
      self.dropout = torch.nn.Dropout(p = dropout)

    def forward(self, q,k,v):
      res = q
      seq_len = q.shape[1]
      q, k, v = self.LN(q), self.LN(k), self.LN(v)
      if self.head == 1:
        q, k, v = self.q_att(q), self.k_att(k).permute(0,2,1), self.v_att(v)
        out = self.attention(q, k, v)
      else:
        # (b_s, seq_len, head, emb//head) > (b_s, head, seq_len, emb_dim//head)
        q = self.q_att(q).view(-1,seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3)
        k = self.k_att(k).view(-1,seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3).permute(0,1,3,2)
        v = self.v_att(v).view(-1,seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3)
        out = self.attention(q, k, v).permute(0,2,1,3).contiguous().view(-1,seq_len,self.emb_dim)
      out = self.WO(out)   
      out = self.dropout(out)
      out = out + res
      return out
 
class Feed_Forward(torch.nn.Module): 
  def __init__(self, emb_dim, dim_expan = 4, dropout = 0):
    super(Feed_Forward,self).__init__()
    self.w1 = torch.nn.Linear(emb_dim, dim_expan*emb_dim)
    self.w2 = torch.nn.Linear(dim_expan*emb_dim, emb_dim)
    self.gelu = torch.nn.GELU()
    self.LN = torch.nn.LayerNorm(emb_dim)
    self.dropout = torch.nn.Dropout(p = dropout)
  def forward(self,x):
    res = x
    x = self.LN(x)
    out = self.dropout(self.gelu(self.w1(x)))
    out = self.w2(out)
    out = self.dropout(out)
    out = out + res
    return out

class Encoder(torch.nn.Module):
  def __init__(self, num_layer, emb_dim, head, dim_expan = 4, dropout = 0):
    super(Encoder, self).__init__()
    self.num_layer = num_layer
    self.attention = Multi_Head_ATT(emb_dim, multi_head = head, dropout = dropout)
    self.FF = Feed_Forward(emb_dim, dim_expan = dim_expan, dropout = dropout)
    self.connect1 = torch.nn.ModuleList([
                                         Multi_Head_ATT(emb_dim, multi_head = head, dropout = dropout) for i in range(num_layer - 1)
                                         ])
    self.connect2 = torch.nn.ModuleList([
                                         Feed_Forward(emb_dim, dim_expan = dim_expan, dropout = dropout) for i in range(num_layer - 1)
                                         ])

  def forward(self, x):
    out = self.FF(self.attention(x, x, x))
    for idx in range(self.num_layer - 1):
      out = self.connect1[idx](out, out, out)
      out = self.connect2[idx](out)
    return out

class Seq_Pooling(torch.nn.Module):
  def __init__(self, emb_dim):
    super(Seq_Pooling, self).__init__()
    self.linear = torch.nn.Linear(emb_dim, 1)
    self.softmax = torch.nn.Softmax(dim = -1)

  def forward(self, x):
    out = self.linear(x) # (b_s, seq_len , 1)
    out = self.softmax(out.permute(0,2,1)) # (b_s, 1, seq_len)
    out = torch.bmm(out, x) # (b_s, 1, emb_dim)
    return out

class CCTransformer(torch.nn.Module):
  def __init__(self,
               img_size,
               head, 
               class_num, 
               in_channel = [3, 64], 
               emb_dim = [64, 128], 
               k_size = [3, 3],
               stride = [1, 1], 
               padding = [1, 1], 
               conv_layer = 2, 
               num_layer = 4,
               dim_expan = 1,
               dropout = 0):
    
    super(CCTransformer,self).__init__()

    self.conv_blk = Conv_Block(conv_layer ,img_size, in_channel, emb_dim, k_size, stride, padding) # num_layer, img_size, in_channel, emb_dim, k_size, stride, padding, position = True
    seq_len = self.conv_blk.seq_len()
    self.pos_emb = torch.nn.Parameter(torch.zeros(seq_len*seq_len , emb_dim[-1]))
    #self.layer_norm = torch.nn.LayerNorm(emb_dim[-1])
    self.encoder = Encoder(num_layer, emb_dim[-1], head, dim_expan = dim_expan, dropout = dropout) # num_layer, seq_len, emb_dim, head
    self.seq_pooling = Seq_Pooling(emb_dim[-1])
    self.linear = torch.nn.Linear(emb_dim[-1], class_num) #,bias = False

  def forward(self, x):
    out = self.conv_blk(x) # (b_s, c, h, w)
    out = out.permute(0,2,3,1) #(b_s, h, w, c)
    out = rearrange(out, 'b h w c -> b (h w) c')
    out = self.pos_emb.repeat(x.shape[0],1,1) + out
    out = self.encoder(out)
    out = self.seq_pooling(out)
    out = self.linear(out.view(out.shape[0],-1)) 
    return out

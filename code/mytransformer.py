import torch
import copy, math
import torch.nn as nn
import torch.nn.functional as F


# Start with the abstract level

class EncoderDecoder(nn.modules):
    def __init__(self,encoder,decoder):
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,src,tgt):
        memory = self.encoder(src)
        return self.decoder(memory,tgt)
    
class Encoder(nn.modules):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderLayer(nn.modules):
    def __init__(self,attn_norm,ff_norm):
        super(EncoderLayer,self).__init__()
        self.attn_norm = attn_norm
        self.ff_norm = ff_norm
    
    def forward(self,x):
        x = self.attn_norm(x)
        x = self.ff_norm(x)

"""
Almost the same for decoder and decoder layers
"""

class NormLayer(nn.modules):
    def __init__(self,layer,norm):
        super.__init__(NormLayer,self)
        self.layer = layer
        self.norm = norm
    
    def forward(self,x):
        return self.norm(x+self.layer(x))
    
class MultiheadAttention(torch.nn.modules):
    def __init__(self,h,d_model,d_v,mask):
        super(MultiheadAttention,self).__init__()
        self.d_model = d_model
        self.d_v = d_v
        self.h = h
        self.mask = mask
        self.linear_layers = nn.ModuleList([ [nn.Linear(d_model,d_v) for _ in range(3)] for _ in range(h)])
        self.w_0 = nn.Linear(h*d_v,d_model)

    def attention(q,k,v,d_v,mask):
    # This fucntion is used to compute the attention matrix of give matrixs
        weight_matrix = torch.functional.softmax(torch.matmul(q,k.troch.transpose(0,1))/torch.sqrt(d_v))
        if mask:
            weight_matrix.masked_fill(mask == 0, -1e9)
        return weight_matrix.torch.matmul(v)

    def forward(self,x):
        """
        x is the input matrix and linear_layers are a list of linear layer with the shape of (num_head,3)
        """
        attention = []
        for layer_batch in self.linear_layers:
            attention.append(attention(
                x.matmul(layer_batch[0]),
                x.matmul(layer_batch[1]),
                x.matmul(layer_batch[2]),
                x.shape[0]),
                mask = self.mask)
        # now the attention should be a list of n*dv matrix, we should concracate them such that get a n,dv*h matrix 
    
        con_attention = torch.cat(attention, dim=1)

        return con_attention.matmul(self.w_0)

class Feedforward(nn.modules):
    def __init__(self,d_model,d_ff):
        super(Feedforward,self).__init__()
        self.layer1 = nn.Linear(d_model,d_ff)
        self.layer2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    

    
    
        
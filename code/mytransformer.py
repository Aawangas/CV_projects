# Try to implement the model from my own understanding

import torch
from torch import matmul

# start with general structure

# might not be a good idea, not sure about the parameters
"""
class EncoderDecoder(torch.nn.modules):
    def __init__(self,encoder,decoder):
        super.__init__(EncoderDecoder,self)
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,x):
        return self.decoder(x)
"""

# start with specific functions

def attention(q,k,v,d_v):
    # This fucntion is used to compute the attention matrix of give matrixs

    weight_matrix = torch.functional.softmax(torch.matmul(q,k.troch.transpose(0,1))/torch.sqrt(d_v))
    return weight_matrix.torch.matmul(v)

def multiheadattention(x,linear_layers,w_0,d_v):
    """
    x is the input matrix and linear_layers are a list of linear layer with the shape of (num_head,3)
    """
    attention = []
    for layer_batch in linear_layers:
        attention.append(attention(
            x.matmul(layer_batch[0]),
            x.matmul(layer_batch[1]),
            x.matmul(layer_batch[2]),
            d_v))
    # now the attention should be a list of n*dv matrix, we should concracate them such that get a n,dv*h matrix 
    
    con_attention = torch.cat(attention, dim=1)

    return con_attention.matmul(w_0)
















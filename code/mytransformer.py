# Try to implement the model from my own understanding
import math
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

def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([torch.copy.deepcopy(module) for _ in range(N)])



class multiheadattention(torch.nn.modules):
    def __init__(self,linear_layers,w_0):
        super.__init__(multiheadattention,self)
        self.linear_layers = linear_layers
        self.w_0 = w_0

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
                math.sqrt(x.shape[1])))
        # now the attention should be a list of n*dv matrix, we should concracate them such that get a n,dv*h matrix 
    
        con_attention = torch.cat(attention, dim=1)

        return con_attention.matmul(self.w_0)




















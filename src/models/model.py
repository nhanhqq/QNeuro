import torch
from torch import nn
from .quantum_latent import QuantumLatent16,ClassicalLatent16
from .kan_linear import KANLinear
class TinyBiLSTM(nn.Module):
 def __init__(self,num_classes,latent='quantum',depth=2,entanglement='ring',reupload=True,no_bilstm=False,rz_noise_std=.10,input_size=8,hidden_size=6):
  super().__init__();self.no_bilstm=no_bilstm;self.input_size=input_size;self.hidden_size=hidden_size
  if not no_bilstm:
   self.bilstm=nn.LSTM(input_size,hidden_size,1,batch_first=True,bidirectional=True)
   self.temporal_norm=nn.LayerNorm(2*hidden_size)
   self.to_quantum=nn.Linear(2*hidden_size,4)
  else:self.to_quantum=nn.Linear(input_size,4)
  self.latent_kind=latent;self.latent=QuantumLatent16(depth,entanglement,reupload,rz_noise_std=rz_noise_std) if latent=='quantum' else (ClassicalLatent16() if latent=='classical' else nn.Identity());self.classifier=KANLinear(4,num_classes)
 def forward(self,x,return_features=False):
  if self.no_bilstm: temporal=x.mean(1)
  else:
   _,(h,_)=self.bilstm(x);temporal=self.temporal_norm(torch.cat((h[-2],h[-1]),-1))
  angles=torch.pi*torch.tanh(self.to_quantum(temporal));q=self.latent(angles);logits=self.classifier(q)
  return {'temporal':temporal,'angles':angles,'quantum_latent':q,'logits':logits} if return_features else logits
def count_trainable_parameters(m):return sum(p.numel() for p in m.parameters() if p.requires_grad)

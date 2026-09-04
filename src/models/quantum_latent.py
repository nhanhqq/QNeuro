"""Four-qubit CUDA state-vector circuit with learnable data re-uploading."""
import torch
from torch import nn
class QuantumLatent16(nn.Module):
 """VQC with 16 variational rotations and 8 encoding-scale parameters.

 ``encoding_scale[l, q]`` starts at one, so the initial circuit exactly
 matches ordinary data re-uploading.  Each layer/qubit can then learn its own
 input-frequency scale without increasing the number of qubits or circuit
 depth.  Gaussian RZ augmentation is train-only and its standard deviation is
 expressed in radians.
 """
 def __init__(self,depth=2,entanglement='ring',reupload=True,rz_noise_std=0.10):
  super().__init__()
  if rz_noise_std < 0: raise ValueError('rz_noise_std must be non-negative')
  self.depth=depth;self.entanglement=entanglement;self.reupload=reupload;self.rz_noise_std=float(rz_noise_std)
  self.weights=nn.Parameter(.05*torch.randn(depth,4,2))
  self.encoding_scale=nn.Parameter(torch.ones(depth,4))
  basis=torch.arange(16)
  for q in range(4):self.register_buffer(f'z{q}',1-2*((basis&(1<<(3-q)))!=0).float(),persistent=False)
  for c in range(4):
   for t in range(4):
    if c!=t:self.register_buffer(f'p{c}{t}',basis^(((basis&(1<<(3-c)))!=0).long()*(1<<(3-t))),persistent=False)
 def _gate(self,s,g,w):
  b=s.size(0); z=s.reshape(b,2,2,2,2).movedim(w+1,-1);z=torch.einsum('bij,b...j->b...i',g,z);return z.movedim(-1,w+1).reshape(b,16)
 def _ry(self,t):
  c,s=torch.cos(t/2),torch.sin(t/2);return torch.stack((torch.stack((c,-s),-1),torch.stack((s,c),-1)),-2).to(torch.complex64)
 def _rz(self,t):
  z=torch.zeros_like(t,dtype=torch.complex64);return torch.stack((torch.stack((torch.exp(-.5j*t),z),-1),torch.stack((z,torch.exp(.5j*t)),-1)),-2)
 def forward(self,x):
  if not x.is_cuda:raise RuntimeError('QuantumLatent16 is CUDA-only to keep state vector and gradients on GPU')
  s=torch.zeros(x.size(0),16,device=x.device,dtype=torch.complex64);s[:,0]=1
  for l in range(self.depth):
   a=x*self.encoding_scale[l] if self.reupload or l==0 else torch.zeros_like(x)
   # Training-only quantum augmentation: independently encode Gaussian RZ
   # angles for every sample, qubit, and re-uploading block.  Evaluation is
   # deterministic and never consumes target information for augmentation.
   rz_noise=torch.randn_like(a)*self.rz_noise_std if self.training and self.rz_noise_std else torch.zeros_like(a)
   for q in range(4):s=self._gate(s,self._ry(a[:,q]),q);s=self._gate(s,self._rz(rz_noise[:,q]),q);s=self._gate(s,self._ry(self.weights[l,q,0].expand_as(a[:,q])),q);s=self._gate(s,self._rz(self.weights[l,q,1].expand_as(a[:,q])),q)
   pairs=[] if self.entanglement=='none' else ([(0,1),(1,2),(2,3)] if self.entanglement=='linear' else [(0,1),(1,2),(2,3),(3,0)])
   for c,t in pairs:s=s.index_select(1,getattr(self,f'p{c}{t}'))
  p=s.abs().square();return torch.stack([(p*getattr(self,f'z{q}').to(p)).sum(1) for q in range(4)],1)
class ClassicalLatent16(nn.Module):
 def __init__(self):super().__init__();self.layer=nn.Linear(4,4,bias=False)
 def forward(self,x):return torch.tanh(self.layer(x))

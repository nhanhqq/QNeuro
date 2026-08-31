"""Variable-electrode LatentSet-QuanKAN with CUDA-native torch state-vector QNN."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class TinyKANLinear(nn.Module):
    """Compact differentiable spline-like linear layer; grid=3, order=2."""
    def __init__(self, inp, out, grid_size=3, spline_order=2):
        super().__init__(); self.linear=nn.Linear(inp,out); self.grid_size=grid_size
        self.spline=nn.Parameter(torch.empty(out,inp,grid_size)); nn.init.normal_(self.spline,std=.02)
    def forward(self,x):
        u=torch.tanh(x).unsqueeze(-1); centers=torch.linspace(-1,1,self.grid_size,device=x.device)
        basis=F.relu(1-(u-centers).abs()).pow(2)
        return self.linear(x)+torch.einsum('...ig,oig->...o',basis,self.spline)
    def regularization(self): return self.spline.square().mean()

class SpectralGate(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.LayerNorm(5),nn.Linear(5,12),nn.SiLU(),nn.Linear(12,5))
    def forward(self,x): return x*(1+.5*torch.tanh(self.net(x)))

class LatentBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.attn=nn.MultiheadAttention(32,4,batch_first=True); self.n1=nn.LayerNorm(32); self.n2=nn.LayerNorm(32); self.ff=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))
    def forward(self, lat, electrodes):
        a,_=self.attn(lat,electrodes,electrodes,need_weights=False); lat=self.n1(lat+a); return self.n2(lat+self.ff(lat))

class TemporalMixer(nn.Module):
    def __init__(self):
        super().__init__(); self.ds=nn.ModuleList([nn.Conv1d(48,48,k,padding=d*(k-1)//2,dilation=d,groups=48) for k,d in ((3,1),(5,2),(7,4))]); self.pw=nn.Conv1d(144,96,1); self.ds2=nn.Conv1d(48,48,3,padding=1,groups=48); self.pw2=nn.Conv1d(48,96,1); self.n1=nn.LayerNorm(48); self.n2=nn.LayerNorm(48)
    def glu(self, z): a,b=z.chunk(2,1); return a*torch.sigmoid(b)
    def forward(self,x):
        z=x.transpose(1,2); z=self.n1((z+self.glu(self.pw(torch.cat([m(z) for m in self.ds],1)))).transpose(1,2)).transpose(1,2); return self.n2((z+self.glu(self.pw2(self.ds2(z)))).transpose(1,2))

class TorchStatevectorQuantumBranch(nn.Module):
    backend_name='torch_statevector_cuda'
    def __init__(self,layers=4, entangle=True, z_only=False):
        super().__init__(); self.layers=layers; self.entangle=entangle; self.z_only=z_only; self.rot=nn.Parameter(.05*torch.randn(layers,4,3))
    def forward(self, angles):
        # Exact differentiable state-vector circuit, shape [B,4,4,2].
        b=angles.shape[0]; state=torch.zeros(b,16,dtype=torch.complex64,device=angles.device); state[:,0]=1
        for layer in range(self.layers):
            for q in range(4):
                state=self._ry(state,angles[:,layer,q,0],q); state=self._rz(state,angles[:,layer,q,1],q)
                a,beta,c=self.rot[layer,q]; state=self._rz(state,a,q); state=self._ry(state,beta,q); state=self._rz(state,c,q)
            if self.entangle:
                pairs=((0,1),(1,2),(2,3),(3,0)) if layer%2==0 else ((1,0),(2,1),(3,2),(0,3))
                for c,t in pairs: state=self._cnot(state,c,t)
        z=torch.stack([self._z(state,q) for q in range(4)],1)
        if self.z_only:return z
        x=torch.stack([self._x(state,q) for q in range(4)],1); zz=torch.stack([self._z(state,a)*self._z(state,bq) for a,bq in ((0,1),(1,2),(2,3),(3,0))],1)
        return torch.cat([z,x,zz],1)
    def _apply_gate(self,s,g,q):
        out=s.clone(); stride=1<<q
        for start in range(0,16,2*stride):
            a=s[:,start:start+stride]; b=s[:,start+stride:start+2*stride]; out[:,start:start+stride]=g[:,0,None]*a+g[:,1,None]*b; out[:,start+stride:start+2*stride]=g[:,2,None]*a+g[:,3,None]*b
        return out
    def _ry(self,s,a,q):
        if a.ndim == 0: a=a.expand(s.shape[0])
        c=torch.cos(a/2).to(torch.complex64); z=torch.sin(a/2).to(torch.complex64); return self._apply_gate(s,torch.stack([c,-z,z,c],1),q)
    def _rz(self,s,a,q):
        if a.ndim == 0: a=a.expand(s.shape[0])
        e=torch.exp((-0.5j*a).to(torch.complex64)); f=torch.conj(e); return self._apply_gate(s,torch.stack([e,torch.zeros_like(e),torch.zeros_like(e),f],1),q)
    def _cnot(self,s,c,t):
        idx=torch.arange(16,device=s.device); mask=((idx>>c)&1).bool(); dst=idx ^ (mask.to(torch.long)<<t); return s[:,dst]
    def _z(self,s,q):
        idx=torch.arange(16,device=s.device); sign=(1-2*((idx>>q)&1)).float(); return (s.abs().square()*sign).sum(1)
    def _x(self,s,q):
        idx=torch.arange(16,device=s.device); return (s.conj()*s[:,idx^(1<<q)]).real.sum(1)

class LatentSetQuanKAN(nn.Module):
    def __init__(self, classes, num_latents=6, quantum_layers=4, no_quantum=False,
                 no_entropy_gate=False, detach_quantum=True, entangle=True,
                 z_only=False, fixed_quantum_scale=False):
        super().__init__(); self.spectral=SpectralGate(); self.de=nn.Linear(5,32); self.eid=nn.Embedding(62,32); self.pos=nn.Sequential(nn.Linear(3,12),nn.SiLU(),nn.Linear(12,32)); self.token_norm=nn.LayerNorm(32); self.latents=nn.Parameter(torch.randn(1,num_latents,32)*.02); self.blocks=nn.ModuleList([LatentBlock(),LatentBlock()]); self.compress=nn.Sequential(nn.Linear(num_latents*32,48),nn.LayerNorm(48),nn.SiLU()); self.temporal=TemporalMixer(); self.pool=nn.Linear(48,1); self.clean=TinyKANLinear(48,16); self.classical=TinyKANLinear(16,classes); self.quantum_layers=quantum_layers; self.angle=TinyKANLinear(16,quantum_layers*8); self.quantum=TorchStatevectorQuantumBranch(quantum_layers,entangle=entangle,z_only=z_only); self.qread=TinyKANLinear(4 if z_only else 12,16); self.qaux=nn.Linear(16,classes); nn.init.zeros_(self.qaux.weight); nn.init.zeros_(self.qaux.bias); self.alpha=nn.Parameter(torch.zeros(()),requires_grad=not fixed_quantum_scale); self.final=TinyKANLinear(16,classes); self.no_quantum=no_quantum; self.no_entropy_gate=no_entropy_gate; self.detach_quantum=detach_quantum
    def forward(self,x,channels=None,positions=None):
        b,t,n,_=x.shape; channels=torch.arange(n,device=x.device) if channels is None else channels.to(x.device)
        if positions is None: positions=torch.zeros(n,3,device=x.device)
        e=self.de(self.spectral(x))+self.eid(channels)[None,None]+self.pos(positions)[None,None]; e=self.token_norm(e).reshape(b*t,n,32); l=self.latents.expand(b*t,-1,-1)
        for block in self.blocks:l=block(l,e)
        h=self.compress(l.flatten(1)).reshape(b,t,48); h=self.temporal(h); w=torch.softmax(self.pool(h).squeeze(-1),1); pooled=(h*w.unsqueeze(-1)).sum(1); clean=self.clean(pooled); classical=self.classical(clean); entropy=-(classical.softmax(-1)*(classical.log_softmax(-1))).sum(-1)/math.log(classical.shape[-1]); gate=torch.ones_like(entropy) if self.no_entropy_gate else entropy.pow(1.5)
        q_input=clean.detach() if self.detach_quantum else clean
        angles=(math.pi*torch.tanh(self.angle(q_input))).reshape(b,self.quantum_layers,4,2); q=self.qread(self.quantum(angles)); alpha=.25*torch.sigmoid(self.alpha)
        hybrid=F.layer_norm(clean+(0 if self.no_quantum else gate[:,None]*alpha*q),(16,)); final=self.final(hybrid); qcr=(gate[:,None]*alpha*q).norm(dim=1)/(clean.norm(dim=1)+1e-8)
        return {'final_logits':final,'classical_logits':classical,'pooled':pooled,'clean':clean,'quantum':q,'qaux':self.qaux(q),'gate':gate,'alpha':alpha,'qcr':qcr}
    def kan_regularization(self): return sum(m.regularization() for m in self.modules() if isinstance(m,TinyKANLinear))

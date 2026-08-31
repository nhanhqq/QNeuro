"""Compact mask-aware QNeuro-v2 model."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class TinyKANLinear(nn.Module):
    def __init__(self, inp, out, grid_size=3):
        super().__init__(); self.linear = nn.Linear(inp, out); self.spline = nn.Parameter(torch.empty(out, inp, grid_size)); nn.init.normal_(self.spline, std=.02)
    def forward(self, x):
        u = torch.tanh(x).unsqueeze(-1); centers = torch.linspace(-1, 1, self.spline.shape[-1], device=x.device)
        basis = F.relu(1 - (u - centers).abs()).pow(2)
        return self.linear(x) + torch.einsum('...ig,oig->...o', basis, self.spline)
    def regularization(self): return self.spline.square().mean()


class SpectralGate(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(5), nn.Linear(5, 12), nn.SiLU(), nn.Linear(12, 5))
    def forward(self, x): return x * (1 + .5 * torch.tanh(self.net(x)))


class GeometryGraphRefiner(nn.Module):
    def __init__(self, dim=32, top_k=8):
        super().__init__(); self.q = nn.Linear(dim, 16, bias=False); self.k = nn.Linear(dim, 16, bias=False); self.v = nn.Linear(dim, dim, bias=False); self.norm = nn.LayerNorm(dim); self.log_sigma = nn.Parameter(torch.tensor(0.0)); self.top_k = top_k
    def forward(self, e, positions):
        if positions is None: raise ValueError('real electrode positions are required')
        q, k = self.q(e), self.k(e); content = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        d2 = (positions[:, None, :] - positions[None, :, :]).square().sum(-1)
        bias = -d2 / self.log_sigma.exp().square().clamp_min(1e-4)
        scores = content + .15 * bias[None]
        if self.top_k and self.top_k < scores.shape[-1]:
            vals, idx = scores.topk(self.top_k, dim=-1); sparse = torch.full_like(scores, -torch.inf); scores = sparse.scatter(-1, idx, vals)
        a = torch.softmax(scores, -1); return self.norm(e + a @ self.v(e))


class LatentBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.attn = nn.MultiheadAttention(32, 4, batch_first=True); self.n1 = nn.LayerNorm(32); self.n2 = nn.LayerNorm(32); self.ff = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Dropout(.12), nn.Linear(64, 32))
    def forward(self, lat, electrodes):
        a, _ = self.attn(lat, electrodes, electrodes, need_weights=False); lat = self.n1(lat + a); return self.n2(lat + self.ff(lat))


def apply_mask(x, mask): return x * mask.unsqueeze(-1).to(x.dtype)


class TemporalMixer(nn.Module):
    def __init__(self):
        super().__init__(); self.ds = nn.ModuleList([nn.Conv1d(48, 48, k, padding=d*(k-1)//2, dilation=d, groups=48) for k, d in ((3, 1), (5, 2), (7, 4))]); self.pw = nn.Conv1d(144, 96, 1); self.ds2 = nn.Conv1d(48, 48, 3, padding=1, groups=48); self.pw2 = nn.Conv1d(48, 96, 1); self.n1 = nn.LayerNorm(48); self.n2 = nn.LayerNorm(48)
    def glu(self, z): a, b = z.chunk(2, 1); return a * torch.sigmoid(b)
    def forward(self, x, mask):
        x = apply_mask(x, mask); z = x.transpose(1, 2); branches = [apply_mask(m(z).transpose(1, 2), mask).transpose(1, 2) for m in self.ds]
        z = self.n1((z + self.glu(self.pw(torch.cat(branches, 1)))).transpose(1, 2)).transpose(1, 2); z = apply_mask(z.transpose(1, 2), mask).transpose(1, 2)
        z = self.n2((z + self.glu(self.pw2(self.ds2(z)))).transpose(1, 2)).transpose(1, 2); return apply_mask(z.transpose(1, 2), mask)


class TorchStatevectorQuantumBranch(nn.Module):
    def __init__(self, layers=4, entangle=True):
        super().__init__(); self.layers = layers; self.entangle = entangle; self.rot = nn.Parameter(.05 * torch.randn(layers, 4, 3))
    def forward(self, angles):
        state = torch.zeros(angles.shape[0], 16, dtype=torch.complex64, device=angles.device); state[:, 0] = 1
        for layer in range(self.layers):
            for q in range(4):
                state = self._ry(state, angles[:, layer, q, 0], q); state = self._rz(state, angles[:, layer, q, 1], q)
                a, b, c = self.rot[layer, q]; state = self._rz(state, a, q); state = self._ry(state, b, q); state = self._rz(state, c, q)
            if self.entangle:
                for c, t in (((0, 1), (1, 2), (2, 3), (3, 0)) if layer % 2 == 0 else ((1, 0), (2, 1), (3, 2), (0, 3))): state = self._cnot(state, c, t)
        z = torch.stack([self._z(state, q) for q in range(4)], 1); x = torch.stack([self._x(state, q) for q in range(4)], 1); zz = torch.stack([self._z(state, a) * self._z(state, b) for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))], 1); return torch.cat([z, x, zz], 1)
    def _apply_gate(self, s, g, q):
        out = s.clone(); stride = 1 << q
        for start in range(0, 16, 2 * stride):
            a, b = s[:, start:start+stride], s[:, start+stride:start+2*stride]; out[:, start:start+stride] = g[:, 0, None] * a + g[:, 1, None] * b; out[:, start+stride:start+2*stride] = g[:, 2, None] * a + g[:, 3, None] * b
        return out
    def _ry(self, s, a, q):
        if a.ndim == 0: a = a.expand(s.shape[0])
        c, z = torch.cos(a / 2).to(torch.complex64), torch.sin(a / 2).to(torch.complex64); return self._apply_gate(s, torch.stack([c, -z, z, c], 1), q)
    def _rz(self, s, a, q):
        if a.ndim == 0: a = a.expand(s.shape[0])
        e = torch.exp((-0.5j * a).to(torch.complex64)); return self._apply_gate(s, torch.stack([e, torch.zeros_like(e), torch.zeros_like(e), torch.conj(e)], 1), q)
    def _cnot(self, s, c, t):
        idx = torch.arange(16, device=s.device); mask = ((idx >> c) & 1).bool(); return s[:, idx ^ (mask.long() << t)]
    def _z(self, s, q):
        idx = torch.arange(16, device=s.device); return (s.abs().square() * (1 - 2 * ((idx >> q) & 1)).float()).sum(1)
    def _x(self, s, q):
        idx = torch.arange(16, device=s.device); return (s.conj() * s[:, idx ^ (1 << q)]).real.sum(1)


class LatentSetQuanKAN(nn.Module):
    def __init__(self, classes, num_latents=6, quantum_layers=4, no_quantum=False, no_entropy_gate=False, beta=0.0, gate_tau=.65, gate_temperature=.15):
        super().__init__(); self.use_positions = True; self.no_quantum = no_quantum; self.no_entropy_gate = no_entropy_gate; self.beta = beta; self.gate_tau = gate_tau; self.gate_temperature = gate_temperature
        self.spectral = SpectralGate(); self.de = nn.Linear(5, 32); self.eid = nn.Embedding(62, 32); self.pos = nn.Sequential(nn.Linear(3, 12), nn.SiLU(), nn.Linear(12, 32)); self.token_norm = nn.LayerNorm(32); self.latents = nn.Parameter(torch.randn(1, num_latents, 32) * .02); self.graph = GeometryGraphRefiner(); self.blocks = nn.ModuleList([LatentBlock(), LatentBlock()]); self.compress = nn.Sequential(nn.Linear(num_latents * 32, 48), nn.LayerNorm(48), nn.SiLU()); self.temporal = TemporalMixer(); self.bigru = nn.GRU(48, 24, batch_first=True, bidirectional=True); self.dropout = nn.Dropout(.15); self.pool = nn.Linear(48, 1); self.clean = TinyKANLinear(48, 16); self.classical = TinyKANLinear(16, classes); self.angle = TinyKANLinear(16, quantum_layers * 8); self.quantum = TorchStatevectorQuantumBranch(quantum_layers); self.qread = TinyKANLinear(12, 16); self.qaux = TinyKANLinear(16, classes); self.alpha_raw = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x, lengths, mask, channels=None, positions=None):
        b, t, n, _ = x.shape; channels = torch.arange(n, device=x.device) if channels is None else channels.to(x.device)
        if positions is None and self.use_positions: raise ValueError('positions must be supplied explicitly')
        if positions is None: positions = torch.zeros(n, 3, device=x.device)
        positions = positions.to(x.device); mask = mask.to(x.device).bool(); e = self.de(self.spectral(x)) + self.eid(channels)[None, None] + self.pos(positions)[None, None]; e = self.token_norm(e).reshape(b*t, n, 32); e = self.graph(e, positions)
        l = self.latents.expand(b*t, -1, -1)
        for block in self.blocks: l = block(l, e)
        h = self.compress(l.flatten(1)).reshape(b, t, 48); h = self.temporal(h, mask); packed = nn.utils.rnn.pack_padded_sequence(h, lengths.detach().cpu().clamp_min(1), batch_first=True, enforce_sorted=False); packed, _ = self.bigru(packed); h, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=t); h = apply_mask(self.dropout(h), mask)
        scores = self.pool(h).squeeze(-1).masked_fill(~mask, -torch.inf); weights = torch.softmax(scores, 1); pooled = (h * weights.unsqueeze(-1)).sum(1); clean = self.clean(pooled); classical = self.classical(clean); entropy = -(classical.softmax(-1) * classical.log_softmax(-1)).sum(-1) / math.log(classical.shape[-1]); gate = torch.ones_like(entropy) if self.no_entropy_gate else torch.sigmoid((entropy - self.gate_tau) / self.gate_temperature)
        q_input = clean.detach() + self.beta * (clean - clean.detach()); angles = (math.pi * torch.tanh(self.angle(q_input))).reshape(b, 4, 4, 2); q_delta = self.qaux(self.qread(self.quantum(angles))); alpha = .5 * torch.sigmoid(self.alpha_raw); final = classical if self.no_quantum else classical + gate[:, None] * alpha * q_delta
        correction = gate[:, None] * alpha * q_delta
        return {'final_logits': final, 'classical_logits': classical, 'qaux': q_delta, 'pooled': pooled, 'clean': clean, 'quantum': q_delta, 'entropy': entropy, 'gate': gate, 'alpha': alpha, 'qcr': correction.norm(dim=1) / (clean.norm(dim=1) + 1e-8), 'attention_weights': weights}

    def kan_regularization(self): return sum(m.regularization() for m in self.modules() if isinstance(m, TinyKANLinear))

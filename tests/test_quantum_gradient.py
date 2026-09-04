import pytest,torch
from src.models.quantum_latent import QuantumLatent16
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_quantum_gradient():
 x=torch.randn(8,4,device='cuda',requires_grad=True);m=QuantumLatent16(rz_noise_std=.1).cuda();q=m(x);q.sum().backward();assert q.shape==(8,4);assert torch.isfinite(x.grad).all();assert torch.isfinite(m.weights.grad).all();assert torch.isfinite(m.encoding_scale.grad).all()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('sigma',[0.,.05,.10,.25,.50,1.])
def test_noise_grid_and_deterministic_evaluation(sigma):
 x=torch.randn(8,4,device='cuda');m=QuantumLatent16(rz_noise_std=sigma).cuda();m.eval();a=m(x);b=m(x);assert torch.equal(a,b);assert torch.isfinite(a).all()

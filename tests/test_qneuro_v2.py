import unittest
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from qneuro.data import CHANNEL_POSITIONS, source_normalize, tensor_mask
from qneuro.model import LatentSetQuanKAN
from qneuro.train import optimizer


class QNeuroV2Tests(unittest.TestCase):
    def make(self, classes=3, t=7, n=3):
        torch.manual_seed(4)
        x=torch.randn(2,t,n,5); lengths=torch.tensor([3, t]); mask=tensor_mask(lengths,t); p=torch.tensor(CHANNEL_POSITIONS[:n]); return x,lengths,mask,p

    def test_optimizer_coverage(self):
        m=LatentSetQuanKAN(3); _, report=optimizer(m); self.assertEqual(report['missing_parameter_count'],0); self.assertEqual(report['trainable_parameters'],report['optimizer_parameters'])

    def test_padding_invariance(self):
        m=LatentSetQuanKAN(3).eval(); x,l,mask,p=self.make(t=7,n=3); short=x[:1,:3].clone(); long=torch.cat([short,torch.zeros(1,4,3,5)],1); ls=torch.tensor([3]); ms=tensor_mask(ls,3); ml=tensor_mask(ls,7)
        with torch.no_grad(): a=m(short,ls,ms,torch.arange(3),p)['final_logits']; b=m(long,ls,ml,torch.arange(3),p)['final_logits']
        self.assertLess(float((a-b).abs().max()),1e-5)

    def test_padding_attention_is_zero(self):
        m=LatentSetQuanKAN(3).eval(); x,l,mask,p=self.make(); out=m(x,l,mask,torch.arange(3),p); self.assertTrue(torch.allclose(out['attention_weights'][0,3:],torch.zeros(4)))

    def test_positions_affect_output(self):
        m=LatentSetQuanKAN(3).eval(); x,l,mask,p=self.make(); q=p.flip(0)
        with torch.no_grad(): a=m(x,l,mask,torch.arange(3),p)['final_logits']; b=m(x,l,mask,torch.arange(3),q)['final_logits']
        self.assertGreater(float((a-b).abs().max()),1e-8)

    def test_quantum_residual_gradient(self):
        m=LatentSetQuanKAN(3); x,l,mask,p=self.make(); out=m(x,l,mask,torch.arange(3),p); loss=out['qaux'].square().mean(); loss.backward(); self.assertGreater(float(m.qaux.linear.weight.grad.abs().sum()),0); self.assertGreater(float(m.quantum.rot.grad.abs().sum()),0)

    def test_source_target_normalization(self):
        x=np.ones((3,4,2,5),dtype=np.float32); x[2]+=1000; z,stats=source_normalize(x,np.array([0,1]),np.array([4,4,4])); self.assertLess(abs(stats[0]['mean']-1),1e-6); self.assertGreater(float(z[2].mean()),100)

    def test_group_split_disjoint(self):
        groups=np.repeat(np.arange(5),4); idx=np.arange(len(groups)); tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=3).split(idx,groups=groups)); self.assertTrue(set(groups[tr]).isdisjoint(set(groups[va])))

    def test_small_channel_forward_backward(self):
        for n in (1,3):
            m=LatentSetQuanKAN(3); x,l,mask,p=self.make(t=4,n=n); out=m(x,l,mask,torch.arange(n),p); out['final_logits'].sum().backward(); self.assertEqual(tuple(out['final_logits'].shape),(2,3))


if __name__=='__main__': unittest.main()

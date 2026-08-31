"""Strict-LOSO ablation entrypoint: source validation selects; target is evaluated once."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from qneuro.data import DATASETS, EEG, load, source_normalize
from qneuro.model import LatentSetQuanKAN
from qneuro.train import seed_all, optimizer, loss_base, metrics, atomic_json
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--target',required=True);p.add_argument('--variant',default='full_hybrid');p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=25);p.add_argument('--output-dir',default='results/ablations');p.add_argument('--seed',type=int,default=3);a=p.parse_args(); seed_all(a.seed)
x,y,persons,c=load(a.dataset); src=np.flatnonzero(persons!=a.target); tgt=np.flatnonzero(persons==a.target); x,stats=source_normalize(x,src)
tr,va=train_test_split(src,test_size=.2,random_state=a.seed,stratify=y[src]); device=torch.device('cuda'); train=DataLoader(EEG(x[tr],y[tr]),batch_size=a.batch_size,shuffle=True,num_workers=0); val=DataLoader(EEG(x[va],y[va]),batch_size=a.batch_size,num_workers=0); test=DataLoader(EEG(x[tgt],y[tgt]),batch_size=a.batch_size,num_workers=0)
noq=a.variant=='no_quantum'; model=LatentSetQuanKAN(c,no_quantum=noq,no_entropy_gate=a.variant=='no_entropy_gate').to(device); opt,sch=optimizer(model,a.epochs); best=-1.; out=Path(a.output_dir)/a.dataset/a.variant/f'target_{a.target}';out.mkdir(parents=True,exist_ok=True)
for e in range(1,a.epochs+1):
  model.train()
  for xb,yb in train:
    opt.zero_grad(set_to_none=True);loss,_=loss_base(model(xb.to(device)),yb.to(device),model);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  sch.step(); vm=metrics(model,val,device,torch.arange(62,device=device),c)
  if vm['accuracy']>best:best=vm['accuracy'];torch.save({'model':model.state_dict(),'epoch':e,'source_validation':vm,'protocol':'strict LOSO; source-only validation'},out/'best_source_val.pt')
state=torch.load(out/'best_source_val.pt',map_location=device);model.load_state_dict(state['model']); result=metrics(model,test,device,torch.arange(62,device=device),c); result.update({'target':a.target,'variant':a.variant,'selection':'source-only validation','target_test_evaluations':1});atomic_json(out/'result.json',result);print(json.dumps(result,indent=2))

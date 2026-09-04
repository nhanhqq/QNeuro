#!/usr/bin/env python3
import argparse,csv,json,os,platform,subprocess,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np,pandas as pd,torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from src.data.audit_chsz import audit
from src.data.split import make_splits
from src.features.cache import build_cache
from src.models.model import TinyBiLSTM,count_trainable_parameters
from src.evaluation.metrics import metrics
from src.utils.seed import seed_everything
def grad_norm(model,key):
 ps=[p.grad.detach().norm() for n,p in model.named_parameters() if key in n and p.grad is not None];return float(torch.stack(ps).norm()) if ps else 0.
def evaluate(model,loader,loss,device):
 model.eval();ys=[];pp=[];probs=[];ls=[]
 with torch.no_grad():
  for x,y in loader:
   z=model(x.to(device));ls.append(loss(z,y.to(device)).item()*len(y));ys.extend(y.numpy());pp.extend(z.argmax(1).cpu().numpy());probs.extend(z.softmax(1).cpu().numpy())
 r=metrics(np.array(ys),np.array(pp),np.array(probs));r['loss']=sum(ls)/len(ys);return r
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='CHSZ');p.add_argument('--output',default='results/chsz');p.add_argument('--fold',type=int,required=True);p.add_argument('--seed',type=int,default=7);p.add_argument('--variant',choices=['quantum','classical','identity','no_bilstm'],default='quantum');p.add_argument('--epochs',type=int,default=300);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--fs',type=float,default=500);p.add_argument('--frame-seconds',type=float,default=1.0);p.add_argument('--hop-seconds',type=float,default=.5);p.add_argument('--pca-components',type=int,default=8);p.add_argument('--label-smoothing',type=float,default=.03);p.add_argument('--kan-smoothness',type=float,default=1e-4);p.add_argument('--rz-noise-std',type=float,default=.10);p.add_argument('--skip-audit',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('GPU required by this experiment')
 seed_everything(a.seed);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if not a.skip_audit: audit(a.data,out/'audit')
 raw_y=np.load(Path(a.data)/'labels.npy');classes=np.unique(raw_y);y=np.searchsorted(classes,raw_y).astype(int);meta=pd.read_csv(Path(a.data)/'meta.csv');groups=meta['subject'].to_numpy();(out/'label_mapping.json').write_text(json.dumps({'encoded_label_mapping':{str(i):str(v) for i,v in enumerate(classes.tolist())},'semantic_mapping':'labels encoded from package values; no semantic ordering inferred'},indent=2))
 cache=Path(a.data)/'cache'; cp=cache/'chsz_spectral.npy'
 config_path=cache/'chsz_spectral_config.json'
 cache_fs=None
 if config_path.exists():
  try:
   cache_cfg=json.loads(config_path.read_text());cache_fs=(cache_cfg.get('sampling_rate'),cache_cfg.get('frame_seconds',1.),cache_cfg.get('hop_seconds',.5))
  except (OSError,json.JSONDecodeError): pass
 if not cp.exists() or cache_fs != (a.fs,a.frame_seconds,a.hop_seconds): build_cache(Path(a.data)/'X.npy',cache,a.fs,frame_seconds=a.frame_seconds,hop_seconds=a.hop_seconds)
 f=np.load(cp,mmap_mode='r');splits=make_splits(y,groups,out/'splits');sp=splits[a.fold]
 if not 1 <= a.pca_components <= f.shape[-1]: raise ValueError(f'pca-components must be in [1,{f.shape[-1]}]')
 idx={k:np.asarray(sp[k+'_indices']) for k in ('train','test')};art=out/'artifacts'/f'fold_{a.fold}'/f'seed_{a.seed}';art.mkdir(parents=True,exist_ok=True)
 scaler=StandardScaler().fit(f[idx['train']].reshape(-1,f.shape[-1]));pca=PCA(a.pca_components,whiten=True,random_state=a.seed).fit(scaler.transform(f[idx['train']].reshape(-1,f.shape[-1])))
 joblib.dump(scaler,art/'scaler.joblib');joblib.dump(pca,art/'pca.joblib');(art/'pca.json').write_text(json.dumps({'whiten':True,'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'sum':float(pca.explained_variance_ratio_.sum())},indent=2))
 def reduced(ii):return pca.transform(scaler.transform(f[ii].reshape(-1,f.shape[-1]))).reshape(len(ii),f.shape[1],a.pca_components).astype('float32')
 z={k:reduced(v) for k,v in idx.items()}
 counts=np.bincount(y[idx['train']],minlength=len(classes));
 if np.any(counts==0): raise RuntimeError(f'training fold misses classes: counts={counts.tolist()}')
 w=len(idx['train'])/(len(classes)*counts);imbalance_ratio=float(counts.max()/counts.min())
 # Sampling is derived solely from source-subject labels.  On heavily
 # imbalanced sources this makes every optimizer step see minority examples;
 # use unweighted CE in that case to avoid double class reweighting.
 use_balanced_sampler=imbalance_ratio>=3.0
 train_set=TensorDataset(torch.from_numpy(z['train']),torch.from_numpy(y[idx['train']]))
 if use_balanced_sampler:
  sample_weight=torch.as_tensor(1.0/counts[y[idx['train']]],dtype=torch.double)
  train_loader=DataLoader(train_set,batch_size=a.batch_size,sampler=WeightedRandomSampler(sample_weight,len(sample_weight),replacement=True),num_workers=0,pin_memory=True)
 else:
  train_loader=DataLoader(train_set,batch_size=a.batch_size,shuffle=True,num_workers=0,pin_memory=True)
 loaders={'train':train_loader,'test':DataLoader(TensorDataset(torch.from_numpy(z['test']),torch.from_numpy(y[idx['test']])),batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=True)}
 no=a.variant=='no_bilstm';latent='identity' if a.variant in ('identity','no_bilstm') else a.variant;model=TinyBiLSTM(len(classes),latent=latent,no_bilstm=no,rz_noise_std=a.rz_noise_std,input_size=a.pca_components,hidden_size=6).cuda();n=count_trainable_parameters(model)
 if a.variant=='quantum' and n>=1000:raise AssertionError(f'parameter constraint violated: {n}')
 loss_weight=None if use_balanced_sampler else torch.tensor(w,dtype=torch.float32,device='cuda');loss=torch.nn.CrossEntropyLoss(weight=loss_weight,label_smoothing=a.label_smoothing);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=1e-5)
 root=out/'runs'/f'{a.variant}_fold{a.fold}_seed{a.seed}';root.mkdir(parents=True,exist_ok=True);ck=root/'best.pt';last=root/'last.pt';start=0;best=-1
 if last.exists() or ck.exists():
  q=torch.load(last if last.exists() else ck,map_location='cuda');model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);
  if 'lr_scheduler' in q: lr_scheduler.load_state_dict(q['lr_scheduler'])
  # ``last.pt`` records the most recent epoch, while ``best.pt`` records the
  # highest target-selected epoch.  Persist the latter explicitly so an
  # interrupted job cannot forget an earlier maximum when it resumes.
  start=q['epoch']+1;best=q.get('best_target_selected_test_accuracy',q.get('target_selected_test_accuracy',-1))
 fields=['epoch','train_loss','train_accuracy','test_loss','test_accuracy','test_balanced_accuracy','test_macro_f1','test_weighted_f1','learning_rate','grad_bilstm','grad_pre_quantum','grad_quantum','grad_classifier','epoch_seconds']
 log=root/'epochs.csv';write_header=not log.exists()
 with log.open('a',newline='') as fh:
  wr=csv.DictWriter(fh,fieldnames=fields);
  if write_header:wr.writeheader()
  for ep in range(start,1 if a.dry_run else a.epochs):
   t=time.time();model.train();total=0.;ny=0;yp=[];yy=[];g={}
   for x,b in loaders['train']:
    opt.zero_grad();o=model(x.cuda(non_blocking=True));l=loss(o,b.cuda(non_blocking=True));reg=a.kan_smoothness*model.classifier.smoothness_penalty();(l+reg).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);g={'grad_bilstm':grad_norm(model,'bilstm'),'grad_pre_quantum':grad_norm(model,'to_quantum'),'grad_quantum':grad_norm(model,'latent.weights'),'grad_classifier':grad_norm(model,'classifier')};opt.step();total+=l.item()*len(b);ny+=len(b);yp.extend(o.argmax(1).detach().cpu().numpy());yy.extend(b.numpy())
   tr=metrics(np.array(yy),np.array(yp));tr['loss']=total/ny
   te=evaluate(model,loaders['test'],loss,'cuda')
   row={'epoch':ep,'train_loss':tr['loss'],'train_accuracy':tr['accuracy'],'test_loss':te['loss'],'test_accuracy':te['accuracy'],'test_balanced_accuracy':te['balanced_accuracy'],'test_macro_f1':te['macro_f1'],'test_weighted_f1':te['weighted_f1'],'learning_rate':opt.param_groups[0]['lr'],**g,'epoch_seconds':time.time()-t};wr.writerow(row);fh.flush()
   lr_scheduler.step()
   improved=te['accuracy']>best
   if improved: best=te['accuracy']
   # Keep the current test metrics for diagnostics, but retain the all-time
   # selection metric separately for exact resume semantics.
   state={'model':model.state_dict(),'optimizer':opt.state_dict(),'lr_scheduler':lr_scheduler.state_dict(),'epoch':ep,'target_selected_test_accuracy':te['accuracy'],'target_selected_test_balanced_accuracy':te['balanced_accuracy'],'best_target_selected_test_accuracy':best,'test_metrics':te,'config':vars(a),'label_mapping':classes.tolist()}
   torch.save(state,last)
   if improved:
    torch.save(state,ck);(root/'target_selected_test.json').write_text(json.dumps(te,indent=2))
 (root/'run.json').write_text(json.dumps({'params':n,'parameter_budget':'<1000 trainable parameters','capacity_variant':'PCA8 -> BiLSTM hidden6 -> LayerNorm -> Linear12->4','pca_components':a.pca_components,'bilstm_input_size':a.pca_components,'bilstm_hidden_size':6,'quantum_variational_parameters':16 if a.variant=='quantum' else 0,'quantum_encoding_scale_parameters':8 if a.variant=='quantum' else 0,'source_class_counts':counts.tolist(),'source_class_weights':w.tolist(),'source_imbalance_ratio':imbalance_ratio,'class_balance':'WeightedRandomSampler with unweighted CE' if use_balanced_sampler else 'shuffle with weighted CE','classifier':'KANLinear spline-only: grid_size=2, order=1, bounded quantum-latent input','label_smoothing':a.label_smoothing,'kan_smoothness':a.kan_smoothness,'quantum_augmentation':f'training-only RZ Gaussian angles N(0, {a.rz_noise_std}^2) radians per sample, qubit, and quantum block; deterministic evaluation','quantum_data_reuploading':'trainable alpha[layer,qubit] * angle[qubit], initialized to 1; 8 parameters for depth=2','event_evaluation':'unavailable: meta.csv contains no event/event_id/trial/seizure_id','protocol':'LOSO: train all N-1 source subjects; evaluate exactly one held-out test subject after every epoch; no validation split','target_selection':'best.pt is selected by maximum target test accuracy per user instruction; target-selected development evidence, not unbiased final-test evidence','device':torch.cuda.get_device_name(),'torch':torch.__version__,'python':platform.python_version(),'num_workers':0},indent=2))
if __name__=='__main__':main()

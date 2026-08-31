"""Target-selected LOSO trainer. Target samples are evaluation-only by construction."""
import argparse, csv, json, os, random, time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
from .data import DATASETS, EEG, load, source_normalize
from .model import LatentSetQuanKAN

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def configure_cuda():
    """Enable fast RTX kernels while keeping DataLoader workers disabled."""
    if not torch.cuda.is_available(): return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'): torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'): torch.backends.cudnn.allow_tf32 = True
def rng_state(): return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def set_rng(s): random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch']); torch.cuda.set_rng_state_all(s['cuda'])
def atomic_json(path,obj):
    tmp=str(path)+'.tmp'; Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(tmp,'w') as f: json.dump(obj,f,indent=2)
    os.replace(tmp,path)
def metrics(model, loader, device, channels, classes):
    model.eval(); ys=[]; pred=[]; probs=[]; loss=[]
    with torch.no_grad():
      for x,y in loader:
        o=model(x.to(device,non_blocking=True)[:,:,channels],channels); p=o['final_logits']; ys.extend(y.tolist()); pred.extend(p.argmax(1).cpu().tolist()); probs.extend(p.softmax(1).cpu().tolist()); loss.append(F.cross_entropy(p,y.to(device,non_blocking=True)).item())
    pr,re,f1,_=precision_recall_fscore_support(ys,pred,labels=list(range(classes)),average='macro',zero_division=0)
    return {'cross_entropy':float(np.mean(loss)),'accuracy':accuracy_score(ys,pred),'macro_precision':pr,'macro_recall':re,'macro_f1':f1,'weighted_f1':precision_recall_fscore_support(ys,pred,average='weighted',zero_division=0)[2],'balanced_accuracy':balanced_accuracy_score(ys,pred),'confusion_matrix':confusion_matrix(ys,pred,labels=list(range(classes))).tolist()}
def optimizer(model,epochs):
    groups=[{'params':model.spectral.parameters(),'lr':3e-4},{'params':list(model.de.parameters())+list(model.blocks.parameters())+list(model.temporal.parameters()),'lr':3e-4},{'params':list(model.clean.parameters())+list(model.classical.parameters())+list(model.angle.parameters())+list(model.qread.parameters())+list(model.final.parameters())+list(model.quantum.parameters())+[model.alpha],'lr':5e-4}]
    o=torch.optim.AdamW(groups,weight_decay=1e-4); return o,torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs,eta_min=1e-6)
def loss_base(out,y,model):
    residual=F.one_hot(y,out['final_logits'].shape[-1]).float()-out['classical_logits'].softmax(-1).detach()
    q=F.smooth_l1_loss(torch.tanh(out['qaux']),residual); fin=F.cross_entropy(out['final_logits'],y,label_smoothing=.05); cls=F.cross_entropy(out['classical_logits'],y,label_smoothing=.05); return fin+.25*cls+.15*q+1e-5*model.kan_regularization(),{'final_ce':fin,'classical_ce':cls,'q':q}
def kd(student, global_teacher, adjacent_teacher, x, y, channels, adjacent_channels, k):
    s=student(x[:,:,channels],channels); g=global_teacher(x,torch.arange(62,device=x.device)); a=adjacent_teacher(x[:,:,adjacent_channels],adjacent_channels)
    T=4.; kg=F.kl_div(F.log_softmax(s['final_logits']/T,1),F.softmax(g['final_logits']/T,1),reduction='batchmean')*T*T; ka=F.kl_div(F.log_softmax(s['final_logits']/T,1),F.softmax(a['final_logits']/T,1),reduction='batchmean')*T*T
    js=.5*(F.kl_div(g['final_logits'].log_softmax(1),a['final_logits'].softmax(1),reduction='batchmean')+F.kl_div(a['final_logits'].log_softmax(1),g['final_logits'].softmax(1),reduction='batchmean'))
    r=max(.15,min(.85,(k/62.)**.5)); ug=r*torch.exp(-js.detach()/.5); wg=ug/(ug+1-r+1e-8); wa=1-wg; fg=1-F.cosine_similarity(s['pooled'],g['pooled']).mean(); fa=1-F.cosine_similarity(s['pooled'],a['pooled']).mean(); base,parts=loss_base(s,y,student); return base+wg*kg+wa*ka+.25*(wg*fg+wa*fa),{'wg':wg,'wa':wa,'js':js,'kg':kg,'ka':ka,'fg':fg,'fa':fa},s
def save_checkpoint(path, **state): torch.save(state,path)

def run_target(args):
    seed_all(args.seed); configure_cuda(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x,y,persons,classes=load(args.dataset,args.data_root); target=args.target
    src=np.flatnonzero(persons!=target); tgt=np.flatnonzero(persons==target)
    if not len(src) or not len(tgt): raise ValueError(f'invalid target {target}; choices={sorted(set(persons))}')
    x,normalization=source_normalize(x,src); train=DataLoader(EEG(x[src],y[src]),batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True); test=DataLoader(EEG(x[tgt],y[tgt]),batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True)
    root=Path(args.output_dir)/args.dataset/f'target_{target}'; base=root/'base'; prog=root/'progressive'; metric_dir=root/'metrics'; [p.mkdir(parents=True,exist_ok=True) for p in (base,prog,metric_dir,root/'runtime')]
    atomic_json(root/'config.json',{'args':vars(args),'dataset':args.dataset,'target':target,'normalization_source_only':normalization,'protocol':'target-selected LOSO; fixed final epoch; target evaluation only'})
    events=root/'runtime'/'events.jsonl'
    with open(events,'a') as f: f.write(json.dumps({'event':'start','timestamp':time.time(),'dataset':args.dataset,'target':target,'device':str(device),'batch_size':args.batch_size,'num_workers':0})+'\n')
    model=LatentSetQuanKAN(classes,quantum_layers=args.quantum_layers).to(device); opt,sch=optimizer(model,args.epochs_base); start=1; latest=base/'resume.pt'; best_path=base/'DEV_ONLY_best_target.pt'; best_acc=-1.; fields=['epoch','lr_backbone','train_total_loss','train_final_ce','train_classical_ce','train_quantum_residual_loss','train_acc','test_acc','test_macro_f1','quantum_gate_mean','quantum_alpha','quantum_correction_ratio','peak_vram_mb','epoch_seconds']
    if args.resume and latest.exists():
      s=torch.load(latest,map_location=device); model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sch.load_state_dict(s['scheduler']);set_rng(s['rng']);start=s['epoch']+1;best_acc=s.get('best_target_accuracy',-1.)
    elif not (base/'base_training.csv').exists():
      with open(base/'base_training.csv','w',newline='') as f: csv.DictWriter(f,fieldnames=fields).writeheader()
    for epoch in range(start,args.epochs_base+1):
      model.train(); losses=[]; correct=n=0; gates=[]; qcr=[]; parts={'final_ce':0.,'classical_ce':0.,'q':0.}; torch.cuda.reset_peak_memory_stats(); started=time.time()
      for xb,yb in train:
        xb,yb=xb.to(device,non_blocking=True),yb.to(device,non_blocking=True); opt.zero_grad(set_to_none=True); out=model(xb); loss,p=loss_base(out,yb,model); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step(); losses.append(loss.item());correct+=(out['final_logits'].argmax(1)==yb).sum().item();n+=len(yb);gates+=out['gate'].detach().tolist();qcr+=out['qcr'].detach().tolist()
        for k in parts: parts[k]+=p[k].item()
      sch.step(); test_m=metrics(model,test,device,torch.arange(62,device=device),classes); row={'epoch':epoch,'lr_backbone':opt.param_groups[0]['lr'],'train_total_loss':np.mean(losses),'train_final_ce':parts['final_ce']/len(losses),'train_classical_ce':parts['classical_ce']/len(losses),'train_quantum_residual_loss':parts['q']/len(losses),'train_acc':correct/n,'test_acc':test_m['accuracy'],'test_macro_f1':test_m['macro_f1'],'quantum_gate_mean':np.mean(gates),'quantum_alpha':float(.25*torch.sigmoid(model.alpha).detach()),'quantum_correction_ratio':np.mean(qcr),'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20,'epoch_seconds':time.time()-started}
      with open(base/'base_training.csv','a',newline='') as f: csv.DictWriter(f,fieldnames=fields).writerow(row)
      with open(events,'a') as f: f.write(json.dumps({'event':'epoch','timestamp':time.time(),**row})+'\n')
      # Explicit requested target-selected development protocol for the base model.
      if test_m['accuracy'] > best_acc:
        best_acc=test_m['accuracy']; save_checkpoint(best_path,epoch=epoch,model=model.state_dict(),metrics=test_m,selection_protocol='DEV_ONLY target-selected highest test accuracy')
      save_checkpoint(latest,epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=rng_state(),active_channels=list(range(62)),removed_history=[],best_target_accuracy=best_acc)
      print(f'BASE target={target} epoch={epoch}/{args.epochs_base} acc={row["test_acc"]:.4f} f1={row["test_macro_f1"]:.4f}',flush=True)
    final_m=metrics(model,test,device,torch.arange(62,device=device),classes); save_checkpoint(base/'final_epoch.pt',model=model.state_dict(),epoch=args.epochs_base,normalization=normalization); atomic_json(metric_dir/'62ch_final_epoch.json',final_m); atomic_json(metric_dir/'62ch_target_selected.json',torch.load(best_path,map_location='cpu')['metrics'])
    with open(events,'a') as f: f.write(json.dumps({'event':'complete','timestamp':time.time(),'final_epoch_metrics':final_m,'target_selected_best_accuracy':best_acc})+'\n')
    print('QNEURO_TARGET_COMPLETE base_only=1 progressive_requested='+str(args.run_progressive),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--target',required=True);p.add_argument('--data-root');p.add_argument('--output-dir',default='results');p.add_argument('--epochs-base',type=int,default=100);p.add_argument('--batch-size',type=int,default=25);p.add_argument('--quantum-layers',type=int,default=4);p.add_argument('--seed',type=int,default=3);p.add_argument('--resume',action='store_true');p.add_argument('--run-progressive',action='store_true');a=p.parse_args();run_target(a)
if __name__=='__main__': main()

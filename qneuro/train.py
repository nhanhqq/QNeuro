"""QNeuro-v2 strict source-validation trainer."""
import argparse, csv, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
from .data import DATASETS, CHANNEL_POSITIONS, EEG, load, source_normalize, tensor_mask
from .model import LatentSetQuanKAN


def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def configure_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = str(path) + '.tmp'; json.dump(obj, open(tmp, 'w'), indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)); os.replace(tmp, path)


class SubjectBalancedBatchSampler(Sampler):
    def __init__(self, subject_ids, batch_subjects=8, samples_per_subject=4, seed=3):
        self.subject_ids = np.asarray(subject_ids); self.subjects = np.unique(self.subject_ids); self.bs = batch_subjects; self.k = samples_per_subject; self.seed = seed; self.epoch = 0
    def set_epoch(self, epoch): self.epoch = epoch
    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch); pools = {s: np.flatnonzero(self.subject_ids == s) for s in self.subjects}; n_batches = max(1, int(np.ceil(len(self.subject_ids) / float(self.bs * self.k)))); batches = []
        for _ in range(n_batches):
            chosen = rng.choice(self.subjects, size=min(self.bs, len(self.subjects)), replace=False)
            batch = [int(rng.choice(pools[s], self.k, replace=len(pools[s]) < self.k)[j]) for s in chosen for j in range(self.k)]
            batches.append(batch)
        return iter(batches)
    def __len__(self): return max(1, int(np.ceil(len(self.subject_ids) / float(self.bs * self.k))))


def optimizer(model, lr=3e-4, head_lr=5e-4, weight_decay=1e-3):
    groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': lr}]
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}; covered = {id(p) for g in opt.param_groups for p in g['params']}; missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in covered]
    assert trainable == covered, 'optimizer does not cover all trainable parameters'
    return opt, {'total_parameters': sum(p.numel() for p in model.parameters()), 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 'optimizer_parameters': sum(p.numel() for g in opt.param_groups for p in g['params']), 'missing_parameter_names': missing, 'missing_parameter_count': 0}


def batch_metrics(model, loader, device, positions, channels, classes):
    model.eval(); ys=[]; ps=[]; losses=[]
    with torch.no_grad():
        for x, y, lengths, _ in loader:
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True); lengths=lengths.to(device); mask=tensor_mask(lengths, x.shape[1]); out=model(x[:, :, channels], lengths, mask, channels, positions[channels]); logits=out['final_logits']; ys.extend(y.cpu().tolist()); ps.extend(logits.argmax(1).cpu().tolist()); losses.append(F.cross_entropy(logits, y).item())
    pr,re,f1,sup=precision_recall_fscore_support(ys,ps,labels=list(range(classes)),zero_division=0); cm=confusion_matrix(ys,ps,labels=list(range(classes)))
    return {'cross_entropy':float(np.mean(losses)), 'accuracy':float(accuracy_score(ys,ps)), 'macro_precision':float(pr.mean()), 'macro_recall':float(re.mean()), 'macro_f1':float(f1.mean()), 'weighted_f1':float(precision_recall_fscore_support(ys,ps,average='weighted',zero_division=0)[2]), 'balanced_accuracy':float(balanced_accuracy_score(ys,ps)), 'per_class_precision':pr.tolist(), 'per_class_recall':re.tolist(), 'per_class_f1':f1.tolist(), 'support':sup.tolist(), 'confusion_matrix':cm.tolist()}


def supcon(z, y, subjects, temperature=.1):
    z=F.normalize(z,dim=1); sim=z@z.T/temperature; eye=torch.eye(len(z),device=z.device,dtype=torch.bool); valid=~eye; pos=(y[:,None]==y[None,:]) & (subjects[:,None]!=subjects[None,:]) & valid
    logp=sim.masked_fill(eye, -torch.inf).log_softmax(1); count=pos.sum(1); return (-(logp*pos.float()).sum(1)/count.clamp_min(1)).masked_select(count>0).mean() if (count>0).any() else z.sum()*0


def train_epoch(model, loader, device, positions, channels, classes, epoch, args):
    model.train(); total=[]; correct=n=0; subject_losses={}; subject_correct={}; subject_count={}; grad_norms={}; gates=[]; qcr=[]; ent=[]; opt=args.optimizer; sampler=loader.batch_sampler; sampler.set_epoch(epoch)
    for x,y,lengths,subjects in loader:
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True); lengths=lengths.to(device); subjects=subjects.to(device); mask=tensor_mask(lengths,x.shape[1]); opt.zero_grad(set_to_none=True); out=model(x[:, :, channels],lengths,mask,channels,positions[channels]); ce=F.cross_entropy(out['final_logits'],y,label_smoothing=args.label_smoothing); cls=F.cross_entropy(out['classical_logits'],y,label_smoothing=args.label_smoothing); sc=supcon(out['clean'],y,subjects); per=[]
        for s in subjects.unique(): per.append(ce if False else F.cross_entropy(out['final_logits'][subjects==s],y[subjects==s]))
        vrex=torch.stack(per).var(unbiased=False) if len(per)>1 else ce*0; lvrex=0 if epoch <= args.vrex_warmup else args.lambda_vrex*min(1.,(epoch-args.vrex_warmup)/10); residual=F.one_hot(y,classes).float()-out['classical_logits'].softmax(-1).detach(); qloss=F.smooth_l1_loss(torch.tanh(out['qaux']),residual); alpha_weight=0 if epoch <= args.quantum_warmup else 1; loss=ce+args.lambda_cls*cls+args.lambda_supcon*sc+lvrex*vrex+alpha_weight*args.lambda_q*qloss+1e-5*model.kan_regularization(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total.append(float(loss.item())); correct+=(out['final_logits'].argmax(1)==y).sum().item(); n+=len(y); gates.extend(out['gate'].detach().cpu().tolist()); qcr.extend(out['qcr'].detach().cpu().tolist()); ent.extend(out['entropy'].detach().cpu().tolist())
        pred=out['final_logits'].argmax(1)
        for s in subjects.unique().tolist():
            sel=subjects==s; subject_losses.setdefault(int(s),[]).append(float(F.cross_entropy(out['final_logits'][sel],y[sel]).detach().item())); subject_correct[int(s)]=subject_correct.get(int(s),0)+int((pred[sel]==y[sel]).sum()); subject_count[int(s)]=subject_count.get(int(s),0)+int(sel.sum())
    for name in ('qaux','qread','quantum.rot','clean'):
        ps=[p for n,p in model.named_parameters() if n == name or n.startswith(name+'.')]; grad_norms[name]=float(torch.sqrt(sum((p.grad.detach().square().sum() for p in ps if p.grad is not None), torch.tensor(0.,device=device))).item())
    subj_acc={str(s):subject_correct[s]/max(subject_count[s],1) for s in subject_correct}; return {'train_loss':float(np.mean(total)), 'train_accuracy':correct/max(n,1), 'train_subject_accuracy':json.dumps(subj_acc,sort_keys=True), 'worst_source_accuracy':float(min(subj_acc.values())) if subj_acc else 0., 'supcon_loss':float(sc.detach()), 'vrex_loss':float(vrex.detach()), 'quantum_residual_loss':float(qloss.detach()), 'mean_gate':float(np.mean(gates)), 'mean_entropy':float(np.mean(ent)), 'mean_alpha':float((.5*torch.sigmoid(model.alpha_raw)).detach()), 'mean_qcr':float(np.mean(qcr)), 'grad_norms':json.dumps(grad_norms,sort_keys=True)}


def run_target(args):
    seed_all(args.seed); configure_cuda(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); x,y,persons,subject_ids,classes=load(args.dataset,args.data_root); target=args.target; target_idx=np.flatnonzero(persons==target); source_idx=np.flatnonzero(persons!=target); groups=persons[source_idx]; split=GroupShuffleSplit(n_splits=1,test_size=args.val_subject_fraction,random_state=args.seed); tr_rel,va_rel=next(split.split(source_idx,y[source_idx],groups)); tr=source_idx[tr_rel]; va=source_idx[va_rel]; assert set(persons[tr]).isdisjoint(set(persons[va])); assert target not in set(persons[tr])|set(persons[va]); x,norm=source_normalize(x,source_idx); lengths=np.asarray([np.count_nonzero(np.any(np.abs(v)>0,axis=(1,2))) for v in x]); root=Path(args.output_dir)/args.dataset/f'target_{target}'; root.mkdir(parents=True,exist_ok=True); atomic_json(root/'normalization.json',norm); atomic_json(root/'split.json',{'target':target,'train_subjects':sorted(set(persons[tr])),'validation_subjects':sorted(set(persons[va])),'target_samples':int(len(target_idx)),'protocol':'strict source-only inner subject split; target evaluated once after selection'}); atomic_json(root/'config.json',{**vars(args),'device':str(device),'classes':classes,'seed':args.seed,'padding_ratio':float(1-np.mean(lengths/x.shape[1]))}); runtime_path=root/'runtime.jsonl'; runtime_path.write_text(json.dumps({'event':'start','dataset':args.dataset,'target':target,'device':str(device),'seed':args.seed,'padding_ratio':float(1-np.mean(lengths/x.shape[1]))})+'\n')
    train_ds=EEG(x[tr],y[tr],lengths[tr],subject_ids[tr]); val_ds=EEG(x[va],y[va],lengths[va],subject_ids[va]); target_ds=EEG(x[target_idx],y[target_idx],lengths[target_idx],subject_ids[target_idx]); sampler=SubjectBalancedBatchSampler(train_ds.subject_ids,args.batch_subjects,args.samples_per_subject,args.seed); train_loader=DataLoader(train_ds,batch_sampler=sampler,num_workers=0,pin_memory=True); val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True); target_loader=DataLoader(target_ds,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True); model=LatentSetQuanKAN(classes,no_quantum=args.no_quantum,no_entropy_gate=args.no_entropy_gate,beta=args.beta).to(device); opt,coverage=optimizer(model,args.lr,args.head_lr,args.weight_decay); args.optimizer=opt; atomic_json(root/'optimizer.json',coverage); positions=torch.tensor(CHANNEL_POSITIONS,device=device); channels=torch.arange(62,device=device); fields=['epoch','train_loss','train_accuracy','train_subject_accuracy','worst_source_accuracy','source_val_accuracy','source_val_macro_f1','source_val_balanced_accuracy','supcon_loss','vrex_loss','quantum_residual_loss','mean_gate','mean_entropy','mean_alpha','mean_qcr','grad_norms','epoch_seconds']; best=-1.; bad=0; csv_path=root/'training.csv'; f=open(csv_path,'w',newline=''); writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
    for epoch in range(1,args.epochs+1):
        started=time.time(); trm=train_epoch(model,train_loader,device,positions,channels,classes,epoch,args); vm=batch_metrics(model,val_loader,device,positions,channels,classes); row={'epoch':epoch,'epoch_seconds':time.time()-started,**trm,'source_val_accuracy':vm['accuracy'],'source_val_macro_f1':vm['macro_f1'],'source_val_balanced_accuracy':vm['balanced_accuracy']}; writer.writerow({k:row.get(k) for k in fields}); f.flush();
        with runtime_path.open('a') as rf: rf.write(json.dumps({'event':'epoch','timestamp':time.time(),'epoch':epoch,'train':trm,'source_val':vm})+'\n')
        print(f'V2 dataset={args.dataset} target={target} epoch={epoch}/{args.epochs} train={trm["train_accuracy"]:.4f} val_acc={vm["accuracy"]:.4f} val_f1={vm["macro_f1"]:.4f}',flush=True)
        if vm['macro_f1'] > best:
            best=vm['macro_f1']; bad=0; torch.save({'model':model.state_dict(),'epoch':epoch,'source_val':vm,'coverage':coverage},root/'best_source_val.pt')
        else: bad+=1
        if bad >= args.patience: break
    f.close(); state=torch.load(root/'best_source_val.pt',map_location=device); model.load_state_dict(state['model']); tm=batch_metrics(model,target_loader,device,positions,channels,classes); atomic_json(root/'final_metrics.json',tm); atomic_json(root/'confusion_matrix.json',tm['confusion_matrix']); runtime={'epochs_completed':state['epoch'],'best_source_val_macro_f1':best,'target_evaluated_once':True,'device':str(device),'peak_vram_mb':float(torch.cuda.max_memory_allocated()/2**20) if torch.cuda.is_available() else 0}; atomic_json(root/'runtime.json',runtime)
    with runtime_path.open('a') as rf: rf.write(json.dumps({'event':'complete','timestamp':time.time(),'target_test':tm,**runtime})+'\n')
    print(f'QNEURO_V2_COMPLETE dataset={args.dataset} target={target} epoch={state["epoch"]} target_acc={tm["accuracy"]:.4f} target_f1={tm["macro_f1"]:.4f}',flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=DATASETS,required=True); p.add_argument('--target',required=True); p.add_argument('--data-root'); p.add_argument('--output-dir',default='results_v2'); p.add_argument('--epochs',type=int,default=150); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--batch-subjects',type=int,default=8); p.add_argument('--samples-per-subject',type=int,default=4); p.add_argument('--val-subject-fraction',type=float,default=.2); p.add_argument('--patience',type=int,default=20); p.add_argument('--seed',type=int,default=3); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--head-lr',type=float,default=5e-4); p.add_argument('--weight-decay',type=float,default=1e-3); p.add_argument('--label-smoothing',type=float,default=.0); p.add_argument('--lambda-cls',type=float,default=.2); p.add_argument('--lambda-supcon',type=float,default=.08); p.add_argument('--lambda-vrex',type=float,default=.05); p.add_argument('--lambda-q',type=float,default=.15); p.add_argument('--vrex-warmup',type=int,default=10); p.add_argument('--quantum-warmup',type=int,default=10); p.add_argument('--beta',type=float,default=.1); p.add_argument('--no-quantum',action='store_true'); p.add_argument('--no-entropy-gate',action='store_true'); run_target(p.parse_args())
if __name__=='__main__': main()

#!/usr/bin/env python3
"""One strict LOSO fold. Target data is never read during model selection."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ext_ablation.model import FEATURE_MASKS, VARIANTS, HybridNode11ExtendedAblation, parameter_count
from qneuro_paperlite.features import FEATURE_NAMES, FEATURE_VERSION
from src.evaluation.metrics import metrics
from src.utils.seed import seed_everything


def gpu(features, indices, device):
    value = np.array(features[indices], dtype=np.float32, copy=True)
    return torch.from_numpy(value).to(device)


@torch.no_grad()
def evaluate(model, x, y, criterion, batch_size):
    model.eval(); pred=[]; prob=[]; loss=0.0
    for start in range(0, len(y), batch_size):
        stop=min(start+batch_size, len(y)); logits=model(x[start:stop]); target=y[start:stop]
        loss += criterion(logits, target).item() * len(target)
        pred.append(logits.argmax(1).cpu()); prob.append(logits.softmax(1).cpu())
    result=metrics(y.cpu().numpy(), torch.cat(pred).numpy(), torch.cat(prob).numpy())
    result["loss"] = loss / len(y)
    return result


def make_strict_fold(labels, groups, fold, seed, out):
    outer=list(LeaveOneGroupOut().split(np.zeros(len(labels)), labels, groups))
    train_all, target = outer[fold]
    source_y=labels[train_all]
    inner=StratifiedShuffleSplit(n_splits=1, test_size=.20, random_state=seed + fold)
    local_train, local_val=next(inner.split(np.zeros(len(train_all)), source_y))
    train, val=train_all[local_train], train_all[local_val]
    source_subjects=sorted(map(str, np.unique(groups[train_all])))
    target_subjects=sorted(map(str, np.unique(groups[target])))
    record={"fold":fold, "protocol":"strict LOSO: source-pool stratified validation; target evaluated once after selection",
            "seed":seed, "train_indices":train.tolist(), "validation_indices":val.tolist(), "test_indices":target.tolist(),
            "source_subjects":source_subjects, "target_subjects":target_subjects,
            "validation_scheme":"stratified source-pool holdout", "validation_fraction":.20,
            "test_evaluations":1}
    if set(source_subjects) & set(target_subjects) or len(target_subjects) != 1:
        raise AssertionError("target subject leakage")
    if set(train) & set(val) or set(train) & set(target) or set(val) & set(target):
        raise AssertionError("index leakage")
    path=out/"splits"/f"fold_{fold}.json"; path.parent.mkdir(parents=True, exist_ok=True)
    # Variants share this deterministic fold descriptor; atomic replacement
    # avoids a truncated JSON if several isolated workers reach it together.
    temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(record, indent=2)); os.replace(temporary,path)
    return train, val, target, record


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--output",required=True)
    p.add_argument("--fold",type=int,required=True); p.add_argument("--variant",choices=VARIANTS,required=True)
    p.add_argument("--seed",type=int,default=7); p.add_argument("--epochs",type=int,default=50); p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--fs",type=float,required=True); p.add_argument("--frame-seconds",type=float,required=True); p.add_argument("--hop-seconds",type=float,required=True)
    p.add_argument("--patience",type=int,default=12); p.add_argument("--rz-noise-std",type=float,default=.10); p.add_argument("--dry-run",action="store_true"); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("strict ablation requires CUDA")
    seed_everything(a.seed); device=torch.device("cuda"); data=Path(a.data); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    raw=np.load(data/"labels.npy"); classes=np.unique(raw); labels=np.searchsorted(classes,raw).astype(np.int64)
    meta=pd.read_csv(data/"meta.csv"); groups=meta["subject"].to_numpy()
    if len(meta) != len(labels): raise RuntimeError("metadata/label length mismatch")
    train_i, val_i, test_i, split=make_strict_fold(labels,groups,a.fold,a.seed,out)
    # Read-only by design: an ablation must never rebuild or mutate a legacy
    # feature cache. The cache is audited by the original pipeline beforehand.
    cache=data/"cache"/f"{FEATURE_VERSION}.npy"
    if not cache.exists():
        raise RuntimeError(f"missing read-only cache: {cache}; build it outside this ablation campaign")
    features=np.load(cache,mmap_mode="r")
    train_x,val_x,test_x=(gpu(features,idx,device) for idx in (train_i,val_i,test_i))
    mean=train_x.mean((0,1,2)); scale=train_x.var((0,1,2),unbiased=False).clamp_min(1e-12).sqrt()
    for x in (train_x,val_x,test_x): x.sub_(mean).div_(scale)
    train_y=torch.as_tensor(labels[train_i],device=device); val_y=torch.as_tensor(labels[val_i],device=device); test_y=torch.as_tensor(labels[test_i],device=device)
    counts=np.bincount(labels[train_i],minlength=len(classes))
    if np.any(counts==0): raise RuntimeError(f"training split misses class: {counts.tolist()}")
    weight=torch.as_tensor(np.sqrt(len(train_i)/(len(classes)*counts)),dtype=torch.float32,device=device)
    # The augmentation table changes only this train-time RZ standard deviation.
    rz_noise_std={"no_rz_augmentation":0.0,"high_rz_augmentation":0.20}.get(a.variant,a.rz_noise_std)
    model=HybridNode11ExtendedAblation(len(classes),features.shape[2],a.variant,rz_noise_std).to(device)
    criterion=torch.nn.CrossEntropyLoss(weight=weight,label_smoothing=.03); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=1e-5)
    run=out/"runs"/a.variant/f"fold_{a.fold}_seed_{a.seed}"; run.mkdir(parents=True,exist_ok=True)
    fields=["epoch","train_loss","train_accuracy","val_loss","val_accuracy","val_balanced_accuracy","val_macro_f1","learning_rate","epoch_seconds"]
    best=-float("inf"); best_epoch=-1; stale=0; history=run/"epochs.csv"
    with history.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for epoch in range(1 if a.dry_run else a.epochs):
            tick=time.time(); model.train(); order=torch.randperm(len(train_y),device=device); loss_sum=0.; pred=[]; truth=[]
            for start in range(0,len(order),a.batch_size):
                ix=order[start:start+a.batch_size]; opt.zero_grad(set_to_none=True); logits=model(train_x[ix]); loss=criterion(logits,train_y[ix])
                if hasattr(model.classifier,"smoothness_penalty"): loss=loss+1e-4*model.classifier.smoothness_penalty()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); loss_sum+=loss.item()*len(ix); pred.append(logits.argmax(1).detach().cpu()); truth.append(train_y[ix].detach().cpu())
            train_m=metrics(torch.cat(truth).numpy(),torch.cat(pred).numpy()); val_m=evaluate(model,val_x,val_y,criterion,a.batch_size)
            writer.writerow({"epoch":epoch,"train_loss":loss_sum/len(train_y),"train_accuracy":train_m["accuracy"],"val_loss":val_m["loss"],"val_accuracy":val_m["accuracy"],"val_balanced_accuracy":val_m["balanced_accuracy"],"val_macro_f1":val_m["macro_f1"],"learning_rate":opt.param_groups[0]["lr"],"epoch_seconds":time.time()-tick}); f.flush(); sched.step()
            score=val_m["macro_f1"]
            if score > best:
                best,best_epoch,stale=score,epoch,0; torch.save({"model":model.state_dict(),"epoch":epoch,"validation":val_m},run/"best_source_validation.pt")
            else: stale+=1
            if stale >= a.patience: break
    chosen=torch.load(run/"best_source_validation.pt",map_location=device); model.load_state_dict(chosen["model"])
    # The only held-out target inference in the entire run.
    test_m=evaluate(model,test_x,test_y,criterion,a.batch_size)
    torch.save({"model":model.state_dict(),"epoch":best_epoch,"validation":chosen["validation"],"test":test_m},run/"final_selected_on_source_validation.pt")
    (run/"strict_test_once.json").write_text(json.dumps(test_m,indent=2))
    (run/"run.json").write_text(json.dumps({"variant":a.variant,"model_selection":"maximum source-validation macro-F1","target_selection":False,"test_evaluations":1,"best_epoch":best_epoch,"trainable_parameters":parameter_count(model),"num_workers":0,"feature_masked_indices":list(FEATURE_MASKS[a.variant]),"rz_noise_std":rz_noise_std,"source_class_counts":counts.tolist(),"feature_version":FEATURE_VERSION,"feature_names":list(FEATURE_NAMES),"split":split,"python":platform.python_version(),"torch":torch.__version__,"device":torch.cuda.get_device_name()},indent=2))


if __name__ == "__main__": main()

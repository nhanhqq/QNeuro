#!/usr/bin/env python3
import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--output',default='results/chsz');p.add_argument('--epochs',type=int,default=100);p.add_argument('--variant',default='quantum',choices=['quantum','classical','identity','no_bilstm']);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
for fold in range(3):
 cmd=[sys.executable,'scripts/train_chsz.py','--fold',str(fold),'--output',a.output,'--epochs',str(a.epochs),'--variant',a.variant]
 if a.dry_run:cmd.append('--dry-run')
 subprocess.check_call(cmd)

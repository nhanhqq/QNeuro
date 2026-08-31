"""Sequential all-target driver. Scheduler runs one of these per SEED family."""
import argparse, subprocess, sys
from qneuro.data import load
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--epochs-base',type=int,default=100); p.add_argument('--output-dir',default='results'); p.add_argument('--batch-size',type=int,default=25); p.add_argument('--resume',action='store_true'); a=p.parse_args()
_,_,persons,_=load(a.dataset)
for target in sorted(set(persons),key=lambda s:int(s[1:])):
    cmd=[sys.executable,'run_target.py','--dataset',a.dataset,'--target',target,'--epochs-base',str(a.epochs_base),'--batch-size',str(a.batch_size),'--output-dir',a.output_dir]
    if a.resume:cmd.append('--resume')
    subprocess.run(cmd,check=True)

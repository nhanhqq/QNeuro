"""Audit and summarize QNeuro-v2 strict-LOSO artifacts."""
import argparse, json, statistics
from pathlib import Path

DATASETS = ('seed', 'seediv', 'seedv', 'seedvii')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='results_v2'); a=p.parse_args(); root=Path(a.root); total=0
    print('| dataset | complete | expected | accuracy mean ± std | macro-F1 mean ± std |')
    print('|---|---:|---:|---:|---:|')
    for d in DATASETS:
        expected={'seed':15,'seediv':15,'seedv':16,'seedvii':20}[d]; vals=[]; f1=[]; invalid=[]
        for target in range(1,expected+1):
            r=root/d/('target_P%d'%target); m=r/'final_metrics.json'; s=r/'split.json'; n=r/'normalization.json'; runtime=r/'runtime.json'
            if not (m.exists() and s.exists() and n.exists() and runtime.exists()): continue
            obj=json.load(open(m)); split=json.load(open(s));
            if split.get('target') != 'P%d'%target or split.get('target') in split.get('train_subjects',[]) or split.get('target') in split.get('validation_subjects',[]): invalid.append(r); continue
            vals.append(float(obj['accuracy'])); f1.append(float(obj['macro_f1']))
        total += len(vals)
        fmt=lambda z: ('%.4f ± %.4f'%(statistics.mean(z),statistics.pstdev(z))) if z else '--'
        print('| %s | %d | %d | %s | %s |'%(d,len(vals),expected,fmt(vals),fmt(f1)))
        if invalid: print('INVALID', *invalid)
    print('complete_total=%d/66'%total)

if __name__=='__main__': main()

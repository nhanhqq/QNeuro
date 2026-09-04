#!/usr/bin/env python3
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.audit_chsz import audit
p=argparse.ArgumentParser();p.add_argument('--data',default='CHSZ');p.add_argument('--results',default='results');a=p.parse_args();print(json.dumps(audit(a.data,a.results),indent=2))

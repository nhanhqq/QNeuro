import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut


def make_splits(y, groups, out_dir):
 """Return one LOSO split per subject with no validation partition."""
 out=Path(out_dir);out.mkdir(parents=True,exist_ok=True); result=[]
 outer=LeaveOneGroupOut()
 for fold,(train,test) in enumerate(outer.split(np.zeros(len(y)),y,groups)):
  ss=lambda ix: sorted(map(str,np.unique(groups[ix])))
  a,c=map(set,(ss(train),ss(test)));assert not(a&c) and len(c) == 1
  rec={'fold':fold,'protocol':'LOSO target-selected: train N-1 subjects, test one held-out subject every epoch; no validation split','train_subjects':sorted(a),'test_subjects':sorted(c),'train_indices':train.tolist(),'test_indices':test.tolist()}
  (out/f'fold_{fold}.json').write_text(json.dumps(rec,indent=2));result.append(rec)
 return result

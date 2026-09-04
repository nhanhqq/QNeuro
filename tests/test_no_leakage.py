import numpy as np
from src.data.split import make_splits
def test_subjects_disjoint(tmp_path):
 y=np.tile([0,1],18);g=np.repeat(np.arange(9),4);s=make_splits(y,g,tmp_path);assert len(s)==9
 for split in s:
  assert not(set(split['train_subjects']) & set(split['test_subjects']))
  assert 'validation_subjects' not in split
  assert len(split['test_subjects']) == 1

import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,confusion_matrix,roc_auc_score
def metrics(y,p,prob=None):
 pr,re,f,_=precision_recall_fscore_support(y,p,average=None,zero_division=0);out={'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'macro_f1':float(precision_recall_fscore_support(y,p,average='macro',zero_division=0)[2]),'weighted_f1':float(precision_recall_fscore_support(y,p,average='weighted',zero_division=0)[2]),'per_class_precision':pr.tolist(),'per_class_recall':re.tolist(),'per_class_f1':f.tolist(),'confusion_matrix':confusion_matrix(y,p).tolist()}
 if prob is not None and len(np.unique(y))>1:
  try:out['ovr_auroc']=float(roc_auc_score(y,prob,multi_class='ovr')) if prob.shape[1]>2 else float(roc_auc_score(y,prob[:,1]))
  except ValueError:out['ovr_auroc']=None
 return out

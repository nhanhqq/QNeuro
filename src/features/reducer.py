"""Fold-local scaler/PCA; intentionally has no dataset-wide fit entrypoint."""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def fit_train_reducer(train_frames, components=4, seed=7):
 scaler=StandardScaler().fit(train_frames.reshape(-1,train_frames.shape[-1]));pca=PCA(components,random_state=seed).fit(scaler.transform(train_frames.reshape(-1,train_frames.shape[-1])))
 return scaler,pca
def transform_frames(frames,scaler,pca):
 return pca.transform(scaler.transform(frames.reshape(-1,frames.shape[-1]))).reshape(len(frames),frames.shape[1],-1)

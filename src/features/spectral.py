import numpy as np
BANDS={'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,45)}
FEATURE_NAMES=list(BANDS)+['spectral_entropy']
def spectral_features(x, fs=128., frame_seconds=1., hop_seconds=.5):
    """x is [N,C,T] or one [C,T], output [N,L,C*6], preserving time."""
    one=x.ndim==2
    if one:x=x[None]
    n,c,t=x.shape; fl=int(round(fs*frame_seconds)); hop=int(round(fs*hop_seconds))
    if t<fl: raise ValueError(f'signal length {t} shorter than frame {fl}')
    starts=np.arange(0,t-fl+1,hop); win=np.hanning(fl).astype(np.float32); freq=np.fft.rfftfreq(fl,1/fs)
    ans=np.empty((n,len(starts),c*6),np.float32)
    for j,s in enumerate(starts):
      z=x[:,:,s:s+fl].astype(np.float32,copy=False);z=z-z.mean(axis=-1,keepdims=True); power=np.abs(np.fft.rfft(z*win,axis=-1))**2
      vals=[]
      for lo,hi in BANDS.values():
       mask=(freq>=lo)&(freq<hi); vals.append(np.log(power[:,:,mask].mean(axis=-1)+1e-8))
      p=power/(power.sum(axis=-1,keepdims=True)+1e-12);vals.append(-(p*np.log(p+1e-12)).sum(axis=-1))
      ans[:,j]=np.stack(vals,axis=-1).reshape(n,-1)
    return ans[0] if one else ans

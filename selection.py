"""Uncertainty/diversity sampling for an unlabeled pool; NumPy only."""
import numpy as np

def select(embeddings, uncertainty, budget, weight=.5):
    x=np.asarray(embeddings,dtype=float); u=np.asarray(uncertainty,dtype=float)
    if x.ndim!=2 or u.shape!=(len(x),): raise ValueError('Expected N x D embeddings and N uncertainty scores')
    if not np.isfinite(x).all() or not np.isfinite(u).all(): raise ValueError('Inputs must be finite')
    if not 0<=budget<=len(x) or not 0<=weight<=1: raise ValueError('Invalid budget or weight')
    if budget==0: return []
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    u=(u-u.min())/max(float(np.ptp(u)),1e-12)
    chosen=[int(np.argmax(u))]; distance=np.full(len(x),2.)
    while len(chosen)<budget:
        distance=np.minimum(distance,np.clip(1-x@x[chosen[-1]],0,2))
        scores=weight*u+(1-weight)*distance/2
        scores[chosen]=-np.inf; chosen.append(int(np.argmax(scores)))
    return chosen

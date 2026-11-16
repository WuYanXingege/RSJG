"""CPU endpoint scores. Representative sets and probability samples are distinct."""
import numpy as np

def endpoint_metrics(prediction,truth,valid):
    p=np.asarray(prediction);y=np.asarray(truth);v=np.asarray(valid,bool)
    if p.ndim!=4 or p.shape[0]!=len(y) or p.shape[2:]!=y.shape[1:] or v.shape!=y.shape[:2]:raise ValueError('expected [S,K,N,2], [S,N,2], [S,N]')
    errors=np.linalg.norm(p-y[:,None],axis=-1).min(1)[v]
    if not len(errors) or not np.isfinite(errors).all():raise ValueError('empty/nonfinite valid predictions')
    return {'minFDE':float(errors.mean()),'P50':float(np.quantile(errors,.5)),'P90':float(np.quantile(errors,.9)),'P95':float(np.quantile(errors,.95)),**{f'recall_{r}':float((errors<=r).mean()) for r in [.25,.5,1.]},'agents':len(errors)}

def energy_score_iid(samples,truth):
    """Unbiased pair term for genuine IID samples; not for coverage representatives."""
    x=np.asarray(samples);y=np.asarray(truth)
    if x.ndim!=3 or x.shape[-1]!=2 or y.shape!=x.shape[1:] or len(x)<2:raise ValueError('IID [M,N,2] with M>=2 required')
    first=np.linalg.norm(x-y[None],axis=-1).mean(0)
    pair=np.linalg.norm(x[:,None]-x[None,:],axis=-1).sum((0,1))/(len(x)*(len(x)-1))
    return first-.5*pair

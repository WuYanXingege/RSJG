import numpy as np
from whole_scene_swap.selector import distances,objectives,select_swaps

def brute(raw,s):
 d=distances(raw);s=list(s);j0,p=objectives(d,s);hist=[]
 while True:
  options=[]
  for slot,out in enumerate(s):
   for inc in sorted(set(range(len(raw)))-set(s)):
    new=s.copy();new[slot]=inc;j,q=objectives(d,new)
    if j<=j0 and p-q>1e-12:options.append(((q,out,inc),slot,new))
  if not options:return np.array(s),hist
  key,slot,s=min(options,key=lambda x:x[0]);p=key[0];hist.append((key[1],key[2],slot))

def test_two_people():
 raw=np.array([[[0,0],[0,0]],[[0,0],[1,0]],[[1,0],[0,0]],[[1,0],[1,0]]],dtype=float)
 d=distances(raw)
 assert objectives(d,[0,1])==(.25,.25)
 assert objectives(d,[0,3])==(.25,0.)
 chosen,result=select_swaps(raw,[0,1]);expected,h=brute(raw,[0,1])
 assert np.array_equal(chosen,expected) and result['P']==0

def test_against_exhaustive():
 for seed in range(15):
  raw=np.random.default_rng(seed).normal(size=(12,4,2));s=[8,2,0]
  actual,r=select_swaps(raw,s);expected,h=brute(raw,s)
  assert np.array_equal(actual,expected)
  assert [(v['out'],v['incoming'],v['slot']) for v in r['history']]==h
  assert r['constraint_violations']==0

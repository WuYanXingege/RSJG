#!/usr/bin/env python3
"""Public GitHub immutable raw bytes vs local git objects; no .git mutations."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,time,urllib.parse,urllib.request
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    changed=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',a.commit],cwd=root,text=True).splitlines()
    proxy=os.environ.get('HTTPS_PROXY','')
    def fetch(name):
        url='https://raw.githubusercontent.com/WuYanXingege/RSJG/'+a.commit+'/'+urllib.parse.quote(name,safe='/')
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':proxy}) if proxy else urllib.request.ProxyHandler({}))
        req=urllib.request.Request(url,headers={'Cache-Control':'no-cache','User-Agent':'RSJG-immutable-preflight-verifier'})
        with opener.open(req,timeout=45) as response:remote=response.read(16*1024**2+1)
        if len(remote)>16*1024**2:raise RuntimeError('remote per-file bound')
        local=subprocess.check_output(['git','show',a.commit+':'+name],cwd=root)
        if remote!=local:raise RuntimeError('remote bytes differ '+name)
        blob=hashlib.sha1(b'blob '+str(len(remote)).encode()+b'\0'+remote).hexdigest()
        expected=subprocess.check_output(['git','rev-parse',a.commit+':'+name],cwd=root,text=True).strip()
        if blob!=expected:raise RuntimeError('Git blob differs '+name)
        return dict(path=name,url=url,bytes=len(remote),sha256=hashlib.sha256(remote).hexdigest(),git_blob_sha1=blob,byte_identical=True)
    results=[];errors=[];start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,name):name for name in changed}
        for f in concurrent.futures.as_completed(futures):
            try:results.append(f.result())
            except Exception as e:errors.append(dict(path=futures[f],error=str(e)))
            if (len(results)+len(errors))%25==0:print(json.dumps(dict(verified=len(results),errors=len(errors))),flush=True)
    receipt=dict(commit=a.commit,all_files_exact=not errors and len(results)==len(changed),file_count=len(results),expected=len(changed),files=sorted(results,key=lambda x:x['path']),errors=errors,seconds=time.monotonic()-start)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    with a.receipt.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('files','errors')}));raise SystemExit(0 if receipt['all_files_exact'] else 2)
if __name__=='__main__':main()

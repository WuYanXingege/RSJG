"""Verify GitHub fetch_file base64 responses against an immutable local commit.

The MCP response files are supplied externally; this script performs no network
access and no model/CUDA calls. Receipt is an output artifact, not repository code.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--commit',required=True)
p.add_argument('--payload-dir',type=Path,required=True)
p.add_argument('--receipt',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[4]
paths=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',a.commit],cwd=root,text=True).splitlines()
rows=[]
for i,path in enumerate(paths):
    response=json.loads((a.payload_dir/(str(i)+'.json')).read_text())
    assert response['path']==path and response['commit']==a.commit
    r=response['response']
    assert r['encoding']=='base64'
    remote=base64.b64decode(r['content'])
    local=subprocess.check_output(['git','show',a.commit+':'+path],cwd=root)
    assert remote==local, path
    blob=hashlib.sha1(b'blob '+str(len(remote)).encode()+b'\0'+remote).hexdigest()
    assert blob==r['sha'], path
    rows.append({'path':path,'bytes':len(remote),'sha256':hashlib.sha256(remote).hexdigest(),
                 'github_blob_sha1':r['sha'],'byte_identical':True})
receipt={'commit':a.commit,'all_files_exact':True,'file_count':len(rows),'files':rows}
a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'commit':a.commit,'all_files_exact':True,'file_count':len(rows),'receipt':str(a.receipt)},indent=2))

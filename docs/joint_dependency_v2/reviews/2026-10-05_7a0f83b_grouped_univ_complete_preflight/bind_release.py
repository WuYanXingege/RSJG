"""Byte-only archive/member verification; never parses outer coordinates."""
import ast,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R))
from src.p2_protocol import Registry,file_hash,digest,path,sealed
from src.p2_grouped_artifacts import write_manifest
from src.p2_observation import atomic_json
D=Path(__file__).resolve().parent;O=R/'outputs/joint_dependency_v2/grouped_univ_preflight_7a0f83b_20261005'
archive=O/'downloads/author_dataset.zip';sha=file_hash(archive)
commit='297d508558c10831983ea4b19c2b3e657459a449'
author={}
for name in ['src/data_src/experiment_src/experiment_eth5.py','src/data_src/scene_src/scene_eth5.py','src/data_src/dataset_src/dataset_eth5.py','scripts/download_data.sh']:
    try:raw=subprocess.check_output(['git','-C',str(R.parent/'GDTS'),'show',commit+':'+name],stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        if name=='scripts/download_data.sh':
            name='download_data.sh'
            raw=subprocess.check_output(['git','-C',str(R.parent/'GDTS'),'show',commit+':'+name])
        else:raise
    author[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode(),url='https://github.com/Winderting/GDTS/blob/'+commit+'/'+name)
tree=ast.parse(author['src/data_src/experiment_src/experiment_eth5.py']['text'])
mapping=None
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and any(isinstance(x,ast.Attribute) and x.attr=='data_file_name_to_scene' for x in node.targets):mapping=ast.literal_eval(node.value)
assert mapping
mp=D/'GROUPED_RUNTIME_MANIFEST.json';m=json.loads(mp.read_text());reg=Registry(m,m['manifest_hash'])
refs={}
for r in reg.records:
    s=r['source'];assert mapping[Path(s['original_source_path']).stem]==s['scene_family']
    for alias in r['aliases']:
        p=path(alias);refs[str(p.resolve())]=dict(member='data/eth5/'+str(p).split('/eth5/',1)[1],path=str(p),sha256=s['source_sha256'],kind='raw_byte_hash_only')
for scene,assets in m['assets'].items():
    for name in ('H.txt','RGB.jpg','pred_mask.png'):
        ref=assets[name];refs[ref['path']]=dict(member='data/eth5/'+scene+'/'+name,path=ref['path'],sha256=ref['sha256'],kind='static_asset')
members=[]
with zipfile.ZipFile(archive) as z:
    assert not any(Path(x.filename).suffix.lower() in {'.mp4','.avi','.pt','.pth','.ckpt'} for x in z.infolist())
    assert archive.stat().st_size<4*1024**3
    for ref in refs.values():
        raw=z.read(ref['member']);member_sha=hashlib.sha256(raw).hexdigest();actual=file_hash(ref['path'])
        members.append(dict(ref,member_sha256=member_sha,local_actual_sha256=actual,bytes=len(raw),match=member_sha==actual==ref['sha256']))
matched=all(x['match'] for x in members)
receipt=dict(status='VERIFIED_AUTHOR_ARCHIVE_MEMBERS' if matched else 'AUTHOR_MEMBER_MISMATCH',url='https://www.dropbox.com/s/luu2t6c6d24xvrb/dataset.zip?dl=1',
    archive_sha256=sha,archive_bytes=archive.stat().st_size,downloaded=True,extracted_files=0,
    members=members,author_code={k:{field:v for field,v in value.items() if field!='text'} for k,value in author.items()},source_scene_mapping=mapping,
    scope='exact author-distributed processed file version; not original recording/converter execution certification',
    retained_unknown=['original recording/session/remap/converter history','specific semantic producer weights and training label scope'])
atomic_json(receipt,D/'AUTHOR_RELEASE_BINDING.json')
evidence=[dict(path=str((D/'AUTHOR_RELEASE_BINDING.json').relative_to(R)),sha256=file_hash(D/'AUTHOR_RELEASE_BINDING.json'))]
q=m['qualification'];q['release_binding']=dict(status=receipt['status'],evidence=evidence)
q['group_isolation']=dict(status='VERIFIED_CROSS_ROLE_GROUPS' if matched else 'UNKNOWN_LOCAL_VERSION_BINDING',evidence=evidence,
    scope='author loader distinct ETH and HOTEL scenes; UNIV three files one group; ZARA all one same-role group; no per-recording certificate')
for scene in q['scenes']:
    scene_matched=all(x['match'] for x in members if x['kind']=='static_asset' and x['member'].startswith('data/eth5/'+scene+'/'))
    q['scenes'][scene].update(geometry_status='VERIFIED_AUTHOR_BINDING' if scene_matched else 'UNKNOWN_ASSET_MISMATCH',evidence=evidence,
        semantic_status='UNKNOWN_LEARNED_RASTER_TRAINING_SCOPE',protocol_permitted=False)
m['qualification']=q
m['preparation_parent_hash']=json.loads((D/'PREPARATION_MANIFEST.json').read_text())['manifest_hash']
write_manifest(m,reg.records,D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json')
atomic_json(dict(status='EXTERNAL_EVIDENCE_BLOCKED',sources=[
    dict(url='https://arxiv.org/pdf/2204.11561',sections=['3.2 printed page 4/PDF index3','4.1 printed page 5/PDF index4','Implementation details printed page 6/PDF index5'],
        finding='Goal-SAR describes learned semantic maps from pretrained Y-Net segmentation; segmentation trained on images of corresponding trajectory-training scenes. No raster-specific checkpoint/fold/label manifest binds these bytes to the new grouped split.'),
    dict(url='https://arxiv.org/html/2311.14922v2',section='III-B',finding='GDTS describes pretrained semantic predictor, not the exact released raster producer weight provenance.')],
    blocked_assets={s:m['assets'][s]['pred_mask.png'] for s in m['assets']},
    remedy='Provide author raster generation manifest linking each SHA to segmentation weight SHA and training scene/label list excluding new inner/outer labels. Alternatively separately authorize a new input variant with an independently qualified train-only preprocessor; do not silently swap maps.'),D/'SEMANTIC_SOURCE_EVIDENCE.json')
print(json.dumps(dict(archive_sha256=sha,members=len(members),all_match=matched,semantic='BLOCKED')))

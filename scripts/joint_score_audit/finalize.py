"""Rehash every protected original asset; emit sanitized identity and zero-step receipt."""
from datetime import datetime,timezone
from common import *


def main():
    setup()
    from bcep.contracts import canonical_hash
    roots=dict(v2=FORMAL,v3=V3RUN,candidate_cost=FORMAL.parent/'zara2_costproxy_stage2_20261119',
        sharedworld=FORMAL.parent/'zara2_sharedworld_stage2_onerun_20261119',previous_replay=REPLAY)
    before=read(OUT/'PROTECTED_BEFORE.json');after={};sizes={}
    for label,root in roots.items():
        for p in root.rglob('*'):
            if p.is_file():after[label+'/'+str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=sha(p))
        a=[v for k,v in after.items() if k.startswith(label+'/')]
        sizes[label]=dict(files=len(a),bytes=sum(v['bytes'] for v in a))
    assert before==after,'protected asset inventory/content changed'
    states={name:read(root/'STATUS.json') for name,root in roots.items() if name!='previous_replay'}
    assert [states[k]['status'] for k in ['v2','candidate_cost','sharedworld','v3']]==['PAUSED_RESUMABLE','SESSION_BUDGET_PAUSED','COMPLETE','COMPLETE']
    assert states['v3']['optimizer_updates']==5040 and states['v3']['completed_epoch']==30
    for label in MODELS:
        assert sha(MODELS[label]['path'])==MODELS[label]['sha']
        for proof in [read(OUT/label/'PROOF.json'),read(DOC/f'GRADIENT_PROOF_{label}.json')]:
            assert proof['optimizer_guard']==GUARD
        write(DOC/f'COLLECTION_PROOF_{label}.json',read(OUT/label/'PROOF.json'))
    frozen=read(V3RUN/'FROZEN_CONFIG.json')
    assert canonical_hash(frozen)=='294054c2cf7605caafe016507bb601e2d85518b70555369ca3690b74138e9e43'
    protocol=read(OUT/'PROTOCOL.json')
    protocol.update(frozen_config_canonical_sha256=canonical_hash(frozen),frozen_config_file_sha256=sha(V3RUN/'FROZEN_CONFIG.json'),
        split_canonical_sha256='716e44710ff74faeffae1a6d7d814687443621e44623a0a9a1233cf8bf4f9bbb',
        bank_sha256=dict(train='2883ba7a847c6dcf34e08159529d1b10803e9ff5626cbffbd9c8bd66aa8d605a',selection_dev='b6aee909ed8bcdac6bb08b93fba554303829f94707cf54e87eccd399a07c1560'),
        single_windows=89,multi_windows=501,N_ge_9_windows=127,agents=4262,dev_max_agents=42,
        strict_tie_definition='complete best-second gap == 0, FP64 sums of original FP32 scores',
        near_tie_definition='0 < complete gap <= 1e-6 (diagnostic only; never alters solver)',
        synthetic_optimizer_steps=0,real_optimizer_steps=0,
        gradient_mode='original train mode required for cuDNN GRU backward; no dropout, no BatchNorm; unchanged score/permutation verified on all 108 sampled forwards',
        final_indirect_gradient='measured initial-ST adjoint followed by original Sinkhorn VJP; avoid FP32 subtraction cancellation')
    write(DOC/'FROZEN_AUDIT.json',protocol)
    receipt=dict(utc=datetime.now(timezone.utc).isoformat(),base_commit=BASE,real_optimizer_steps=0,synthetic_optimizer_steps=0,
        protected_files=len(before),protected_bytes=sum(v['bytes'] for v in before.values()),roots=sizes,
        protected_manifest_canonical_sha256=canonical_hash(before),all_protected_assets_byte_identical=True,
        queue_states={k:v['status'] for k,v in states.items()},historical_v3_updates=5040,historical_v3_complete_epochs=30,
        new_training_launched=False,test_data_used=False,weights_or_banks_uploaded=False,
        original_tracked_source_diff=subprocess.check_output(['git','diff',BASE,'--','bcep','ebjd','bcep_v2','bcep_v3','bcep_joint'],cwd=SOURCE,text=True))
    assert receipt['original_tracked_source_diff']==''
    write(DOC/'ASSET_RECEIPT.json',receipt)
    write(OUT/'PROTECTED_AFTER.json',after)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

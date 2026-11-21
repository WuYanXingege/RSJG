"""Run only CPU algebra tests and worst-case N=57 capacity checks."""
import platform,subprocess,time
from common import *
from energy_prototype import synthetic


def main():
    setup()
    command=[sys.executable,'-m','unittest','discover','-s','tests','-p','test_joint_scene_energy_prototype.py','-v']
    p=subprocess.run(command,cwd=SOURCE,capture_output=True,text=True)
    (DOC/'PROTOTYPE_TESTS.txt').write_text(p.stdout+p.stderr)
    assert p.returncode==0
    m,p=synthetic(n=57,k=20,r=16,dense=True);start=time.perf_counter()
    q,trace=m.sweep(p);seconds=time.perf_counter()-start;m.validate(q)
    assert all(r['energy_after']<=r['energy_before'] for r in trace)
    csvwrite(DOC/'PROTOTYPE_57_TRACE.csv',trace)
    result=dict(platform=platform.platform(),cpu=platform.processor(),numpy=np.__version__,torch=torch.__version__,
        real_data_used=False,optimizer_guard=GUARD,N=57,K=20,R=16,edges=len(m.edges),
        dtype='float64',table_bytes=m.tables.nbytes,latent_bytes=m.z.nbytes,
        FP32_table_bytes=len(m.edges)*20*20*4,FP32_pair_relation_12_bytes=len(m.edges)*20*20*12*4,
        replacement_score_FP32_bytes=57*20*20*4,seconds_one_sweep=seconds,
        native_Hungarian_calls=57*21,accepted=sum(r['accepted'] for r in trace),
        initial_energy=float(m.world_energy(p).sum()),final_energy=float(m.world_energy(q).sum()),
        note='untrained synthetic algebra, CPU only; includes exact second-best tie audit; excludes learned neural edge-feature computation; not JFDE or training throughput')
    write(DOC/'PROTOTYPE_CHECK.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""All 590 official dev windows, native 20-step H DDIM, initial weights only."""
from common import *
from bcep_scene_energy.formal_contract import locks,freeze
from bcep_scene_energy.formal_validation import validation
from bcep_scene_energy.model import build
from bcep_v3.formal_inputs import FormalInputs


def main():
    setup(cuda=True)
    with locks(ROOTS):
        inputs=FormalInputs();frozen,_=freeze(inputs)
        assert frozen==read(ROOTS['run']/'FROZEN_CONFIG.json')
        seed_all(3201);model,_=build(inputs.config,variant='energy_only');model.cuda();initial=model_hash(model)
        assert initial==frozen['initial_model_sha256']
        validate=validation(model,inputs,OUT,0);start=time.perf_counter();metric=validate()
        assert model_hash(model)==initial
        write(PUBLIC/'INITIAL_DEV.json',dict(passed=True,H_JADE=metric,metrics=validate.metrics,groups=validate.last_groups,
            completed_windows=590,DDIM_steps=20,K=20,selection=False,epoch_label='initial_zero_update_not_completed_training_epoch',
            seconds=time.perf_counter()-start,initial_model_sha256=initial,model_unchanged=True,
            frozen_file_sha256=file_sha(ROOTS['run']/'FROZEN_CONFIG.json'),implementation_hash=frozen['implementation_hash'],
            optimizer_guard=GUARD,real_optimizer_updates=0,synthetic_optimizer_updates=0))
        print('Full initial dev PASS H-JADE',metric,'updates 0',flush=True)
if __name__=='__main__':main()

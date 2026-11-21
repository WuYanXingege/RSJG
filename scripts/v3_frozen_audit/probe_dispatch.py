"""Reproduce the preflight numerical discrepancy with one strict-loaded model.

Not needed for the accepted audit. No training/optimizer/backward and no weight
values change. Only the requires_grad flag is toggled under native no_grad eval.
"""
import json
import torch
from replay import install_no_training_guard,GUARD
from bcep_v3.formal_inputs import FormalInputs,paths
from bcep_v3.formal_validation import window
from bcep_v3.formal import model_hash
from bcep_v3.model import build
from bcep_v2.runtime import configure_cuda
from bcep_joint.common import seed_all


def main():
    install_no_training_guard();configure_cuda();torch.set_num_threads(1);seed_all(3201)
    inputs=FormalInputs()
    state=torch.load(paths()['run']/'joint/epoch_012.pt',map_location='cpu',weights_only=False)
    model,_=build(inputs.config);model.load_state_dict(state['model'],strict=True);model.cuda();model.eval()
    before=model_hash(model)
    batch=inputs.data.batch([2681],'selection_dev');goals=inputs.banks[1].batch([2681],batch)
    rows=[]
    for flag in [True,False,False,True]:
        model.requires_grad_(flag)
        row,_=window(model,batch,goals);rows.append(dict(requires_grad=flag,**row))
    assert model_hash(model)==before
    print(json.dumps(dict(rows=rows,parameter_values_unchanged=True,guard=GUARD),indent=2))


if __name__=='__main__':main()

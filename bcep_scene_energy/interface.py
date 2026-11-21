"""Unchanged native objective/callback/state interfaces; no train command is enabled."""
from bcep_v2.objective import bridge_and_pair_loss


def callbacks(model,inputs,device):
    from bcep_v3.formal_inputs import FormalInputs
    from bcep_v2.stages import joint_callbacks
    if type(inputs) is not FormalInputs:raise TypeError('verified official ZARA2 inputs required')
    if inputs.config['allow_real_training'] is not False:raise PermissionError('no training authorization in this release')
    return joint_callbacks(model,inputs.data,device,inputs.config,*inputs.banks)


def train(*args,**kwargs):
    raise PermissionError('SSEC engineering acceptance only; real/synthetic optimizer updates forbidden')


def main():
    # Reject before loading data, constructing optimizers, or inspecting queues.
    raise PermissionError('No executable training entry in this zero-update release')


if __name__=='__main__':main()

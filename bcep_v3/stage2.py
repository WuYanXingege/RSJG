"""Runnable audit Stage2; cannot accept a real-data adapter in this release."""
from bcep_v2.stages import joint_callbacks


def callbacks(model, data, device, config, training, validation):
    if config.get('allow_real_training') is not False:
        raise PermissionError('v3 audit has no real-training authorization')
    # Exact fixture types prevent accidental use of official real data callbacks.
    from .fixtures import SyntheticData, SyntheticBank
    if type(data) is not SyntheticData or type(training) is not SyntheticBank or type(validation) is not SyntheticBank:
        raise PermissionError('audit entry accepts built-in synthetic fixtures only')
    if config.get('pair_objective') != 'final_coordinate':
        raise ValueError('old effective coordinate objective required')
    return joint_callbacks(model, data, device, config, training, validation)


def formal_callbacks(model, inputs, device):
    """Actual native grouped backward; calling this does not run an optimizer.

    Admission is via verified FormalInputs. Training permission is checked by
    the single-fold controller before run_stage, not inferred from this callback.
    """
    from .formal_inputs import FormalInputs
    if type(inputs) is not FormalInputs:
        raise TypeError('verified ZARA2 FormalInputs required')
    if inputs.config['pair_objective']!='final_coordinate':
        raise ValueError('final_coordinate required')
    return joint_callbacks(model,inputs.data,device,inputs.config,*inputs.banks)

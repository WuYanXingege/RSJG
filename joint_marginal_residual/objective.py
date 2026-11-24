"""Native hard errors with differentiable, correctly weighted batch reduction."""
import torch
from ebjd.metrics import hard_metric_statistics

METRICS = ('minADE','minFDE','JADE','JFDE')


def metric_sums(predictions,truth,valid):
    """One physical microbatch; compute trajectory distances only once.

    Errors and model remain FP32. Scalar accumulation uses FP64 just like the
    native MetricAccumulator; conversion remains differentiable for students.
    """
    if not valid.any(1).all():raise ValueError('empty windows are not admitted')
    st = hard_metric_statistics(predictions,truth,valid)
    numerators = torch.stack((st.agent_minade[st.agent_valid].double().sum(),
        st.agent_minfde[st.agent_valid].double().sum(),st.scene_jade.double().sum(),st.scene_jfde.double().sum()))
    return dict(numerators=numerators,agent_count=st.agent_valid.sum(),
                scene_count=numerators.new_tensor(st.scene_jade.numel(),dtype=torch.long))


def aggregate_metric_sums(parts, *, detach=False):
    """Reduce all physical microbatches of a logical batch before division."""
    if not parts:raise ValueError('no logical-batch parts')
    numerator = torch.stack([p['numerators'].detach() if detach else p['numerators'] for p in parts]).sum(0)
    agents = torch.stack([torch.as_tensor(p['agent_count'],device=numerator.device) for p in parts]).sum()
    scenes = torch.stack([torch.as_tensor(p['scene_count'],device=numerator.device) for p in parts]).sum()
    if bool(agents <= 0) or bool(scenes <= 0):raise ValueError('empty logical batch')
    denominator = torch.stack((agents,agents,scenes,scenes)).to(numerator)
    return dict(metrics=numerator/denominator,numerators=numerator,agent_count=agents,scene_count=scenes)


def paired_objective(student_parts,teacher_parts,scales):
    """Mean plus max of four normalized *logical-batch* paired changes."""
    student = aggregate_metric_sums(student_parts)
    teacher = aggregate_metric_sums(teacher_parts,detach=True)
    if any(not torch.equal(student[key],teacher[key].to(student[key].device)) for key in ('agent_count','scene_count')):
        raise ValueError('student and teacher logical populations differ')
    if isinstance(scales,dict):scales = [scales[k] for k in METRICS]
    scales = torch.as_tensor(scales,device=student['metrics'].device,dtype=student['metrics'].dtype).detach().clamp_min(1e-6)
    if scales.shape != (4,) or not torch.isfinite(scales).all():raise ValueError('four finite train-only scales required')
    relative = (student['metrics']-teacher['metrics'].to(student['metrics']))/scales
    return dict(loss=relative.mean()+relative.amax(),relative=relative,
                student=student['metrics'],teacher=teacher['metrics'],
                agent_count=student['agent_count'],scene_count=student['scene_count'])


def scales_from_train_metric_sums(teacher_parts):
    return aggregate_metric_sums(teacher_parts,detach=True)['metrics'].detach().clamp_min(1e-6)

"""One frozen six-condition dev selector; exact strictness, no tolerance search."""
import math

CONDITIONS = (
    ('all','minFDE',-0.020,False),('all','JFDE',-0.005,False),
    ('all','minADE',0.,True),('all','JADE',0.,True),
    ('N_ge_9','JFDE',0.,False),('N_ge_9','minFDE',0.,False),
)


def evaluate_gate(base_groups,student_groups):
    checks=[]
    for group,metric,threshold,strict in CONDITIONS:
        baseline=base_groups.get(group);student=student_groups.get(group)
        if baseline is None or baseline.get('scene_count',0)==0:
            if group=='all':raise ValueError('full development group cannot be empty')
            assert student is None or student.get('scene_count',0)==0
            checks.append(dict(group=group,metric=metric,available=False,passed=None,status='N/A'))
            continue
        if student is None:raise ValueError('student group missing')
        for count in ('scene_count','agent_count'):
            if student[count]!=baseline[count]:raise ValueError('paired development populations changed')
        b,s=float(baseline[metric]),float(student[metric])
        if not math.isfinite(b) or not math.isfinite(s):raise ValueError('nonfinite dev metric')
        delta=s-b;margin=delta-threshold
        passed=delta<threshold if strict else delta<=threshold
        checks.append(dict(group=group,metric=metric,available=True,B=b,student=s,
            difference_m=delta,threshold_m=threshold,strict=strict,passed=passed,
            margin_m=margin,violation_m=max(0.,margin)))
    available=[c for c in checks if c['available']]
    maximum=max(c['violation_m'] for c in available)
    hardest=max(c['margin_m'] for c in available)
    return dict(passed=all(c['passed'] for c in available),checks=checks,
        max_violation_m=maximum,hardest_conditions=[f"{c['group']}.{c['metric']}" for c in available if c['margin_m']==hardest])


def rank_epoch(record):
    """Official selector, also used for distinct complete-epoch Top5 retention."""
    epoch=int(record['epoch']);metric=record['metrics']['all'];gate=record['gate']
    if gate['passed']:
        return (0,float(metric['minFDE']),float(metric['JFDE']),epoch)
    return (1,float(gate['max_violation_m']),float(metric['minFDE']),epoch)


def select_epoch(records):
    if not records:raise ValueError('no completed epochs')
    epochs=[int(row['epoch']) for row in records]
    if len(set(epochs))!=len(epochs):raise ValueError('duplicate completed epoch')
    return min(records,key=rank_epoch)

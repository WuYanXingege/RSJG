"""Original message-forward frame + parent-module boundaries only; no arithmetic replacement."""
import inspect
import sys
from tools.jdv2_restart_observer import Recorder, compare_tensor


class MessageObserver:
    def __init__(self, layer):
        self.layer = layer
        self.recorder = Recorder()
        self.handles = []
        self.message_calls = 0
        self.active_message = None
        self.previous_trace = None
        self.code = layer.forward.__func__.__code__
        source, start = inspect.getsourcelines(layer.forward)
        anchors = {'weighted': 'aggregated = torch.zeros_like(agent_feat)',
                   'raw': 'normalizer = agent_feat.new_zeros',
                   'normalizer': 'aggregated = aggregated / normalizer.clamp_min',
                   'normalized': 'update = self.update_mlp'}
        self.lines = {}
        for name, anchor in anchors.items():
            found = [start + i for i, line in enumerate(source) if anchor in line]
            if not found:
                raise RuntimeError('Missing verified frame anchor: ' + anchor)
            self.lines[found[-1]] = name

    def take(self, name, value):
        self.recorder.take(name, value)
        self.recorder.events.append({'capture': name})

    def trace(self, frame, event, arg):
        if frame.f_code is not self.code:
            return None
        if event == 'call':
            self.take('entry', {k: frame.f_locals[k] for k in ('agent_feat','edge_index','edge_feat','edge_weight')})
        elif event == 'line':
            stage = self.lines.get(frame.f_lineno)
            values = frame.f_locals
            if stage == 'weighted':
                self.take('weighted/source_index', values['source'])
                self.take('weighted/target_index', values['target'])
                self.take('weighted/source', values['message_to_source'])
                self.take('weighted/target', values['message_to_target'])
            elif stage == 'raw':
                self.take('feature_accumulation/raw_after_both_adds', values['aggregated'])
            elif stage == 'normalizer':
                self.take('normalizer/after_both_adds', values['normalizer'])
            elif stage == 'normalized':
                self.take('normalized/aggregated', values['aggregated'])
        elif event == 'return' and arg is not None:
            self.take('message/final_return', arg)
        return self.trace

    def before_message(self, module, args):
        sides = ('to_source', 'to_target')
        if self.message_calls >= 2:
            raise RuntimeError('Unexpected message_mlp call count')
        self.active_message = f'message_mlp/{self.message_calls}_{sides[self.message_calls]}'
        self.message_calls += 1
        self.take(self.active_message + '/input', args[0])

    def after_message(self, module, args, output):
        self.take(self.active_message + '/output', output)

    def pre(self, name):
        def hook(module, args):
            self.take(name + '/input', args[0])
        return hook

    def post(self, name):
        def hook(module, args, output):
            self.take(name + '/output', output)
        return hook

    def __enter__(self):
        if sys.gettrace() is not None or sys.getprofile() is not None:
            raise RuntimeError('Refuse existing Python observation state')
        if any(m._forward_hooks or m._forward_pre_hooks for m in self.layer.modules()):
            raise RuntimeError('Refuse existing module hooks')
        self.previous_trace = sys.gettrace()
        try:
            self.handles += [self.layer.message_mlp.register_forward_pre_hook(self.before_message),
                             self.layer.message_mlp.register_forward_hook(self.after_message)]
            for name in ('update_mlp', 'norm'):
                module = getattr(self.layer, name)
                self.handles += [module.register_forward_pre_hook(self.pre(name)), module.register_forward_hook(self.post(name))]
            sys.settrace(self.trace)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous_trace)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def finish(self):
        data, meta = self.recorder.finish()
        meta['observer_cost'] = 'Parent Sequential/update/norm hooks; Python line observer restricted to original forward code frame; retained GPU references. No clone/cpu/item/sync/GPU arithmetic in hot observation.'
        meta['line_anchors'] = self.lines
        meta['first_add_snapshot_captured'] = False
        meta['linear_pre_inplace_ReLU_snapshot_captured'] = False
        return data, meta


def compare_traces(a, b, ma, mb):
    """Gate only the first chronological valid mismatch; do not pick a preferred operator."""
    valid = not ma['invalid_snapshots'] and not mb['invalid_snapshots']
    keys_equal = list(a) == list(b)
    layout_fields = ('dtype','device','shape','stride','storage_offset','alias_group')
    layouts = {k:all(ma['tensors'][k][f] == mb['tensors'][k][f] for f in layout_fields) for k in a if k in b}
    rows = {k:compare_tensor(a[k], b[k]) for k in a if k in b}
    first = next((k for k,v in rows.items() if not v['bitwise_equal']),None)
    result = {'valid_snapshots': bool(valid), 'ordered_keys_equal':keys_equal,
              'event_order_equal':ma['events']==mb['events'], 'layouts_equal':layouts,
              'comparisons':rows,'first_difference':first,'eligible_region':None,
              'gate_pass':False,'reason':'No eligible first valid differing region'}
    if not(valid and keys_equal and all(layouts.values()) and result['event_order_equal']):
        result['reason']='Invalid snapshot, alignment or layout'
        return result
    def same(names):
        return all(name in rows and rows[name]['bitwise_equal'] for name in names)
    regions = {
        'feature_accumulation/raw_after_both_adds':('feature_accumulation', ['entry/agent_feat','weighted/source_index','weighted/target_index','weighted/source','weighted/target']),
        'normalizer/after_both_adds':('normalizer', ['entry/edge_weight','weighted/source_index','weighted/target_index']),
        'message_mlp/0_to_source/output':('message_mlp/0_to_source', ['message_mlp/0_to_source/input']),
        'message_mlp/1_to_target/output':('message_mlp/1_to_target', ['message_mlp/1_to_target/input']),
        'update_mlp/output':('update_mlp',['update_mlp/input']),
        'norm/output':('norm',['norm/input'])}
    if first in regions:
        region, names = regions[first]
        if same(names):
            result.update(gate_pass=True, eligible_region=region, region_inputs=names,
                          reason='First chronological valid region has bitwise identical captured entrance tensors and differing output; external model/settings/RNG/semantic gates also required')
    return result

"""Scoped function-boundary observer; never substitutes production arithmetic.

References are held until successful sampling. Version changes invalidate a
capture rather than silently presenting a later value as an earlier snapshot.
No tensor clone, CPU transfer, or GPU arithmetic is performed in hot hooks.
Holding references changes allocator lifetimes: GPU neutrality is NOT assumed.
"""
from __future__ import annotations
import contextlib
import dataclasses
import hashlib
import io
import sys
from collections import Counter
import torch


def raw_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def compare_tensor(a, b):
    result = {"dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
              "shape_a": list(a.shape), "shape_b": list(b.shape),
              "dtype_equal": a.dtype == b.dtype,
              "shape_equal": a.shape == b.shape,
              "raw_sha256_a": hashlib.sha256(raw_bytes(a)).hexdigest(),
              "raw_sha256_b": hashlib.sha256(raw_bytes(b)).hexdigest()}
    result["bitwise_equal"] = (result["dtype_equal"] and result["shape_equal"] and
                                result["raw_sha256_a"] == result["raw_sha256_b"])
    if a.shape == b.shape:
        x, y = a.detach().cpu(), b.detach().cpu()
        different = x != y
        result["different_elements"] = int(different.sum())
        result["numerically_equal"] = not bool(different.any())
        if different.any():
            coordinate = tuple(different.nonzero()[0].tolist())
            result.update(first_coordinate=list(coordinate), first_a=str(x[coordinate].item()),
                          first_b=str(y[coordinate].item()))
            delta = (x.double() - y.double()).abs()
            result["max_abs"] = float(delta[~torch.isnan(delta)].max()) if (~torch.isnan(delta)).any() else None
    return result


class Recorder:
    def __init__(self):
        self.tensors = {}
        self.metadata = {}
        self.values = {}
        self.events = []
        self.storage_ids = {}

    def take(self, name, value):
        if torch.is_tensor(value):
            if name in self.tensors:
                raise RuntimeError('duplicate trace key: ' + name)
            storage = value.untyped_storage()._cdata
            alias = self.storage_ids.setdefault(storage, len(self.storage_ids))
            self.tensors[name] = value
            self.metadata[name] = {"dtype": str(value.dtype), "device": str(value.device),
                "shape": list(value.shape), "stride": list(value.stride()),
                "storage_offset": value.storage_offset(), "alias_group": alias,
                "version_at_capture": value._version, "lifetime": "held_until_deferred_dump"}
        elif dataclasses.is_dataclass(value):
            self.take(name, dataclasses.asdict(value))
        elif isinstance(value, dict):
            for key, item in value.items():
                self.take(name + '/' + str(key), item)
        elif isinstance(value, (tuple, list)):
            # Exact priorities/objectives retain arbitrary precision, as decimal strings.
            if all(isinstance(x, (int, bool)) for x in value):
                self.values[name] = [str(x) if isinstance(x, int) and not isinstance(x, bool) else x for x in value]
            else:
                for index, item in enumerate(value):
                    self.take(name + '/' + str(index), item)
        elif isinstance(value, int) and not isinstance(value, bool):
            self.values[name] = str(value)
        elif isinstance(value, (float, str, bool)) or value is None:
            self.values[name] = value

    def finish(self):
        cpu = {}
        invalid = []
        for name, value in self.tensors.items():
            row = self.metadata[name]
            row['version_at_dump'] = value._version
            row['valid_snapshot'] = value._version == row['version_at_capture']
            if not row['valid_snapshot']:
                invalid.append(name)
            cpu[name] = value.detach().cpu().clone()
            row['raw_sha256'] = hashlib.sha256(raw_bytes(cpu[name])).hexdigest()
        # Verify native dtype/raw-bit serialization, including BF16, not FP32 widening.
        stream = io.BytesIO()
        torch.save(cpu, stream)
        stream.seek(0)
        reloaded = torch.load(stream, weights_only=True, map_location='cpu')
        assert all(compare_tensor(cpu[k], reloaded[k])['bitwise_equal'] for k in cpu)
        return cpu, {'tensors': self.metadata, 'values': self.values, 'events': self.events,
                     'invalid_snapshots': invalid, 'native_dtype_roundtrip': True,
                     'observer_extra_gpu_arithmetic': 0,
                     'observer_cost': 'Python profile/wrapper overhead, generator get_state, retained tensor lifetimes; deferred CPU copies and serialization after forward'}


class BoundaryObserver:
    def __init__(self, net=None, enabled=False, extra_functions=()):
        from src.models.joint_dependency_v2 import joint_sampler as sampler
        from src.models.joint_dependency_v2 import exact_lexicographic_assignment as exact
        from src.models.joint_dependency_v2.dynamic_relation import DynamicHypothesisRelation as Relation
        from src.models.joint_dependency_v2.joint_energy import RelationSpecificJointEnergy as Energy
        functions = [sampler._unit_gumbel, sampler.weighted_gumbel_top_p,
                     sampler.ParallelConditionalSampler.forward,
                     sampler.ParallelConditionalSampler._resolve_sampling_generators,
                     sampler.ParallelConditionalSampler._emit_diagnostic,
                     exact.solve_exact_persistent_tie, Relation.selected_neighbor_relation,
                     Relation.selected_joint_relation, Energy.selected_effective_energy]
        if net is not None:
            functions += [net.social_encoder.forward, net.relation_inference.forward,
                          net.jdv2_unary.forward, net._jdv2_goal_outputs]
        functions += list(extra_functions)
        self.codes = {f.__code__: f.__qualname__ for f in functions}
        self.enabled = enabled
        self.recorder = Recorder()
        self.counts = Counter()
        self.frames = {}
        self.generators = {}
        self.original_random = {}

    def profile(self, frame, event, arg):
        if frame.f_code not in self.codes or event not in ('call', 'return'):
            return
        label = self.codes[frame.f_code]
        if event == 'call':
            index = self.counts[label]
            self.counts[label] += 1
            key = f'{label}/{index:03d}'
            self.frames[id(frame)] = key
            names = frame.f_code.co_varnames[:frame.f_code.co_argcount + frame.f_code.co_kwonlyargcount]
            self.recorder.take(key + '/input', {k: frame.f_locals[k] for k in names if k != 'self' and k in frame.f_locals})
            if label.endswith('_emit_diagnostic'):
                parent = frame.f_back.f_locals
                if frame.f_locals['round_index'] > 0:
                    self.recorder.take(key + '/accumulated_after_both_index_add', parent['accumulated'])
        else:
            key = self.frames.pop(id(frame))
            self.recorder.take(key + '/output', arg)
            if label.endswith('weighted_gumbel_top_p'):
                self.recorder.take(key + '/actual', {k: frame.f_locals[k] for k in ('logits', 'gumbel_noise', 'perturbed', 'local_mask')})
            if label.endswith('_unit_gumbel'):
                self.recorder.take(key + '/actual_clamped_uniform', frame.f_locals['uniform'])
            if label.endswith('_resolve_sampling_generators'):
                self.generators = arg
                self.recorder.take('explicit_generators/before', {k: g.get_state() for k, g in arg.items()})
            if label == 'ParallelConditionalSampler.forward':
                self.recorder.take('explicit_generators/after', {k: g.get_state() for k, g in self.generators.items()})
        self.recorder.events.append({'key': key, 'event': event})

    def __enter__(self):
        if not self.enabled:
            return self
        if sys.getprofile() is not None:
            raise RuntimeError('existing profiler: refuse replacement')
        # Capture actual production draws without any new random consumption.
        for name in ('randn', 'randn_like'):
            original = getattr(torch, name)
            self.original_random[name] = original
            def wrapped(*args, _name=name, _fn=original, **kwargs):
                result = _fn(*args, **kwargs)
                index = self.counts['noise']
                self.counts['noise'] += 1
                self.recorder.take(f'diffusion_draw/{index:03d}/{_name}', result)
                return result
            setattr(torch, name, wrapped)
        sys.setprofile(self.profile)
        return self

    def __exit__(self, *exc):
        if self.enabled:
            sys.setprofile(None)
            for name, original in self.original_random.items():
                setattr(torch, name, original)
        return False


def state_fingerprint(net):
    return {name: {'dtype': str(value.dtype), 'shape': list(value.shape),
                   'sha256': hashlib.sha256(raw_bytes(value)).hexdigest(),
                   'version': value._version}
            for name, value in net.state_dict(keep_vars=True).items()}

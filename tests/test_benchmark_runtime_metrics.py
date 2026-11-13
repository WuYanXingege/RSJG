import torch
import pytest

from scripts.benchmark_runtime_metrics import UnifiedMetricBuffer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_deferred_preserves_original_python_summation_and_detaches(dtype):
    original = UnifiedMetricBuffer(False)
    deferred = UnifiedMetricBuffer(True)
    for count in (4, 3):
        for buffer in (original, deferred):
            buffer.begin_microbatch()
            for index in range(count):
                total = torch.tensor(1.123456789 + index, dtype=dtype, requires_grad=True)
                buffer.add(total, total * .25, total * .6, total * .8,
                           (index + 1) / 10, 1 / count)
    assert all(not value.requires_grad and value.grad_fn is None
               for group in deferred.groups for row, _, _ in group for value in row)
    assert deferred.flush() == original.flush()
    assert deferred.flush() == []


def test_lifecycle_rejects_missing_or_empty_microbatch():
    buffer = UnifiedMetricBuffer(True)
    with pytest.raises(RuntimeError, match="begin_microbatch"):
        buffer.add(*(torch.tensor(1.) for _ in range(4)), 1., 1.)
    buffer.begin_microbatch()
    with pytest.raises(RuntimeError, match="empty"):
        buffer.flush()

"""P3 benchmark-only switches; never modify the frozen formal defaults."""
from contextlib import contextmanager


@contextmanager
def checkpoint_arm(model, *, map_checkpoint=None, bridge_checkpoint=None):
    modules = (model.bridge.encoder,model.bridge.denoiser)
    prior = [m.activation_checkpointing for m in modules]
    try:
        for m,value in zip(modules,(map_checkpoint,bridge_checkpoint)):
            if value is not None:m.activation_checkpointing = bool(value)
        yield
    finally:
        for m,value in zip(modules,prior):m.activation_checkpointing = value

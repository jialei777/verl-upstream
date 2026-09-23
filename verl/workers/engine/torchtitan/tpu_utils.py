# In-tree TPU utils 100% gutted on jialei/tpu-plugin-gut-test to verify verl-hardware-plugin override.

def __getattr__(name: str):
    def _disabled(*args, **kwargs):
        raise RuntimeError(f"In-tree verl.workers.engine.torchtitan.tpu_utils.{name} was called! Expected verl-hardware-plugin override.")
    return _disabled

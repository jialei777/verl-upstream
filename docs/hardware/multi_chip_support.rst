Multi-Chip Support
==================

Last updated: 09/22/2026.

Overview
--------

verl supports RL training across multiple hardware platforms through a unified
plugin system. The architecture consists of two main subsystems:

1. **Platform Plugin System** (``verl.plugin.platform``) — A hardware
   abstraction layer with auto-detection and a unified device API.
2. **Engine Plugin System** (``verl.workers.engine.base``) — Training engine
   extensions that add chip-specific optimizations on top of existing
   FSDP/Megatron engines.

Hardware Support
----------------

**Built-in (verl core):**

- NVIDIA GPU (CUDA)
- Huawei Ascend NPU

**Via verl-hardware-plugin (reference implementations):**

Other hardware platforms are supported through the external
`verl-hardware-plugin <https://github.com/verl-project/verl-hardware-plugin>`_
package, which provides reference implementations for vendors to adapt:

- Intel XPU (Data Center GPU Max / Arc)
- Cambricon MLU (MLU370 / MLU590)
- MetaX (CUDA-compatible)
- Google TPU (v6e), see :ref:`tpu-support`

.. note::

   The implementations in verl-hardware-plugin are **examples only**. Full
   production support requires collaboration with the respective hardware
   vendors. Vendors can use these as templates to build and maintain their own
   plugins.

.. _platform-extension-points:

Platform Extension Points
-------------------------

Beyond the device API, ``PlatformBase`` exposes a few optional hooks that verl core calls at
well-defined points. All of them have a default that preserves the CUDA behaviour, so a platform
only implements the ones it needs.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Hook
     - Called by / purpose
   * - ``ray_local_rank_override()``
     - ``Worker._configure_before_init``. Return the local rank when Ray does not enumerate the
       accelerator itself. Default ``None`` (use Ray's accelerator ids).
   * - ``supports_colocated_worker_groups()``
     - ``RayResourcePool`` / ``ResourcePoolManager``. Return ``False`` when a device cannot be
       shared by two WorkerGroups; verl then caps ``max_colocate_count`` at 1. Default ``True``.
   * - ``get_worker_env_vars(...)``
     - ``RayWorkerGroup``. Extra env vars for a worker's ``runtime_env``, built once the
       placement groups exist (mesh addresses, device index, ...). Default ``{}``.
   * - ``get_ray_init_kwargs()``
     - The training entrypoints. Extra kwargs merged into ``ray.init()``, e.g. a
       ``runtime_env.worker_process_setup_hook``. Default ``{}``.
   * - ``requires_remote_driver()``
     - The training entrypoints. Return ``True`` when the trainer has to run inside a Ray actor
       because the driver node owns no device. Default ``False``.
   * - ``supports_eager_collectives()``
     - ``TrainingWorker`` metric reduction. Return ``False`` for backends that only run
       collectives inside a traced graph; verl then merges the per-rank metrics on the driver
       instead of all-gathering them. Default ``True``.

.. _tpu-support:

Google TPU
----------

TPU support has two halves: the platform itself (device API, Ray resources, slice environment)
lives in `verl-hardware-plugin <https://github.com/verl-project/verl-hardware-plugin>`_, and the
training path in verl core adapts to the XLA execution model:

- The TorchTitan engine is registered for ``device="tpu"``, uses the ``tpu`` compile backend and
  the non-fused optimizer and gradient-clipping paths.
- Packed sequences are padded to a static bucket (``VERL_TPU_SEQ_BUCKET_SIZE``, default 256)
  before they reach the device, so XLA compiles a bounded set of shapes per run instead of one
  per micro batch.
- ``VarlenAttention`` is routed through ``scaled_dot_product_attention``, and the few operators
  that ``torch_tpu`` does not implement (``unique_consecutive``, the chunked log-prob kernel)
  fall back to portable equivalents.
- Metrics are reduced on the driver, since the device backend cannot issue a collective outside
  its traced graph.

Use pure FSDP2 (``tensor_parallel_size=1``): with tensor parallelism the DTensor backward path
intermittently produces non-finite gradients, which the optimizer step silently skips.

See ``examples/tpu/sft/README.md`` for a runnable SFT example and the tuning notes.

Design Principles
-----------------

1. **Plugin Architecture**: Platform backends and engine extensions register
   via decorator-based registries (``PlatformRegistry``, ``EngineRegistry``),
   requiring no modifications to verl core code.

2. **Auto-Detection + Manual Override**: The platform auto-detects hardware
   type by probing ``is_available(use_smi_check=True)`` on each registered
   platform. Can be explicitly overridden via the ``VERL_PLATFORM`` environment
   variable.

3. **Two-Dimensional Engine Lookup**: Engines register with both ``device``
   (torch device type) and ``vendor`` (hardware vendor). Lookup priority:

   - Exact match ``(device, vendor)`` — vendor-specific engine
   - Fallback to device-only key — base engine for that device type
   - For CUDA-compatible devices, fallback to base CUDA engine

4. **Backward Compatibility**: The legacy ``verl.utils.device`` API is
   preserved as a thin wrapper over the platform plugin system. Existing code
   continues to work without modification.

Architecture Overview
---------------------

::

    +-------------------------------------------------------------------+
    |                  verl Multi-Chip Architecture                      |
    +-------------------------------------------------------------------+
    |                                                                    |
    |  +---------------------------------------------------------+      |
    |  |              Platform Plugin System                      |      |
    |  |            (verl.plugin.platform)                        |      |
    |  |                                                          |      |
    |  |  PlatformRegistry                                        |      |
    |  |    ├─ "nvidia"    → PlatformCUDA      (built-in)         |      |
    |  |    ├─ "huawei"    → PlatformNPU       (built-in)         |      |
    |  |    ├─ "intel"     → PlatformXPU       (plugin)           |      |
    |  |    ├─ "cambricon" → PlatformMLU       (plugin)           |      |
    |  |    └─ "metax"     → PlatformMetaX     (plugin)           |      |
    |  |                                                          |      |
    |  +---------------------------------------------------------+      |
    |                                                                    |
    |  +---------------------------------------------------------+      |
    |  |              Engine Plugin System                        |      |
    |  |            (verl.workers.engine.base)                    |      |
    |  |                                                          |      |
    |  |  EngineRegistry  (device, vendor) → Engine class         |      |
    |  |       |                                                  |      |
    |  |       +-- ("cuda", None)     → FSDPEngineWithLMHead      |      |
    |  |       +-- ("npu", None)      → FSDPNPUEngineWithLMHead   |      |
    |  |       +-- ("cuda", "metax")  → FSDPMetaXEngineWithLMHead |      |
    |  |       +-- ("xpu", "intel")   → FSDPXPUEngineWithLMHead   |      |
    |  |       +-- ("mlu","cambricon")→ FSDPMLUEngineWithLMHead   |      |
    |  |                                                          |      |
    |  +---------------------------------------------------------+      |
    |                                                                    |
    +-------------------------------------------------------------------+

Plugin Loading
--------------

verl discovers plugins through two mechanisms:

1. **setuptools entry_points** (``verl.plugins`` group) — standard Python
   packaging mechanism. After ``pip install``, the plugin is auto-discovered.

2. **``VERL_USE_EXTERNAL_MODULES``** environment variable — for development
   or non-packaged plugins:

   .. code-block:: bash

      export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin

Platform Registration
---------------------

Each platform class registers via decorator:

.. code-block:: python

   @PlatformRegistry.register(platform="my_vendor")
   class PlatformMyDevice(PlatformBase):
       @property
       def device_name(self) -> str:
           return "my_device"  # torch device type

       @property
       def vendor_name(self) -> str:
           return "my_vendor"  # used for engine lookup

Platform selection priority:

1. ``VERL_PLATFORM`` environment variable (explicit override)
2. Auto-detection via ``is_available(use_smi_check=True)``
3. Fallback to ``"nvidia"``

Engine Registration
-------------------

Engine classes register with device and vendor:

.. code-block:: python

   @EngineRegistry.register(
       model_type="language_model",
       backend=["fsdp", "fsdp2"],
       device="cuda",           # torch device type
       vendor="my_vendor",      # vendor name
   )
   class FSDPMyVendorEngineWithLMHead(FSDPEngineWithLMHead):
       def initialize(self):
           super().initialize()
           # vendor-specific initialization

Engine lookup calls ``get_device_name()`` and ``get_vendor()`` from the active
platform, then resolves the engine by ``(device_name, vendor_name)`` key.

Environment variable overrides for engine selection:

- ``VERL_ENGINE_DEVICE`` — override detected device name
- ``VERL_ENGINE_VENDOR`` — override detected vendor name

Adding New Hardware
-------------------

For a step-by-step guide on adding support for a new hardware platform, see
the `verl-hardware-plugin Development Guide <https://github.com/verl-project/verl-hardware-plugin/blob/main/docs/development.md>`_.

The core platform and engine registry mechanism is implemented in
`PR #6086 <https://github.com/verl-project/verl/pull/6086>`_.

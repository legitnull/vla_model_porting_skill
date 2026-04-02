## Troubleshooting

Issues that may occur depending on the source repo, transformers version, or server environment. Not every issue will happen every time — check this section when you hit a problem during Steps 3e–3i or A7–A8.

### T1: flash-attn `undefined symbol` error

**Symptom:**
```
ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
```

**Cause:** Pre-built flash-attn wheel was compiled against a different PyTorch version than what's installed.

**Fix:** Rebuild from source to match the current PyTorch:
```bash
pip uninstall flash-attn -y
pip install --no-build-isolation flash-attn==2.6.3
```
Build takes ~10 minutes. Check [flash-attn releases](https://github.com/Dao-AILab/flash-attention/releases) for prebuilt wheels matching your exact PyTorch+CUDA+Python combo first.

### T2: `Tensor.item() cannot be called on meta tensors` (transformers 5.x)

**Symptom:**
```
RuntimeError: Tensor.item() cannot be called on meta tensors
```
during `from_pretrained()` model initialization.

**Cause:** transformers ≥5.0 initializes models on meta device during `from_pretrained()` (via `torch.device("meta")` context manager in `get_init_context()`). Any code in `__init__` that creates non-parameter objects involving tensor validation — such as `torch.distributions.Beta()`, `torch.distributions.Normal()`, or any `.item()` call — will fail.

**Fix:** Defer the problematic object creation to a lazy property or to the first `forward()` call. Example:
```python
# Before (fails on meta device):
self.beta_dist = Beta(config.alpha, config.beta)

# After (lazy creation):
self._alpha = config.alpha
self._beta = config.beta
self._beta_dist = None

@property
def beta_dist(self):
    if self._beta_dist is None:
        self._beta_dist = Beta(self._alpha, self._beta)
    return self._beta_dist
```

### T3: `'GR00TN15' object has no attribute 'all_tied_weights_keys'` (transformers 5.x)

**Symptom:**
```
AttributeError: 'ModelClass' object has no attribute 'all_tied_weights_keys'
```
during `from_pretrained()` weight loading.

**Cause:** transformers ≥5.0 requires `post_init()` to be called at the end of `PreTrainedModel.__init__()` to set up `all_tied_weights_keys` and other metadata. Models written for older transformers may not call it.

**Fix:** Add `self.post_init()` at the end of the model's `__init__`:
```python
def __init__(self, config):
    super().__init__(config)
    # ... all module creation ...
    self.post_init()
```

### T4: Image processor returns lists instead of tensors (transformers 5.x)

**Symptom:**
```
AttributeError: 'list' object has no attribute 'shape'
```
when accessing `image_inputs["pixel_values"].shape`.

**Cause:** In transformers ≥5.0, image processors return `pixel_values` as a Python list by default instead of a stacked tensor. Code that calls `.shape` on the result will fail.

**Fix:** Pass `return_tensors="pt"` to the image processor call:
```python
image_inputs = self.image_processor(
    images=[image],
    return_tensors="pt",
    **kwargs,
)
```

### T5: Vendored processor files — cache chain overwrites patches

**Symptom:** You patch a processor `.py` file but the fix doesn't take effect, or the error still points to an unpatched copy.

**Cause:** Some models (e.g., GR00T with Eagle2) have a 3-level cache chain:
1. **Vendored source** (e.g., `flagscale/models/vla/{model}/eagle2_hg_model/`) — the authoritative copy in the codebase
2. **Data cache** (e.g., `$HF_LEROBOT_HOME/lerobot/eagle2hg-processor-groot-n1p5/`) — `copytree`'d from (1) by `ensure_eagle_cache_ready()` during model init
3. **Transformers module cache** (e.g., `$HF_HOME/modules/transformers_modules/...`) — re-created from (2) by `get_class_from_dynamic_module()` inside `AutoProcessor.from_pretrained(..., trust_remote_code=True)`

Patches to (2) or (3) get overwritten. But more critically, (1) → (2) only works if both use the **same `HF_LEROBOT_HOME`**. If the model init code hardcodes `Path.home() / ".cache" / "lerobot"` while the processor code reads from `os.getenv("HF_LEROBOT_HOME")` (e.g., `/share/project/.../datasets`), the vendored files get copied to the wrong location and the processor loads stale unpatched files from the real `HF_LEROBOT_HOME`.

**Fix:**
1. Always patch the vendored source at (1) — never patch (2) or (3) directly.
2. Ensure both the model init and processor code resolve `HF_LEROBOT_HOME` from the same source (e.g., import from a shared `constants.py` that reads the env var).
3. In `_build_eagle_processor()` (or equivalent), clear the stale transformers module cache before calling `from_pretrained` so transformers re-copies from the patched data cache:
```python
from transformers.dynamic_module_utils import HF_MODULES_CACHE
stale = Path(HF_MODULES_CACHE) / "transformers_modules" / sanitized_repo_name
if stale.exists():
    shutil.rmtree(stale)
```
4. Pass `local_files_only=True` to `from_pretrained` to prevent downloading from HF hub.
5. After patching vendored files, **delete `__pycache__/` directories** at every level of the cache chain. Stale `.pyc` files will be loaded instead of the patched `.py` source. Either delete manually or add `shutil.rmtree(pycache)` in the cache refresh logic.

### T8: Vendored HF model code incompatible with FlagScale's transformers version

**Symptom:** `ImportError: cannot import name 'ImagesKwargs'` or `KeyError: 'do_convert_rgb'` / `KeyError: 'resample'` from a vendored image processor.

**Cause:** The source repo's vendored HF model code (e.g., `image_processing_eagle2_5_vl_fast.py`) was written for a newer transformers version (e.g., 5.x) that has `ImagesKwargs` in `image_processing_utils_fast`. FlagScale's env uses an older transformers (e.g., 4.57.6) where this class doesn't exist.

Even if you add a fallback `class ImagesKwargs(TypedDict): pass`, this creates an empty TypedDict that lacks base annotations (`do_convert_rgb`, `resample`, `device`, etc.). The `preprocess()` method's `for kwarg_name in self.valid_kwargs.__annotations__` loop won't populate these, causing `KeyError` when later code tries to `kwargs.pop("do_convert_rgb")`.

**Fix:** In the vendored file, use `DefaultFastImageProcessorKwargs` (available in transformers 4.x) as the fallback. This has all the base image processing annotations:
```python
try:
    from transformers.image_processing_utils_fast import ImagesKwargs
except ImportError:
    from transformers.image_processing_utils_fast import DefaultFastImageProcessorKwargs as ImagesKwargs
```

### T9: `flagscale` pip editable install points to wrong repo

**Symptom:** Generated torchrun scripts have `cd /wrong/path` and `PYTHONPATH=/wrong/path`, causing `can't open file` errors.

**Cause:** `flagscale` was pip-installed as editable (`pip install -e .`) from a different directory (e.g., `fs3`). The runner's `get_pkg_dir()` uses `os.path.abspath(__file__)` to find the repo root, which resolves to the editable install location, not the current working directory.

**Fix:** Prepend the correct repo to `PYTHONPATH` before running:
```bash
export PYTHONPATH="/path/to/correct/FlagScale:${PYTHONPATH}"
python flagscale/run.py --config-path=... --config-name=... action=run
```

### T10: `attn_mask dtype long int` in attention

**Symptom:**
```
RuntimeError: Expected attn_mask dtype to be bool or float or to match query dtype,
but got attn_mask.dtype: long int and query.dtype: c10::BFloat16
```

**Cause:** An attention mask from the VLM backbone (e.g., `backbone_attention_mask`) is `torch.long` (0/1 integers). When passed to `F.scaled_dot_product_attention`, it requires `bool`, `float`, or matching query dtype.

**Fix:** Check the source repo to see if the mask is actually used in attention. In many cases (e.g., lerobot's GR00T), the mask is passed to the DiT's `forward()` signature but the DiT blocks receive `encoder_attention_mask=None` — the mask is effectively unused. Match the source behavior: if the source passes `None`, pass `None` from the caller rather than modifying shared attention code.

### T11: Model-specific preprocessor step missing runtime overrides (normalization stats)

**Symptom:** Preprocessed batch values differ between source and FlagScale even though raw batches are identical. Specifically, normalized fields (action, state) show max_diff > 0 while non-normalized fields (eagle_pixel_values, input_ids, attention_mask) match exactly. Padded dimensions (zeros) also match — only the real data dimensions differ.

**Cause:** Model-specific preprocessor steps (e.g., `groot_pack_inputs`) may accept optional runtime parameters like `stats: dict | None = None` for min-max normalization. These stats come from `dataset.meta.stats` and must be injected at runtime via `preprocessor_overrides` — they are NOT part of the YAML config (which only has static parameters like `normalize_min_max: true`).

In the training script, the `preprocessor_overrides` dict may only pass stats to generic steps (e.g., `normalizer_processor`, `device_processor`) but forget the model-specific step. The model-specific step then runs with `stats=None` and skips normalization entirely (e.g., `if self.stats is None: return x`).

In lerobot, this is handled in `make_pre_post_processors()` which explicitly overrides `groot_pack_inputs_v3` with `{"stats": dataset_stats, "normalize_min_max": True}`.

**Fix:** Add the model-specific step to `preprocessor_overrides` in the training script:
```python
preprocessor_overrides = {
    "device_processor": {"device": device.type},
    "normalizer_processor": {"stats": dataset.meta.stats, ...},
    # Model-specific step needs stats too!
    "groot_pack_inputs": {
        "stats": dataset.meta.stats,
        "normalize_min_max": True,
    },
}
```
Similarly for the postprocessor if it has an inverse normalization step:
```python
postprocessor_overrides = {
    "groot_action_unpack_unnormalize": {
        "stats": dataset.meta.stats,
        "normalize_min_max": True,
    },
}
```

**Verification:** Dump preprocessed batches from both source and FlagScale (`torch.save`) with `shuffle=False` and `train_steps=1`, then compare all tensor keys with `(a - b).abs().max()`. All keys should show max_diff=0.0.

### T6: torchcodec fails with missing FFmpeg libraries

**Symptom:**
```
RuntimeError: Could not load libtorchcodec ... OSError: libavutil.so.60: cannot open shared object file
```

**Cause:** `torchcodec` requires FFmpeg system libraries (`libavutil`, `libavcodec`, etc.) to be installed. Many servers don't have them.

**Fix:** Switch to PyAV as the video backend. PyAV bundles its own FFmpeg:
```bash
# lerobot CLI:
--dataset.video_backend=pyav

# In FlagScale YAML config, no change needed — video decoding is handled at the dataset level
```

### T7: `accelerate launch` double-wraps the Python binary

**Symptom:**
```
SyntaxError: source code cannot contain null bytes
```
when running `accelerate launch /path/to/python -m some.module`.

**Cause:** `accelerate launch` already calls the Python binary internally. Passing the Python binary path as the "script" argument makes accelerate try to execute the binary as a Python script.

**Fix:** Use `-m` flag with accelerate to run a module:
```bash
# Wrong:
accelerate launch /path/to/python -m lerobot.scripts.lerobot_train

# Right:
accelerate launch -m lerobot.scripts.lerobot_train

# Or just use python directly for single-GPU:
python -m lerobot.scripts.lerobot_train
```

### T12: Missing `policy.train()` after `from_pretrained()`

**Symptom:** FlagScale training loss is lower than source baseline, and dropout layers produce identical input/output sums (no stochastic variation).

**Cause:** `PreTrainedModel.from_pretrained()` calls `model.eval()` at the end of weight loading (transformers line ~5068 in v4, similar in v5). If the training script doesn't explicitly call `policy.train()` afterwards, the model stays in eval mode — all `nn.Dropout` modules pass through inputs unchanged, and `nn.BatchNorm` uses running stats instead of batch stats.

This is easy to miss because:
- The model still trains (gradients flow, weights update)
- Loss still decreases
- The only visible symptom is a slightly different loss value vs. the source baseline

**Fix:** Add `policy.train()` after model creation and FSDP wrapping, before the training loop:
```python
policy = make_policy(config, ds_meta)
apply_fsdp2(policy, device_mesh)
optimizer, lr_scheduler = setup_optimizer_and_scheduler(policy, config)
policy.train()  # Required — from_pretrained() leaves model in eval mode
```

**Verification:** Use per-layer hooks (A8 step 8) and compare dropout layer outputs. In train mode, `dropout.output` should differ between FlagScale and source (different RNG). In eval mode, they should match exactly.

### T13: CUDA RNG divergence between frameworks

**Symptom:** Losses differ between FlagScale and source even with identical model weights, data, and train/eval mode. Per-layer hooks show divergence at noise sampling or time sampling (e.g., `action_encoder.W1.input` for flow matching models). Adding `torch.manual_seed(42); torch.cuda.manual_seed_all(42)` right before the forward call makes losses match.

**Cause:** The CUDA RNG state diverges between frameworks by the time the forward pass runs. Different framework initialization patterns consume different amounts of CUDA random numbers:
- FSDP2 `fully_shard()` vs. Accelerator `prepare()`
- `from_pretrained()` internals differ between transformers v4 and v5
- Different model wrapping order and parameter materialization

Any operation that calls `torch.randn()` or samples from CUDA distributions will produce different values.

**Impact:** None for real training. Both models are mathematically equivalent — they just sample different noise on each step. The per-step loss values differ, but training converges to the same quality.

**Verification:** To confirm parity despite RNG divergence:
1. Set both sides to eval mode (eliminates dropout noise)
2. Add fixed seed right before forward: `torch.manual_seed(42); torch.cuda.manual_seed_all(42)`
3. Run 1 step — losses must match exactly
4. If they do: RNG divergence is the sole cause, models are equivalent
5. If they don't: there's a real computation difference — use hooks to find it

### T14: `return_tensors` conflict with `DefaultFastImageProcessorKwargs` fallback

**Symptom:**
```
TypeError: got multiple values for argument 'return_tensors'
```
when calling `self.image_processor(images=..., return_tensors="pt")`.

**Cause:** After applying the T8 fix (`DefaultFastImageProcessorKwargs as ImagesKwargs`), the base class's `preprocess()` method reads `return_tensors` from the kwargs dataclass. If the caller also passes `return_tensors="pt"` as an explicit keyword argument, it conflicts.

**Fix:** Remove the explicit `return_tensors="pt"` from the caller (e.g., `processing_eagle2_5_vl.py`). The `DefaultFastImageProcessorKwargs` already provides a `return_tensors` field. If the default is wrong, set it in the kwargs dataclass or pass it via `data_format` config, not as a positional override.

### T15: Multi-GPU race condition on dynamic module loading

**Symptom:** Intermittent `ModuleNotFoundError`, `SyntaxError`, or corrupted `.py` files when running multi-GPU training with `trust_remote_code=True` processors.

**Cause:** Multiple ranks call `AutoProcessor.from_pretrained(trust_remote_code=True)` simultaneously. This triggers `get_class_from_dynamic_module()` which writes to the shared `$HF_HOME/modules/transformers_modules/` directory. Concurrent writes from multiple ranks corrupt files.

**Fix:** Guard processor/model loading with a distributed barrier:
```python
if dist.is_initialized() and dist.get_rank() != 0:
    dist.barrier()  # Non-rank-0 processes wait

processor = AutoProcessor.from_pretrained(cache_dir, trust_remote_code=True)

if dist.is_initialized() and dist.get_rank() == 0:
    dist.barrier()  # Rank 0 signals it's done
```
Rank 0 loads first (populating the cache), then the barrier lets other ranks load from the already-cached files.

### T16: Source repo's `pretrained_path` vs `base_model_path` confusion

**Symptom:** `ProcessorMigrationError: policy_preprocessor.json not found` or similar errors when launching source repo training.

**Cause:** Source repos (e.g., lerobot) may have two config fields for model paths:
- `pretrained_path` — used by the **factory/preprocessor loading** code (`make_pre_post_processors`) to find processor JSON config files. Setting this to a base model checkpoint (which has no processor JSON) triggers migration errors.
- `base_model_path` — used by the **model class `__init__`** to load weights.

Passing `--policy.pretrained_path=/path/to/model` when the model checkpoint doesn't contain processor config files will fail.

**Fix:** Use `--policy.base_model_path=/path/to/model` and do NOT set `pretrained_path`. The model class loads weights via `base_model_path`; `pretrained_path` is only for loading previously-saved fine-tuned policies that include processor configs. Also pass `--policy.push_to_hub=false` to avoid HF Hub auth errors.

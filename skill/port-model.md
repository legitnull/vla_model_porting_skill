# Skill: Port a Model to FlagScale

You are helping the user port a model from an external repo into FlagScale.
Follow each step below in order. Do not skip steps.

## Reporting

Every step in this skill must be logged. At the start of each step, append to the report file for that step.

Report directory: `{workspace}/reports/{model_name}_port/`

The `{workspace}` is the directory where you cloned the source repo and FlagScale (e.g. `/share/project/fengyupu/cc_workspace/`).

Each step writes its own report file named `{date}_{step_name}.md` (e.g. `2025-03-01_source_analysis.md`, `2025-03-01_env_setup.md`, `2025-03-01_training_port.md`).

For every command you run, log:
- The exact command
- The full output (or relevant excerpt if very long)
- Whether it succeeded or failed
- If it failed, what you did to fix it

For every file you read or decision you make, log:
- What you read and why
- What you learned from it
- What decision was made and the reasoning

Create the report directory as soon as you have the model name (after Step 1).

## Step 1: Gather inputs

Ask the user:

1. **Source repo path** — the local path to the repo containing the model to port
2. **Target model** — which model/class in that repo they want to port

Do not proceed until you have both answers.

## Step 2: Understand the source model

Report file for this step: `{date}_source_analysis.md`

Once you have the source repo path and target model, gather all of the following information before writing any code.

### 2a: Repo structure

List the source repo's directory structure. Identify where these live:
- Model definitions (the main model class and its sub-modules)
- Training scripts
- Inference / evaluation scripts
- Serving code (if any)
- Config files (YAML, JSON, Python dataclasses)
- Checkpoint conversion scripts (if any)

### 2b: Model architecture

Read the target model class and trace every import. For each component, record:

| Component | Source file | Class name | What it does |
|---|---|---|---|
| Top-level model | | | Orchestrates forward pass |
| VLM backbone | | | Vision-language encoder |
| Action head / decoder | | | Predicts actions or generates output |
| Projector(s) | | | Maps between embedding spaces |
| Tokenizer / processor | | | Preprocesses inputs |
| Other sub-modules | | | (list each) |

### 2c: Forward pass (training)

Trace the training forward pass and document the exact data flow:

1. What are the raw inputs? (images, text, actions, states — with shapes and dtypes)
2. How are inputs preprocessed? (tokenization, image resizing, normalization)
3. What goes into the VLM? What comes out? (input format, output hidden states shape)
4. What goes into the action head? What comes out? (conditioning, noise, target actions)
5. How is the loss computed? (loss function, masking, reduction)

### 2d: Forward pass (inference)

Trace the inference / predict path:

1. How does it differ from training? (no loss, different action head call, sampling vs. teacher forcing)
2. What is the output format? (action tensor shape, normalization)
3. Are there inference-specific parameters? (num_inference_timesteps, sampling strategy)

### 2e: Configuration

Find how the model is configured in the source repo:

- What are the hyperparameters? (hidden dims, num layers, action dim, horizon, etc.)
- Where do they come from? (config file, CLI args, hardcoded defaults)
- What pretrained weights does it load? (HuggingFace model ID, local path)

### 2f: Dependencies

List external packages the model needs beyond standard PyTorch:
- transformers (which model classes?)
- diffusers
- Any custom CUDA kernels or C++ extensions
- Dataset libraries (lerobot, energon, webdataset)

### 2g: Serving (if applicable)

Check if the source repo has serving/deployment code:
- API endpoint definitions
- Model loading for inference servers
- Batch handling for serving
- Any ONNX/TensorRT export

Present all of the above as a structured summary to the user before proceeding.

## Step 3: Set up source repo environment

The goal is to run the source model in its own repo first, so we have a baseline to match against.

Report file for this step: `{date}_env_setup.md`

### 3a: Find conda

Locate the conda binary on the server:
```bash
which conda
conda --version
```

If conda is not on PATH, check common locations:
```bash
ls ~/miniconda3/bin/conda
ls ~/anaconda3/bin/conda
ls /opt/conda/bin/conda
```

If conda does not exist anywhere, tell the user and stop.

If conda exists but is not on PATH, initialize it for shell use. All subsequent conda commands in this skill must be run as:
```bash
eval "$({conda_bin} shell.bash hook)" && conda {command}
```
where `{conda_bin}` is the full path to the conda binary (e.g. `/root/miniconda3/bin/conda`).

Log the conda path and version.

### 3b: Check server CUDA environment

Before reading the source repo's requirements, check what's available on this server:
```bash
nvidia-smi                    # GPU model, driver version, CUDA version
nvcc --version                # CUDA toolkit version
cat /usr/local/cuda/version.txt  # Alternative CUDA version check
```

Log the output. This determines which package versions are compatible.

### 3c: Read source repo install instructions

Read the source repo's README (and any `requirements.txt`, `setup.py`, `pyproject.toml`, `environment.yml`).
Extract:
- Required Python version
- Required CUDA version
- Required PyTorch version
- All other dependencies with version constraints

**If the source repo requires a CUDA or Python version incompatible with the server**, tell the user immediately and list the conflicts. Do not proceed until resolved.

### 3d: Ask user for conda env location

Ask the user where to place the new conda environment. Default suggestion: alongside their other envs (e.g. `/share/project/fengyupu/conda_envs/`).

The env name should be: `cc_port_model_{source_repo_name}` (e.g. `cc_port_model_starVLA`). One env per source repo — all models in the same repo share the same env.

### 3e: Create the conda env and install dependencies

Create the env with the correct Python version:
```bash
conda create -p {env_path}/{env_name} python={python_version} -y
conda activate {env_path}/{env_name}
```

**Use a pip mirror for faster downloads (China):**
When installing packages, pass `-i` to use the Tsinghua mirror:
```bash
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple {package}
```
Do NOT alter the global `~/.pip/pip.conf` — use `-i` per command instead. This avoids side effects on other projects.

Install PyTorch matching the server's CUDA version (use the official PyTorch index, not the mirror):
```bash
pip install torch=={version} --index-url https://download.pytorch.org/whl/cu{cuda_version}
```

Install the source repo's dependencies:
```bash
cd {source_repo_path}
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -e .           # or pip install -r requirements.txt, depending on the repo
```

If the source repo has optional extras for the target model (e.g., `pip install -e ".[groot]"`), install those too.

Log every command and its output (including any errors) to the report file.

### 3f: Verify the environment works

Run a basic import check from the source repo:
```bash
conda activate {env_path}/{env_name}
cd {source_repo_path}
python -c "from {source_module} import {TargetModelClass}; print('OK')"
```

If imports fail, debug, fix, and log the resolution.

### 3g: Find required datasets and pretrained models

Before running anything, read through the source repo to find what data and weights are needed.

**Start by finding the author's recommended starting point.** Most repos have a "Quick Start" or "Getting Started" section that recommends a specific benchmark, dataset, or training config for first-time users. This is where you should start — not the most general or most complex setup.

1. Read the repo's README end-to-end — look for:
   - **Recommended first benchmark** — phrases like "we recommend starting with", "quick start", "getting started", "try this first"
   - **Recommended training scripts** — the README often points to a specific example dir or script
   - Dataset names, download links, expected directory structure
   - Pretrained model names (HuggingFace IDs, download URLs, local paths)
   - Any data preprocessing scripts that must be run first

2. Read the recommended example directory thoroughly — look for:
   - Its own README with step-by-step instructions
   - Data download/preparation scripts
   - Training configs (YAML/JSON) specific to that benchmark
   - Training launch scripts
   - Evaluation scripts and published results

3. Read example configs and training scripts — look for:
   - Data path arguments (`--data_path`, `data_dir`, etc.)
   - Model path arguments (`--model_name_or_path`, `--pretrained`, `base_vlm`, etc.)
   - Hardcoded paths or HuggingFace model IDs in the code

4. Read evaluation / benchmark sections — look for:
   - Which datasets they report results on (these are the benchmarks to match)
   - Exact metrics and numbers (success rate, loss, action error, etc.)
   - Which checkpoint was used for each result
   - Training hyperparameters used to produce those results

Log all findings in the report. Present to the user:
- **Datasets needed**: name, source, size, download instructions
- **Pretrained models needed**: name, source, size, download instructions
- **Benchmark results to match**: dataset → metric → value

Before downloading anything, check if the required datasets and models already exist on the server:
```bash
find {common_data_dirs} -maxdepth 4 -name "{dataset_name}*" -type d
find {common_model_dirs} -maxdepth 4 -name "{model_name}*" -type d
```

Present what's already available and what's missing. Only download what's missing.

Ask the user:
- Where to download / store datasets on this server (or confirm existing paths)
- Where to download / store pretrained weights (or confirm existing paths)
- Which benchmark to target for parity validation

To download models or datasets from HuggingFace, use the script bundled with this skill:
```bash
# Download a pretrained model (use the repo_id found from the source repo)
python {skill_repo}/scripts/download_hf.py \
    --repo_id {repo_id_from_source} \
    --output_dir {user_models_dir} \
    --source huggingface

# Download a dataset (use the repo_id found from the source repo)
python {skill_repo}/scripts/download_hf.py \
    --repo_id {dataset_repo_id_from_source} \
    --output_dir {user_datasets_dir} \
    --repo_type dataset \
    --source huggingface
```

For users in China, pass `--source modelscope` instead. Note: not all repos are mirrored on ModelScope — if the repo is not found, fall back to HuggingFace with the mirror:
```bash
HF_ENDPOINT=https://hf-mirror.com python {skill_repo}/scripts/download_hf.py \
    --repo_id {repo_id} \
    --output_dir {user_models_dir} \
    --source huggingface
```

**Download priority order (China):**
1. `--source modelscope` — fastest when the repo exists on ModelScope
2. `HF_ENDPOINT=https://hf-mirror.com --source huggingface` — HuggingFace via China mirror
3. `--source huggingface` (no mirror) — direct, slowest

Do not proceed until datasets and models are available locally.

### 3h: Find the training script in the source repo

Locate the exact script used to train the target model:

1. Check the README for training commands (look for `python train.py`, `torchrun`, `deepspeed`, `accelerate launch`, etc.)
   - Some repos use non-standard CLI frameworks. E.g., lerobot uses `draccus` with `--policy.type=groot` style args, not argparse. Read the example commands from the README verbatim rather than guessing the syntax.
2. Search for training entrypoints:
   ```bash
   grep -r "def main\|def train\|if __name__" {source_repo}/scripts/ {source_repo}/train* {source_repo}/examples/
   ```
3. If multiple training scripts exist, identify which one matches the target model. Look for:
   - Script name containing the model name (e.g. `run_lerobot_datasets_qwenpi.sh`)
   - The recommended example directory's training script (from 3g)
   - Config or CLI args that select the framework/model name

4. Read the chosen script — identify:
   - What CLI arguments it takes
   - What config file(s) it expects
   - How it launches (single GPU, torchrun, deepspeed, accelerate, etc.)
   - How many nodes/GPUs the default config uses
   - How many steps/epochs the published results used

5. **Check node requirements.** If the script or config defaults to multi-node training:
   - Tell the user how many nodes the default config expects
   - Adapt the command for single-node by adjusting `--num_processes`, `--num_machines`, etc. to match the current server's GPU count
   - Note: batch size and learning rate may need scaling — log any changes made

6. Construct the exact command to run training on this server, using:
   - The datasets and models found/confirmed in Step 3g (with local paths)
   - The correct framework/model name for the target model
   - `--num_processes` matching the number of GPUs on this server
   - A reduced `--max_train_steps` for the baseline test (e.g. 100–500 steps)
   - wandb disabled or pointed to the user's account

Present the full command to the user for confirmation before running.

### 3i: Run the source model (baseline)

Run the training command confirmed in 3h.

1. Run the command — log exact command and full output
2. Watch for:
   - Import errors or missing dependencies
   - CUDA out of memory — reduce batch size and log the change
   - Data loading errors — check dataset format/version matches code expectations. E.g. LeRobot v2.1 uses `tasks.jsonl` while v3.0 uses `tasks.parquet` — if the user has multiple versions, ask which to use. Set up a local data directory with symlinks rather than modifying the user's data dirs.
   - NCCL timeouts — adjust NCCL env vars
   - flash-attn, video backend, transformers compatibility issues — see Troubleshooting (T1–T10)
3. Confirm training starts and loss decreases
4. Record baseline metrics: loss at step 0, loss after N steps, learning rate, throughput (samples/sec)
5. If the source repo has an eval script, run it and compare against published results

Save these baseline numbers — they are the parity target for the FlagScale port.

## Step 4: Set up FlagScale destination repo

Report file for this step: `{date}_flagscale_setup.md`

The port target is FlagScale. Clone a fresh copy — do **not** use the user's existing FlagScale repo.

### 4a: Clone FlagScale

Get the FlagScale remote URL from the user's repo (if available), or ask the user for it:
```bash
cd {user_flagscale_repo} && git remote -v
```

Clone into the workspace:
```bash
git clone {flagscale_remote_or_local} {workspace}/FlagScale
```

If the user's fork has VLA infrastructure not yet in upstream, clone from the user's local repo instead of GitHub. Use `git clone` (not `cp`) to avoid copying large untracked files.

Log the clone source, branch, and HEAD commit.

### 4b: Set up FlagScale conda env

FlagScale may need its own conda environment, separate from the source repo's env. Check if the user already has one:

```bash
ls {user_conda_envs_dir}/flagscale*
```

If the user has an existing FlagScale env, ask them for the path. If not, create one following the same pattern as Step 3e, but with FlagScale's requirements:
```bash
cd {workspace}/FlagScale
pip install -e .
```

### 4c: Verify FlagScale imports

```bash
conda activate {flagscale_env}
cd {workspace}/FlagScale
python -c "from flagscale.train.train_config import TrainConfig; print('OK')"
```

If the FlagScale repo has existing VLA models, also verify those import:
```bash
python -c "from flagscale.models.vla.base_policy import TrainablePolicy; print('OK')"
```

### 4d: Explore FlagScale VLA infrastructure

Read these files to understand the existing patterns:

**Base classes and interfaces:**
- `flagscale/models/vla/base_policy.py` — `TrainablePolicy` base class with `forward()` and `predict_action()` abstract methods, plus `input_features`/`output_features` properties
- `flagscale/models/vla/protocols.py` — `VLMBackbone` protocol (prepare_input, forward, fsdp_units) and `ActionModel` protocol (forward, predict, fsdp_units)
- `flagscale/models/vla/registry.py` — `VLM_REGISTRY`/`ACTION_MODEL_REGISTRY` dicts, `register_vlm()`/`register_action_model()` decorators, `build_vlm()`/`build_action_model()` factories

**Configuration:**
- `flagscale/train/train_config.py` — Pydantic models: `TrainConfig` (top-level), `SystemConfig`, `ModelConfig`, `DataConfig`, `OptimizerConfig`, `SchedulerConfig`, `FreezeConfig`, `CheckpointConfig`
- `ModelConfig` has a `validate_model_name` validator with a set of valid names — new models must be added here

**Existing VLM backends (`flagscale/models/vla/vlm/`):**
- List all files and their registered VLM names (e.g., `qwen_vl.py` → `Qwen25VLBackbone`, `Qwen3VLBackbone`)
- Each backend extends `QwenVLBackbone` (or similar base) and implements `_load_model()`, `prepare_input()`, `forward()`, `fsdp_units()`

**Existing action models (`flagscale/models/vla/action_models/`):**
- List all files and their registered names (e.g., `flow_matching.py` → `FlowMatchingHead`)
- Each wraps an underlying action head and implements `forward(vlm_output, action_input)`, `predict_action(vlm_output, action_input)`, `fsdp_units()`

**Existing model ports (`flagscale/models/vla/`):**
- Read at least one complete existing port (e.g., `qwen_gr00t.py` → `QwenGr00t`) end-to-end
- Note how it: extends `TrainablePolicy`, uses `build_vlm()`/`build_action_model()`, implements `forward()` returning `{"loss": Tensor}`, implements `predict_action()`, implements `save_pretrained_configs()`

**Existing training entrypoints (`flagscale/train/train_*.py`):**
- Read one VLA training entrypoint (e.g., `train_qwen_gr00t.py`) — note the structure:
  - `main(config, seed)` — dist init, dataset, policy, FSDP2, optimizer, training loop
  - `make_policy(config, ds_meta)` — instantiate model, set features, move to CUDA
  - `apply_fsdp2(policy, device_mesh)` — per-unit sharding with MixedPrecisionPolicy
  - `make_dataset(cfg)` — LeRobotDataset with delta_timestamps
  - `update_policy(...)` — zero_grad, forward, backward, clip, step
  - Config loaded via `TrainConfig.from_hydra_config()`

**Existing example configs:**
- Read one VLA training YAML (e.g., `examples/qwen_gr00t/conf/train/qwen_gr00t.yaml`) — note the sections: `system` (batch_size, train_steps, checkpoint), `model` (model_name, checkpoint_dir, vlm, qwenvl, action_model, optimizer, freeze), `data` (data_path, delta indices, preprocessor, postprocessor)

**Utility files:**
- `flagscale/models/vla/utils.py` — helper functions (e.g., `get_vlm_config`)
- `flagscale/models/utils/constants.py` — shared constants (`ACTION`, `OBS_PREFIX`, etc.)
- `flagscale/train/utils/optim_setup.py` — `setup_optimizer_and_scheduler()` (handles param groups, freeze, scheduler creation)
- `flagscale/train/utils/train_utils.py` — `save_checkpoint()`, `get_step_checkpoint_dir()`, `update_last_checkpoint()`

Record which source model components already have FlagScale equivalents vs. what needs to be created.

### 4e: Diff analysis — source model vs. existing FlagScale models

Now you have full knowledge of both sides: the source model (from Step 2) and the FlagScale infrastructure (from 4d). Compare them systematically.

**For each component of the source model (from Step 2b), find the closest FlagScale equivalent and document the differences.**

Build a table like this:

| Component | Source | Closest FlagScale equivalent | Can reuse? | Differences |
|---|---|---|---|---|
| VLM backbone | `{source_vlm_class}` | `{flagscale_vlm_class}` | Yes/No/Partial | List specific differences |
| Action head | `{source_action_class}` | `{flagscale_action_class}` | Yes/No/Partial | List specific differences |
| ... | ... | ... | ... | ... |

For each component, check these dimensions:
1. **Architecture** — same model class? same layer structure? different number of layers?
2. **Forward pass inputs** — same input format? (e.g., single hidden state vs. list of all layer hidden states)
3. **Forward pass outputs** — same output format? same loss computation?
4. **Inference path** — same sampling/integration? same number of steps?
5. **Config structure** — same hyperparameters? different defaults?

**Classify each component as one of:**
- **Reuse as-is** — identical architecture, no code changes needed
- **Reuse with config** — same code, different hyperparameters (e.g., different num_layers, hidden_dim)
- **Extend** — needs minor modifications (e.g., add a flag to switch between single-layer and layer-wise cross-attention)
- **Create new** — fundamentally different, must port from source

**For "Create new" components, describe exactly what is different and what needs to be built.**

Present this analysis to the user before proceeding. This determines the scope of work for each track.

## Step 5: Scope the work

The port has up to three separate tracks. Ask the user which to focus on first:

1. **Training** — model class, training entrypoint, config YAML, dataset integration
2. **Inference** — predict_action / generate, checkpoint loading, evaluation scripts
3. **Serving** — API endpoints, model server integration, batch inference

Each track is independent. Do them one at a time. The rest of this skill covers each track.

---

## Track A: Training

Report file for this track: `{date}_training_port.md`

### A1: Set up model subdirectory

Create a subdirectory under `flagscale/models/vla/` for the ported model's specific files:

```
flagscale/models/vla/{model_name}/
├── __init__.py
├── {model_name}_core.py          # Core model (PreTrainedModel), config, sub-modules
├── modeling_{model_name}.py      # TrainablePolicy wrapper
├── processor_{model_name}.py     # Model-specific preprocessor steps (if needed)
└── {vendored_files}/             # Vendored model-specific files from source
```

**Naming consistency:** Decide on the model name spelling upfront (e.g., `gr00t` vs `groot`) and use it consistently across the directory name, all filenames, config `model_name`, class names, and the training entrypoint. Do not mix spellings.

**Two-file split:** If the source repo separates the core model (e.g., `PreTrainedModel` with `from_pretrained()`) from the training wrapper (e.g., the policy class that calls the core model), preserve that split:
- Core model file — contains the model architecture, config dataclass, `from_pretrained()`, `forward()`, `get_action()`. Import shared building blocks from FlagScale rather than duplicating them.
- Modeling file — contains the `TrainablePolicy` subclass that wraps the core model, implements `forward(batch) -> {"loss": Tensor}` and `predict_action(batch) -> {"action": Tensor}`, and exposes `fsdp_units()`.

This directory holds:
- The model's own modeling and configuration code
- Any **vendored files** that the source model depends on which aren't shared with other models (e.g., custom HuggingFace model implementations like `eagle2_hg_model/` for GR00T, custom tokenizer code, etc.)
- **Model-specific preprocessor steps** — if the model needs custom preprocessing (e.g., packing video/state/action into a specific format, running a VLM processor to tokenize images), these go here as `ProcessorStep` subclasses registered via `@ProcessorStepRegistry.register()`. Generic steps (device, normalize, rename) stay in `flagscale/train/processor/`.

Shared components that could be used by multiple models (DiT blocks, action encoders, VLM backbones) still go in the shared locations (`vla/vlm/`, `vla/action_model/`, etc.).

### A2: Plan file mapping

Using the diff analysis from Step 4e, build a concrete file mapping for the training track. For each component, list the exact source file path, the exact FlagScale target file path, and whether to reuse/extend/create.

The mapping must include at minimum:
- Model subdirectory → `flagscale/models/vla/{model_name}/` (always create new)
- Core model file → `flagscale/models/vla/{model_name}/{model_name}_core.py` (create new — contains PreTrainedModel, config, sub-modules)
- Top-level model class → `flagscale/models/vla/{model_name}/modeling_{model_name}.py` (always create new — TrainablePolicy wrapper)
- VLM backbone → reuse or extend existing in `flagscale/models/vla/vlm/`, or create new in the model subdir if model-specific
- Action model wrapper → reuse or create new in `flagscale/models/vla/action_model/`
- Model-specific preprocessor → `flagscale/models/vla/{model_name}/processor_{model_name}.py` (create new if model needs custom preprocessing steps beyond the generic pipeline)
- Vendored dependencies → `flagscale/models/vla/{model_name}/{vendored_dir}/` (copy from source, only for model-specific code)
- Training entrypoint → `flagscale/train/train_{model_name}.py` (always create new)
- Config YAML → `examples/{model_name}/conf/train/{model_name}.yaml` (always create new)

Ask the user to confirm before writing code.

### Porting guidelines

When porting code from the source repo, follow these rules:
- **Preserve original docstrings and comments.** Keep them as-is unless they reference source-specific APIs that no longer apply. Do not strip, rewrite, or summarize them.
- **Preserve original class/method signatures** where possible, to make diffing against upstream easier.
- **Add attribution headers** at the top of each ported file: `# Mainly adopted from:` with source URL, then `# Below is the original copyright:` with the original license block. Do NOT add inline modification comments like `# Modified for FlagScale` or section separator comments — the code should speak for itself.
- **Do not add unnecessary comments** explaining what you changed. The attribution header and git diff are sufficient.
- **State dict key compatibility:** When porting a model that loads pretrained weights via `from_pretrained()`, ensure that attribute names on your ported classes match the pretrained checkpoint's state dict keys exactly. For example, if the checkpoint has keys `backbone.eagle_model.vision_model.*`, the ported class must have `self.backbone.eagle_model.vision_model` with those exact names. Renaming attributes will break weight loading.
- **Reuse shared building blocks** from FlagScale rather than duplicating code. Before creating a new module, check if equivalent implementations already exist in `vla/action_model/`, `vla/vlm/`, or `flagscale/models/`. Import from the shared location and only define new classes when the existing ones are insufficient.

### A3: Port or register sub-components

For each component marked "create new" in A1:

**VLM backbone** → `flagscale/models/vla/vlm/{name}.py`:
- Extend `QwenVLBackbone` (or create new base if needed)
- Implement `_load_model(model_id)` — load from HuggingFace with correct attn_implementation and dtype
- Implement `prepare_input(batch, image_feature_keys)` — convert batch dict to model inputs (tokenize, process images)
- `forward()` is inherited from base class — override only if needed
- Implement `fsdp_units()` — return list of submodules for per-unit FSDP sharding (typically `visual.blocks` + `language_model.layers`)
- Add to `flagscale/models/vla/vlm/__init__.py`

**Action model** → `flagscale/models/vla/action_models/{name}.py`:
- Create wrapper class extending `nn.Module`
- Constructor takes `vlm_config`, `action_config: dict`, `full_config: TrainConfig`
- Use `get_vlm_config(vlm_config)` to extract `hidden_size` from VLM config
- Implement `forward(vlm_output, action_input) -> {"loss": Tensor}` — extract hidden_states from vlm_output, actions/state from action_input, delegate to underlying head
- Implement `predict_action(vlm_output, action_input) -> {ACTION: Tensor}` — Euler integration / sampling loop
- Implement `fsdp_units()` — return DiT transformer blocks
- Add to `flagscale/models/vla/action_models/__init__.py`

**Underlying action head modules** (if not already in FlagScale):
- Check `flagscale/models/action_model/` and `flagscale/models/robobrain_x/` for existing implementations
- Port new modules (e.g., different DiT architectures, new encoders) to `flagscale/models/action_model/`

**Model-specific preprocessor steps** → `flagscale/models/vla/{model_name}/processor_{model_name}.py`:
- Some models need custom preprocessing that goes beyond the generic pipeline (rename → batch → normalize → device). For example, GR00T N1.5 needs steps to pack video/state/action/embodiment, run the Eagle VLM processor to tokenize images, and collate the resulting tokens.
- Implement each step as a `@dataclass` extending `ProcessorStep` from `flagscale.train.processor.pipeline`
- Register each step via `@ProcessorStepRegistry.register(name="...")` so it can be referenced by name in YAML configs
- Implement `__call__(self, transition: EnvTransition) -> EnvTransition` — read from `transition[TransitionKey.OBSERVATION]`, `transition[TransitionKey.ACTION]`, `transition[TransitionKey.COMPLEMENTARY_DATA]` and write back
- Implement `transform_features(self, features) -> features` (can return unchanged if features don't change shape/type)
- If the step has state (e.g., normalization stats), implement `get_config()`, `state_dict()`, `load_state_dict()` for serialization
- Also create a corresponding **postprocessor step** if needed (e.g., inverse normalization, action slicing)
- The training script must `import` this file to trigger step registration — add a `import flagscale.models.vla.{model_name}.processor_{model_name}  # noqa: F401` at the top of the training entrypoint

### A4: Port the model class

Create `flagscale/models/vla/{model_name}/modeling_{model_name}.py`:

1. **Attribution header** — source URL, original copyright, modification notes
2. **Extend `TrainablePolicy`** from `flagscale.models.vla.base_policy`
3. **Constructor** takes `config: TrainConfig`:
   - Build VLM via `build_vlm(type_name, config=config)` where type_name comes from `config.model.vlm.type`
   - Build action model via `build_action_model(type_name, vlm_config=self.vlm.model_config, action_config={}, full_config=config)` where type_name comes from `config.model.action_model.type`
   - Store key config values (future_action_window_size, etc.)
   - Handle input/output feature deserialization from config (for checkpoint loading)
4. **`forward(batch) -> {"loss": Tensor}`**:
   - Extract actions from `batch[ACTION]`
   - Call `self.vlm.prepare_input(batch, image_feature_keys=...)` to build VLM inputs
   - Run VLM forward under `torch.autocast("cuda", dtype=torch.bfloat16)`
   - Extract hidden states from VLM output
   - Apply action repetition for flow matching (`repeated_diffusion_steps`)
   - Call `self.action_model.forward(vlm_output, action_input)` under `torch.autocast("cuda", dtype=torch.float32)`
   - Return `{"loss": output["loss"]}`
5. **`predict_action(batch) -> {ACTION: Tensor}`**:
   - Same VLM forward path
   - Call `self.action_model.predict_action(vlm_output, action_input)`
6. **`save_pretrained_configs(save_dir)`** — save VLM config + processor for checkpoint portability

**Key differences to watch for between source and FlagScale:**
- Source may pass raw `List[dict]` as batch; FlagScale passes a flat `dict[str, Tensor]` from LeRobot DataLoader
- Source may use `output_hidden_states=True` and take all layers; FlagScale typically takes only the last hidden state (check source model's specific needs — e.g., layer-wise cross-attention needs ALL layers)
- Source may handle image preprocessing differently (PIL vs. tensor, different resize)

### A5: Create training entrypoint

Create `flagscale/train/train_{model_name}.py` following `train_qwen_gr00t.py` as the reference:

1. **`main(config, seed)`** — the training loop:
   - `set_seed(seed)`
   - `dist.init_process_group(backend="nccl")`
   - `make_dataset(config.data)` — create LeRobotDataset
   - `make_policy(config, ds_meta)` — instantiate model
   - `apply_fsdp2(policy, device_mesh)` — shard with FSDP2
   - Create preprocessor from `config.data.preprocessor`
   - `setup_optimizer_and_scheduler(policy, config)` — handles param groups, freeze, scheduler
   - Training loop: `next(dl_iter)` → preprocess → `update_policy()` → log → checkpoint

2. **`make_policy(config, ds_meta)`**:
   - `dataset_to_policy_features(ds_meta.features)` to get input/output features
   - Instantiate the model class with `config`
   - Set `policy.input_features` and `policy.output_features`
   - `policy.to("cuda")`

3. **`apply_fsdp2(policy, device_mesh)`**:
   - Cast to float32 first (`policy.float()`)
   - Create `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)`
   - Shard all FSDP-able units: `for unit in policy.fsdp_units(): fully_shard(unit, ...)`
     - The `fsdp_units()` method on the `TrainablePolicy` wrapper should return units from all submodels (backbone + action head). This keeps `apply_fsdp2` clean and decoupled from internal model structure.
     - Alternative (for models using registry): shard VLM and action model separately via `policy.vlm.fsdp_units()` and `policy.action_model.fsdp_units()`
   - Shard root: `fully_shard(policy, ...)`

4. **`update_policy(...)`** — can often be reused from an existing entrypoint or shared utility

5. **Register the model name**: Add `"{model_name}"` to the `valid_names` set in `ModelConfig.validate_model_name` in `flagscale/train/train_config.py`

6. **Import model-specific processor steps**: If the model has custom preprocessor steps in `processor_{model_name}.py`, add a bare import at the top of the training script to trigger `@ProcessorStepRegistry.register()`:
   ```python
   import flagscale.models.vla.{model_name}.processor_{model_name}  # noqa: F401
   ```
   Without this import, the processor steps won't be in the registry and the YAML config's `registry_name` references will fail at runtime.

7. **Export from `__init__.py`**: Add the `TrainablePolicy` wrapper class to both `flagscale/models/vla/{model_name}/__init__.py` and `flagscale/models/vla/__init__.py`.

### A6: Create config YAML

Create `examples/{model_name}/conf/train/{model_name}.yaml` with these sections:

```yaml
system:
  batch_size: {from_source}
  train_steps: {from_source}
  log_freq: 10
  grad_clip_norm: {from_source}
  use_amp: true
  shuffle: true
  num_workers: {from_source}
  checkpoint:
    output_directory: ${experiment.exp_dir}
    save_checkpoint: true
    save_freq: 10000

model:
  model_name: {model_name}
  checkpoint_dir: {path_to_pretrained_vlm}
  vlm:
    type: {vlm_type}          # e.g. qwen3-vl, qwen2.5-vl
  qwenvl:
    base_vlm: {path_to_pretrained_vlm}
    attn_implementation: flash_attention_2
    vl_hidden_dim: {from_source}
  action_model:
    type: {action_model_type}  # e.g. flow_matching, layerwise_flow_matching
    # ... all action head hyperparameters from source config ...
  optimizer:
    name: AdamW
    lr: {from_source}
    betas: {from_source}
    eps: {from_source}
    weight_decay: {from_source}
    scheduler:
      name: {from_source}
      warmup_steps: {from_source}

data:
  data_path: {path_to_dataset}
  tolerance_s: 0.0001
  observation_delta_indices: {from_source}
  action_delta_indices: {from_source}
  preprocessor:
    # ... processor pipeline steps ...
  postprocessor:
    # ... unnormalization steps ...
```

**Do NOT guess parameter values.** Every `{from_source}` placeholder must be filled from the verified values obtained in Step A6a below.

**Model-specific preprocessor in YAML:** If the model has custom preprocessor steps (registered in `processor_{model_name}.py`), list them in the `preprocessor.steps` section using their `registry_name`. The typical pipeline is:
1. `rename_observations_processor` — key renaming (generic)
2. `to_batch_processor` — add batch dim (generic)
3. Model-specific packing step(s) — e.g., `groot_pack_inputs`
4. Model-specific encoding step(s) — e.g., `groot_eagle_encode`, `groot_eagle_collate`
5. `device_processor` — move to GPU (generic)

Ask the user for:
- `checkpoint_dir` / `base_vlm` — path to pretrained VLM weights
- `data_path` — path to training dataset
- Any benchmark-specific overrides

### A6a: Align training parameters with source baseline

**Before the first FlagScale training run**, verify that every training parameter matches the source baseline exactly. Do NOT rely on code reading alone — run a script inside the source repo's environment to print the actual resolved values.

**Step 1: Print actual values from the source repo.**

Write a short Python script that imports the source repo's config classes and prints the resolved training parameters. Run it inside the source repo's conda environment. Example:

```python
# Run inside source repo env
from {source_config_module} import {PolicyConfig}
config = {PolicyConfig}()
opt = config.get_optimizer_preset()
sched = config.get_scheduler_preset()
print(f"optimizer: {type(opt).__name__}")
print(f"lr: {opt.lr}, betas: {opt.betas}, eps: {opt.eps}")
print(f"weight_decay: {opt.weight_decay}, grad_clip_norm: {opt.grad_clip_norm}")
print(f"scheduler: {type(sched).__name__}")
# ... print all scheduler fields ...
```

If the source uses CLI overrides (e.g., `--batch_size=4 --steps=20`), also simulate the effect of those overrides (auto-scaling, preset selection, etc.).

**Step 2: Build a parameter comparison table.**

For every training parameter, document: the lerobot/source value, where it comes from (which class/field/default), the current FlagScale value, and whether they match:

| Parameter | Source Value | Source Location | FlagScale Value | Match? |
|---|---|---|---|---|
| batch_size | 4 | CLI override | 4 | YES |
| seed | 1000 | TrainPipelineConfig.seed | 1000 | YES |
| grad_clip_norm | 10.0 | AdamWConfig.grad_clip_norm | 10.0 | YES |
| lr | 1e-4 | GrootConfig.optimizer_lr | 1e-4 | YES |
| scheduler | cosine_decay_with_warmup | get_scheduler_preset() | cosine_decay_with_warmup | YES |
| param_groups | None (flat) | get_optim_params() | None | YES |
| ... | ... | ... | ... | ... |

Parameters to check (at minimum):
- batch_size, seed, num_workers, shuffle
- optimizer: name, lr, betas, eps, weight_decay
- grad_clip_norm
- param_groups (flat vs. per-module lr)
- scheduler: type, warmup_steps, decay_steps, peak_lr, decay_lr, auto-scaling behavior
- AMP / dtype (bf16, fp32, mixed)
- normalization mode (min_max, mean_std, identity)
- observation_delta_indices, action_delta_indices
- action_horizon, max_state_dim, max_action_dim
- tune_llm, tune_visual, tune_projector, tune_diffusion_model

**Step 3: Fix mismatches.**

For every row where Match? is NO, update the FlagScale config YAML (or code) to match the source value. Common pitfalls:
- **Scheduler mismatch** — the source repo may use a custom scheduler (e.g., `CosineDecayWithWarmup` with auto-scaling) that doesn't exist in `transformers.get_scheduler()`. Check if FlagScale already has it (e.g., in `optim_setup.py` or another training script). If so, reuse it. If not, port it.
- **Param groups mismatch** — the source may use flat `model.parameters()` with a single lr, while the FlagScale config has per-module param groups. Remove param_groups from the YAML if the source doesn't use them.
- **grad_clip_norm** — commonly different between source defaults and FlagScale defaults. Check carefully.
- **num_workers hardcoded in code** — check that the training script actually uses `config.system.num_workers` and doesn't override it with a hardcoded 0.

**Step 4: Verify LR schedule numerically.**

For the scheduler specifically, compute the LR at every step for a short run (e.g., 20 steps) for BOTH the source and FlagScale schedulers. Print both side by side and confirm they match:

```python
# Compare LR schedules
for step in range(num_steps + 1):
    source_lr = source_scheduler_lr_at(step)
    fs_lr = flagscale_scheduler_lr_at(step)
    match = "OK" if abs(source_lr - fs_lr) < 1e-10 else "MISMATCH"
    print(f"step {step}: source={source_lr:.6e} fs={fs_lr:.6e} {match}")
```

Log the comparison table and LR schedule to the report file.

### A7: Set up FlagScale env for the ported model

The FlagScale env may need additional packages from the source model. Install them:
```bash
conda activate {flagscale_env}
pip install {missing_packages}  # e.g. qwen-vl-utils, flash-attn
```

### A8: Validate training

1. **Import check**:
   ```bash
   python -c "from flagscale.models.vla.{model_name} import {ClassName}; print('OK')"
   ```

2. **Config check** — load YAML, verify TrainConfig parses:
   ```bash
   python -c "
   from omegaconf import OmegaConf
   from flagscale.train.train_config import TrainConfig
   cfg = OmegaConf.load('examples/{model_name}/conf/train/{model_name}.yaml')
   # Quick smoke test that model_name is accepted
   print('model_name:', cfg.model.model_name)
   print('OK')
   "
   ```

3. **Run a short training test** using the FlagScale runner or torchrun:
   ```bash
   torchrun --nproc-per-node={num_gpus} \
     flagscale/train/train_{model_name}.py \
     --config-file examples/{model_name}/conf/train/{model_name}.yaml
   ```

4. **Compare against baseline**:
   - Loss at step 0 should be similar to source baseline
   - Loss curve should decrease at a similar rate
   - If loss diverges, check: dtype mismatches, preprocessing differences, action target slicing, noise sampling

---

## Track B: Inference

Report file for this track: `{date}_inference_port.md`

### B1: Explore FlagScale inference infrastructure

Read existing inference code in FlagScale to understand the patterns:
- `flagscale/inference/` — existing inference scripts
- How checkpoints are loaded
- How predict_action / generate is called
- How outputs are post-processed (unnormalization, action space conversion)

### B2: Port inference path

Ensure the model's `predict_action()` method works correctly:
- Checkpoint loading from FlagScale-saved format
- Input preprocessing matches training
- Output unnormalization matches training
- Inference-specific parameters (num_timesteps, sampling) are configurable

### B3: Create evaluation script

Port or create an evaluation script that:
- Loads a trained checkpoint
- Runs inference on a test dataset or environment
- Reports metrics (success rate, action error, etc.)

### B4: Validate inference

1. Load a checkpoint saved during training
2. Run predict_action on a test batch
3. Compare outputs against source model's inference for same inputs

---

## Track C: Serving

Report file for this track: `{date}_serving_port.md`

### C1: Explore FlagScale serving infrastructure

Read existing serving code:
- `flagscale/serve/` or equivalent
- API endpoint patterns
- Model loading and warm-up
- Request/response formats

### C2: Create serving endpoint

- Model loading from checkpoint
- Request parsing (images, instructions)
- Inference call with proper preprocessing
- Response formatting (action trajectory)

### C3: Validate serving

1. Start the server
2. Send a test request
3. Verify response format and correctness

---

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

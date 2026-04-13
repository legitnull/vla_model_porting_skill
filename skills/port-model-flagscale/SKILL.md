---
name: port-model-flagscale
description: >
  Port a VLA (Vision-Language-Action) model from an external repo into FlagScale.
  Covers source analysis, environment setup, model class creation, training entrypoint,
  config YAML, and parity validation for both Native (FSDP2) and Megatron backends.
  Trigger when the user says things like "port X model", "add X to FlagScale",
  "migrate X model to FlagScale", or simply "/port-model".
argument-hint: model_name [source_repo_path]
user-invokable: true
compatibility: "Python 3.8+, FlagScale, CUDA GPU"
metadata:
  version: "1.0.0"
  author: fengyupu
  category: workflow-automation
  tags: [model-porting, vla, flagscale, training, megatron, fsdp]
allowed-tools: "Bash(python:*) Bash(python3:*) Bash(git:*) Bash(pip:*) Bash(ls:*) Bash(cp:*) Bash(mkdir:*) Bash(bash:*) Bash(torchrun:*) Bash(nvidia-smi:*) Read Edit Write Glob Grep AskUserQuestion TaskCreate TaskUpdate TaskList TaskGet Agent"
---

# Skill: Port a Model to FlagScale

You are helping the user port a model from an external repo into FlagScale.
Follow each step below in order. Do not skip steps.

## Execution

### Step 1: Gather inputs

Ask the user for:
- **Source repo path** — the local path to the repo containing the model to port
- **Target model** — which model/class in that repo they want to port

Do not proceed until you have both answers.

**→ Tell user**: Confirm the source repo path and target model name.

### Step 2: Load references

Read these files (relative to this SKILL.md):
- `references/reporting.md` — logging discipline (log-before-you-act rule)
- `references/procedure.md` — step-by-step porting procedure (Steps 2–5, Tracks A/B/C)
- `references/troubleshooting.md` — known issues and fixes (T1–T16)

Internalize the reporting rules. You MUST follow the log-before-you-act discipline from `reporting.md` for every action from this point forward.

The procedure references executable scripts in `scripts/`:
- `scripts/diagnostics.py` — environment and dependency diagnostics
- `scripts/download_hf.py` — download HuggingFace models/datasets
- `scripts/hooks.py` — debug hooks for tracing model execution
- `scripts/plot_loss_comparison.py` — compare loss curves between source and ported model

**→ Tell user**: "References loaded. Starting source analysis."

### Step 3: Execute procedure

With inputs gathered, execute every step in `procedure.md` sequentially, starting from Step 2 (source analysis). Follow the reporting rules throughout. When you hit an error, consult `troubleshooting.md` for known fixes before debugging from scratch.

**→ Tell user**: Before starting, output a numbered plan of all steps. Report progress at each step boundary.

## Overview

| Phase | Steps | What happens |
|---|---|---|
| Gather inputs | Step 1 | Get source repo path and target model from user |
| Source analysis | Step 2 | Understand architecture, forward pass, config, dependencies |
| Environment setup | Step 3 | Set up conda envs, install deps, run source baseline |
| FlagScale setup | Step 4 | Clone FlagScale, explore VLA infra, diff against source |
| Scope the work | Step 5 | Decide which tracks to execute |
| Training port | Track A (A1–A8) | Port model class, create entrypoint + config, validate training |
| Inference port | Track B (B1–B4) | Port inference path, validate predictions |
| Serving port | Track C (C1–C3) | Create serving endpoint, validate end-to-end |

## Examples

**Example 1: Port a new VLA model**
```
User says: "/port-model" or "port QwenGr00t from starVLA to FlagScale"
Actions:
  1. Read all references
  2. Gather inputs (Step 1): source repo path + target model
  3. Analyze source model (Step 2): architecture, forward pass, config
  4. Set up environments (Steps 3–4): source env + FlagScale env
  5. Scope work (Step 5): typically Track A first
  6. Execute Track A: port model, create entrypoint + config, validate training
  7. Execute Track B/C if applicable
Result: Model fully ported and validated in FlagScale
```

**Example 2: Resume after interruption**
```
User says: "continue" or "resume porting"
Actions:
  1. Check TaskList for in_progress/pending tasks
  2. Read the latest report files to recover context
  3. Continue from the first non-completed task
Result: Porting resumed without re-doing completed work
```

## Troubleshooting

See `references/troubleshooting.md` for the full catalog (T1–T16). Common issues:

| Problem | Entry |
|---|---|
| flash-attn build errors | T1 |
| transformers 5.x meta tensor issues | T2, T3, T4 |
| Vendored processor/model code conflicts | T5, T8 |
| torchcodec / accelerate issues | T6, T7 |
| FlagScale env issues | T9, T10 |
| Training parity issues | T11, T12, T13 |
| Multi-GPU race conditions | T15 |
| Source model needs different transformers version | T17 |

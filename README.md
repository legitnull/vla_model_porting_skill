# VLA Model Porting Skill

A Claude Code skill for porting VLA (Vision-Language-Action) models from external frameworks into [FlagScale](https://github.com/FlagOpen/FlagScale).

## What it does

Guides Claude through the full model porting workflow:

1. Reading and understanding the source model
2. Creating the policy/model class with the correct FlagScale base class
3. Registering sub-components (VLM backbone, action head)
4. Creating the training entrypoint
5. Creating the config YAML
6. Validating parity with the source

Supports both FlagScale training backends:
- **Native** — FSDP2 + PyTorch (HuggingFace-based models)
- **Megatron** — Megatron-LM tensor/pipeline parallelism

## Install

Copy the skill file into your FlagScale project's `.claude/skills/` directory:

```bash
./install.sh /path/to/your/FlagScale
```

Or manually:

```bash
mkdir -p /path/to/your/FlagScale/.claude/skills
cp skill/port-model.md /path/to/your/FlagScale/.claude/skills/
```

## Usage

In a Claude Code session inside your FlagScale repo:

```
/port-model
```

Or describe the task naturally — the skill activates when you ask to port a model.

## Included example

The skill includes a documented walkthrough of porting **QwenGr00t** from [starVLA](https://github.com/starVLA/starVLA) to FlagScale, showing the exact source→target file mapping and the adaptations required.

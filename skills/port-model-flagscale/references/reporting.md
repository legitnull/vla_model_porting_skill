## Reporting

Every step in this skill must be logged. At the start of each step, append to the report file for that step.

Report directory: `{workspace}/reports/{model_name}_port/`

The `{workspace}` is the directory where you cloned the source repo and FlagScale (e.g. `/share/project/fengyupu/cc_workspace/`).

Each step writes its own report file named `{date}_{step_name}.md` (e.g. `2025-03-01_source_analysis.md`, `2025-03-01_env_setup.md`, `2025-03-01_training_port.md`).

**Log before you act.** Before executing any command, editing any file, or making any decision, write the intended action to the report first. This ensures you have a record of what you planned to do even if the action fails or the session is interrupted. After execution, update the report with the result.

For every command you run, log:
- The intended action (written BEFORE execution)
- The exact command
- The full output (or relevant excerpt if very long)
- Whether it succeeded or failed
- If it failed, what you did to fix it

For every file you read or decision you make, log:
- What you read and why
- What you learned from it
- What decision was made and the reasoning

Create the report directory as soon as you have the model name (after Step 1).

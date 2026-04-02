# Project: VLA Model Porting Skill

## CRITICAL WORKFLOW RULE — Log Before You Act

When running the port-model skill, you MUST follow this sequence for EVERY action (command, file edit, file read, decision):

1. **WRITE intention to the report file FIRST** — before executing anything
2. **Execute** the command / edit / read
3. **UPDATE the report** with the result (output, success/fail, fix if needed)

DO NOT batch actions. DO NOT skip the pre-log. DO NOT "catch up" on logging after the fact.
If you catch yourself about to run a command without having written the intention first, STOP and write it first.

This is the #1 failure mode observed in past sessions. Treat a missing pre-log as a bug.

## Report structure

- Directory: `{workspace}/reports/{model_name}_port/`
- One file per step: `{date}_{step_name}.md`
- For every command: intended action (before), exact command, output, success/fail, fix
- For every read/decision: what, why, what you learned, decision + reasoning

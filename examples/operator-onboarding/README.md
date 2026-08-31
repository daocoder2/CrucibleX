# Operator Onboarding Examples

This directory contains the smallest complete set of files for bringing a new operator into CrucibleX.

## Files

- `operator_facts.yaml`: operator facts, supported backends, generator intent, and oracle policy
- `case_template.yaml`: a minimal case file you can run locally first
- `executor_template.py`: optional custom executor skeleton
- `nodes_template.yaml`: local node template for the first smoke run

## Start Here

Generate a scaffold from the facts file:

```bash
uv run cx onboard --facts examples/operator-onboarding/operator_facts.yaml --output cx_output/onboarding-smoke
```

Run the minimal local smoke:

```bash
uv run cx run --case examples/operator-onboarding/case_template.yaml --nodes examples/nodes/local.yaml --task run --scheduler local --output cx_output/onboarding-smoke
```

Expand generated coverage before promoting the case set:

```bash
uv run cx generate --case cx_output/onboarding-smoke/fuzz_case.yaml --output cx_output/onboarding-generated
```

Review `generated_cases.json` and `generated_cases.yaml` before running a larger campaign.

If the builtin executor is not enough, load the template executor as a plugin:

```bash
uv run cx run --case examples/operator-onboarding/case_template.yaml --nodes examples/nodes/local.yaml --task run --scheduler local --plugin examples/operator-onboarding/executor_template.py --output cx_output/onboarding-plugin-smoke
```

The onboarding guide in `docs/operator-onboarding.md` describes when to add fuzz coverage, invalid cases, and Ray or accelerator templates. See `docs/cli.md` for the full `cx` command reference.
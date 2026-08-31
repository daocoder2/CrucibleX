# Operator Onboarding

Use this checklist when adding a new operator to CrucibleX. The goal is to make the operator reproducible before expanding coverage. For command-level inputs and outputs, see `docs/cli.md`.

## 1. Capture Operator Facts

Start from `examples/operator-onboarding/operator_facts.yaml` and record. The example directory also includes a short README that points at the scaffold files and smoke commands:

- `examples/operator-onboarding/README.md`

- public API name and invocation style
- supported backends
- parameter names, kinds, dtype families, shape rules, and value ranges
- expected invalid behavior, including expected errors when known
- comparison policy and tolerances
- fuzz intent when you want the scaffold to emit a seed-driven fuzz case

Generate the initial scaffold from those facts:

```bash
uv run cx onboard --facts examples/operator-onboarding/operator_facts.yaml --output cx_output/onboarding-smoke
```

This writes `case.yaml`, `fuzz_case.yaml`, `nodes.yaml`, `hardware_nodes.yaml`, `campaign.yaml`, `hardware_campaign.yaml`, `executor_plugin.py`, `validate.sh`, `hardware_validate.sh`, and a local README under the output directory. The generated local campaign runs both the baseline run smoke and the seed-driven fuzz smoke, using any fuzz overrides declared in the facts file. Hardware templates are generated from `operator.supported_backends` for Ray workers, but `validate.sh` does not execute them by default.

`hardware_nodes.yaml` and `hardware_campaign.yaml` are templates for later Ray or accelerator verification; they are intended for environments that actually expose the listed backends.

## 2. Create A Minimal Case

Copy `examples/operator-onboarding/case_template.yaml` and keep the first case small. Validate the simple path before adding generated coverage.

```bash
uv run cx run --case examples/operator-onboarding/case_template.yaml --nodes examples/nodes/local.yaml --task run --scheduler local --output cx_output/onboarding-smoke
```

## 3. Add Custom Execution Only When Needed

If the operator cannot be called by the builtin function executor, copy `examples/operator-onboarding/executor_template.py` and load it with `--plugin`.

```bash
uv run cx run --case examples/operator-onboarding/case_template.yaml --nodes examples/nodes/local.yaml --task run --scheduler local --plugin examples/operator-onboarding/executor_template.py --output cx_output/onboarding-plugin-smoke
```

## 4. Expand Generation Gradually

Add one generation feature at a time:

- `generation.count` for bounded expansion
- `generation.constraints: [linked_parameters]` for copied shape or dtype relationships
- `generation.constraints: [boundary_coverage]` for boundary dtype/shape/value rotation
- `generation.invalid_count` plus `value_range.invalid` for expected-invalid cases
- `generation.constraints: [random_coverage]` plus facts-level `fuzz.random_dtypes`, `fuzz.random_shapes`, and `fuzz.random_values` for seed-driven fuzz coverage
- facts-level `fuzz.invalid_count`, `fuzz.max_elements`, and `fuzz.max_bytes` to bound generated fuzz cases before promotion

Every generated run writes `generated_cases.json` for review. For a repeatable batch of runs, use a campaign file:

```bash
uv run cx campaign --campaign examples/campaigns/local-fuzz.yaml --output cx_output/campaign
uv run cx report --output cx_output/campaign
```

For a single fuzz smoke, run:

```bash
uv run cx run --case examples/cases/torch.abs.fuzz.yaml --nodes examples/nodes/local.yaml --task fuzz --scheduler local --output cx_output/local-fuzz-smoke
```

## 5. Validate Reports And Repro

For runs with failures, create a repro bundle from clustered failures.

```bash
uv run cx repro --output cx_output/onboarding-smoke --cluster-id cluster-0 --script
```

The bundle contains `rerun_commands` with `--plan-id` filters so each failing plan can be reproduced directly, and `--script` writes an executable shell wrapper per cluster.

## 6. Promotion Checklist

Before promoting an operator case set:

- local CPU smoke passes or expected-invalid failures are understood
- `generated_cases.json` contains the intended case count and constraints
- `postprocess.json` has no unexplained failure clusters
- `cx repro` emits rerun commands for any remaining failures
- Ray CPU/GPU smoke passes when those backends are in scope
- NPU, ACLNN, or DCU runs are validated only on workers exposing matching resources

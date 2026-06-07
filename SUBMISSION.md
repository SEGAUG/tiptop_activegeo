# MolmoSpaces Submission

- policy name: TiPToP-ActiveGeometry
- model type: TAMP + FMs
- embodiment: Franka/DROID-style benchmark robot
- action space: Joint Pos.
- MolmoBot or in-domain data used: No
- source repo: modified TiPToP repository in this workspace
- MolmoSpaces commit: b4ed488c4e8cfb9a4abff177a21787ddc22f922c
- TiPToP commit: e75c1c1dd0c73fcfc7828c0a65365af1243be2fb
- VLM/API backend: Qwen/Tongyi Qianwen via environment-provided API key; no API key is hardcoded.

## Smoke Test

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
bash scripts/molmospaces/run_smoke.sh
```

The smoke script evaluates episode `--idx 0` on the default `pick_v15` benchmark and writes results under:

```text
/data/data2/jinhui.lin/code/aicode/molmospaces_runs/smoke_pick_v15
```

## Selected Benchmarks

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
IDX_ARG="--idx 0" bash scripts/molmospaces/run_selected_benchmarks.sh
```

To run only a subset:

```bash
bash scripts/molmospaces/run_selected_benchmarks.sh pick_v15 pnp_v2
```

Supported environment overrides:

- `WORK`
- `MLSPACES_ASSETS_DIR`
- `OUT`
- `POLICY_CONFIG`
- `TASK_HORIZON`
- `IDX_ARG`

## Export CSV

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
bash scripts/molmospaces/export_results.sh
```

CSV files are written to:

```text
/data/data2/jinhui.lin/code/aicode/molmospaces_submission_csv
```

The script also creates:

```text
/data/data2/jinhui.lin/code/aicode/TiPToP-ActiveGeometry_molmospaces_results_<timestamp>.zip
```

## Known Limitations

- The adapter uses MolmoSpaces Franka joint-position commands and never emits real Piper control commands.
- Live TiPToP perception and TAMP planning are not enabled by default because they require configured external perception/planning services and calibrated runtime paths.
- The modified TiPToP perception stack uses the Qwen/Tongyi Qianwen API path when live perception is enabled; set the API key through the environment rather than storing it in code.
- If no executable TiPToP plan is available, the policy logs the failure and returns a hold-current joint-position action with a safe gripper command.
- A serialized TiPToP plan can be replayed through `policy_config.serialized_plan_path`; trajectory steps are split into per-step MolmoSpaces actions by `tiptop_molmospaces.action_queue.ActionQueue`.
- Observation schemas are saved to `logs/observation_schema.json` on the first policy step of each episode.

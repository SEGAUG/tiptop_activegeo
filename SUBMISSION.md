# MolmoSpaces Submission

- policy name: TiPToP-ActiveGeometry
- model type: TAMP + FMs
- embodiment: Franka/DROID-style benchmark robot
- action space: Joint Pos.
- MolmoBot or in-domain data used: No
- source repo: modified TiPToP repository in this workspace
- MolmoSpaces commit: b4ed488c4e8cfb9a4abff177a21787ddc22f922c
- TiPToP original fork base commit: e75c1c1dd0c73fcfc7828c0a65365af1243be2fb
- Uploaded Qwen/MolmoSpaces source commit: 53aa4f462adb17ce5cf7188b092698242b9497f0
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

## Observation Interface

The current MolmoSpaces smoke observation exposes two RGB cameras, `exo_camera_1`
and `wrist_camera`, both observed as `uint8` images with shape `352x624x3`.
The observation also exposes Franka arm qpos with shape `(7,)`, gripper/base
state under `qpos`, task language through `task_info`, and camera calibration
metadata through `sensor_param_exo_camera_1` and `sensor_param_wrist_camera`
including `intrinsic_cv`.

The policy observation does not currently expose depth images. Live planning
therefore records `no_depth_available` under `logs/molmospaces_live/<episode>/`
and does not use benchmark JSON object poses, target poses, or other privileged
simulator state to construct a scene. Qwen/Tongyi Qianwen remains the configured
VLM backend for live perception, but the current smoke path blocks before VLM,
SAM/M2T2, and cuTAMP because non-privileged depth is unavailable.

## Live Smoke

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
bash scripts/molmospaces/run_smoke_live.sh
python scripts/molmospaces/summarize_live_metrics.py
```

`run_smoke_live.sh` sets `TIPTOP_ENABLE_LIVE_PLANNING=1`,
`TIPTOP_REQUIRE_PLAN=1`, and `TIPTOP_VLM_BACKEND=qwen`. A planning failure still
returns a hold-current joint-position action for evaluator compatibility, but
the failure is written to `planning_failure.json` and
`episode_metrics.jsonl`.

Latest live smoke result on `pick_v15 --idx 0`: live planning was attempted once,
depth was not available in the policy observation, `planning_success=0`,
`fallback_reason=no_depth_available`, and no non-hold action was produced.

Latest rendered-depth smoke result with `TIPTOP_ENABLE_RENDERED_DEPTH=1` and
`TiPToPDepthEvalConfig`: MolmoSpaces exposed `exo_camera_1_depth` and
`wrist_camera_depth`; TiPToP saved depth `.npy` files, created xyz maps for both
cameras, and moved the blocker to `qwen_detection_not_yet_integrated`.

## Known Limitations

- The adapter uses MolmoSpaces Franka joint-position commands and never emits real Piper control commands.
- Live TiPToP perception and TAMP planning are enabled only when `TIPTOP_ENABLE_LIVE_PLANNING=1`.
- Rendered camera depth is disabled by default and enabled only with `TIPTOP_ENABLE_RENDERED_DEPTH=1` or `TiPToPDepthEvalConfig`; maintainer confirmation is needed before using it for final leaderboard scoring.
- The modified TiPToP perception stack uses the Qwen/Tongyi Qianwen API path when live perception is enabled; set the API key through the environment rather than storing it in code.
- If no executable TiPToP plan is available, the policy logs the failure and returns a hold-current joint-position action with a safe gripper command.
- A serialized TiPToP plan can be replayed through `policy_config.serialized_plan_path`; trajectory steps are split into per-step MolmoSpaces actions by `tiptop_molmospaces.action_queue.ActionQueue`.
- Observation schemas are saved to `logs/observation_schema.json` on the first policy step of each episode.

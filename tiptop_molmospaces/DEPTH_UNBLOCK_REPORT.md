# MolmoSpaces Depth Unblock Report

## Summary

MolmoSpaces already supports metric camera depth through its sensor system. The
policy observation did not contain depth in the TiPToP smoke run because the
`pick_v15` JSON benchmark camera specs set `record_depth: false` for both
`wrist_camera` and `exo_camera_1`.

## Findings

1. Default policy observation has no depth because `JsonEvalTaskSampler` replays
   the benchmark JSON camera specs as authoritative camera config, and
   `molmo_spaces.env.sensors.get_core_sensors()` only adds `DepthSensor` when a
   camera config has `record_depth=True`.

2. MolmoSpaces has built-in depth support:
   - `molmo_spaces.configs.camera_configs.CameraConfig.record_depth`
   - `molmo_spaces.env.sensors_cameras.DepthSensor`
   - `molmo_spaces.env.env.CPUMujocoEnv.render_depth_frame(camera_name)`
   - HDF5/video save utilities for sensors ending in `_depth`

3. There is no general `eval_main.py` CLI flag found for enabling recorded
   benchmark camera depth. The JSON schema contains `record_depth`, and the CAP
   robot evaluation override turns it on for CAP-specific evaluation, but normal
   `FrankaRobotConfig` replay keeps the benchmark values.

4. The smallest non-privileged patch is to opt in from the external TiPToP config
   and set `record_depth=True` on the replayed policy cameras before
   `get_core_sensors()` constructs sensors. This uses MolmoSpaces' current
   camera renderer and does not use benchmark object poses, target poses, success
   oracles, or privileged simulator object state.

5. The depth source is camera render depth from the same policy cameras
   (`exo_camera_1`, `wrist_camera`) that already provide RGB and intrinsics.
   It is a camera sensor modality, not an object-state oracle.

6. Leaderboard fairness is a policy question. Technically, MolmoSpaces includes
   camera depth sensors and some configs already use `record_depth=True`, but the
   public benchmark JSON for this smoke task disables it. The adapter keeps depth
   off by default and enables it only with `TIPTOP_ENABLE_RENDERED_DEPTH=1` or
   `TiPToPDepthEvalConfig`. Maintainer confirmation is still needed before using
   rendered depth for final leaderboard scoring.

## Implementation Route

Recommended route: Route B with a minimal external adapter patch.

- `tiptop_molmospaces.configs.TiPToPDepthEvalConfig` installs a guarded wrapper
  around `JsonEvalTaskSampler._build_camera_config_from_spec`.
- When `TIPTOP_ENABLE_RENDERED_DEPTH=1`, the wrapper sets `record_depth=True`
  for `policy_config.camera_names`.
- MolmoSpaces then adds normal `<camera>_depth` observations via its own
  `DepthSensor`.
- `tiptop_molmospaces.policy.TiPToPPolicy` extracts `<camera>_depth`, saves
  depth `.npy` files, creates camera-frame xyz maps with TiPToP
  `depth_to_xyz()`, and writes `depth_summary.json`.

## Current Live Depth Blocker

With depth enabled, the expected first milestone is:

- `depth_available=True`
- `xyz_map_created=True`
- fallback reason moves beyond `no_depth_available`

The current adapter then records `qwen_detection_not_yet_integrated`, because
the next remaining integration step is wiring Qwen detection/masks into the
existing SAM/M2T2/cuTAMP live planning path without privileged state.

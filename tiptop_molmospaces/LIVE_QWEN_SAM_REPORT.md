# Live Qwen/SAM/Object Cloud Report

## Observation Inputs

- Task text is available from the policy task object and is written as
  `task_text` / `task_or_language` in `observation_summary.json`. The current
  smoke task text is `Pick up the shiny silver kettle`.
- RGB images are `uint8` arrays with shape `352x624x3`.
- Depth images are `float32` arrays with shape `352x624`.
- Depth and RGB are aligned by camera name: `exo_camera_1` with
  `exo_camera_1_depth`, and `wrist_camera` with `wrist_camera_depth`.
- XYZ maps are currently camera-frame XYZ maps created from depth plus
  `sensor_param_<camera>.intrinsic_cv`. They are saved as `xyz_<camera>.npy`.
- MolmoSpaces also exposes camera calibration metadata including
  `intrinsic_cv`, `extrinsic_cv`, and `cam2world_gl` in the live observation.
  The current object-cloud milestone only needs camera-frame clouds; world-frame
  conversion should use these sensor params before M2T2/cuTAMP integration.

## Qwen Grounding

`tiptop.perception.qwen_vl.qwen_detect_and_translate_raw()` prompts Qwen to
return JSON bboxes as normalized `0-1000` `[xmin, ymin, xmax, ymax]`. The live
perception adapter normalizes these to internal pixel `bbox_xyxy` boxes and
also keeps `[ymin, xmin, ymax, xmax]` normalized boxes for the existing SAM2
wrapper.

## SAM2

The current public SAM wrapper is `tiptop.perception.sam2.sam2_segment_objects`.
It accepts PIL RGB plus detections whose `box_2d` field is normalized
`[ymin, xmin, ymax, xmax]`, converts those to pixel boxes, and returns masks with
shape equivalent to `N x 1 x H x W`.

## Object Cloud

The adapter applies each SAM mask to the primary camera XYZ map, filters NaN,
Inf, and all-zero points, then saves `object_clouds/*.npy` and
`object_geometry_summary.json`. A successful perception milestone requires:

- `qwen_success=true`
- `sam_success=true`
- `object_cloud_created=true`
- `target_object_points > 0`

## Current Next Blocker

Once object clouds are created, the policy deliberately stops with
`m2t2_cutamp_not_yet_integrated`. The next implementation layer should transform
camera/world geometry as needed and feed object point clouds into M2T2 and then
cuTAMP/cuRobo.

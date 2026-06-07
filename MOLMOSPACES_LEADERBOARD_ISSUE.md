# MolmoSpaces Benchmark Entry: TiPToP-ActiveGeometry

policy name:
TiPToP-ActiveGeometry

authors/institution name:
Jinhui Lin / MIT

MolmoBot or other in-domain data used:
No

multi-task:
Yes

data type (real/sim/mix):
Sim

robot generalist:
Yes

data opensourced:
No

model opensourced:
No

source link:
https://github.com/SEGAUG/tiptop_activegeo

model type:
TAMP + FMs

embodiment:
Any

action space:
Joint Pos.

no. of params:
N/A

data released:
N/A

chunk size:
N/A

context length:
N/A

Results zip:
Attach or link:
https://github.com/SEGAUG/tiptop_activegeo/releases/download/tiptop_activegeo_qwen_molmospaces_v1/TiPToP-ActiveGeometry_molmospaces_results_20260606_191953.zip

Current result scope:
Smoke/inspection only. The zip contains `smoke_pick_v15.csv` and `inspect_observation_latest.csv`; it is not a full benchmark leaderboard score package.

## Reproducibility

MolmoSpaces commit:
b4ed488c4e8cfb9a4abff177a21787ddc22f922c

TiPToP original fork base commit:
e75c1c1dd0c73fcfc7828c0a65365af1243be2fb

Uploaded Qwen/MolmoSpaces source commit:
53aa4f462adb17ce5cf7188b092698242b9497f0

Run command:
See `scripts/molmospaces/run_selected_benchmarks.sh`

Export command:
See `scripts/molmospaces/export_results.sh`

Notes:
- Adapter smoke test `--idx 0` completed and saved MolmoSpaces h5/mp4 outputs.
- Adapter pytest passed.
- Observation schema was saved to `logs/observation_schema.json`; the observed MolmoSpaces camera keys were `exo_camera_1` and `wrist_camera`, both RGB `352x624x3`.
- The current generated zip contains smoke/inspection CSVs only. Run selected or full benchmarks before using this as a final leaderboard score submission.
- The adapter uses MuJoCo filament for benchmark execution.
- The modified TiPToP live perception path uses Qwen/Tongyi Qianwen API credentials supplied through environment variables. No API key is hardcoded.
- The default smoke configuration falls back to hold-current joint-position actions when no executable TiPToP plan is available.

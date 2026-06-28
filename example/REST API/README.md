# REST API examples (curl)

One bash script per cnnaio endpoint, each issuing a `curl` request with **every
applicable option field populated at its default value** — so you can see the
full request shape at a glance and tweak from there.

## Setup

Start the server (dev mode ships with `no_auth: true`, so no token needed):

```sh
# from the repo root
go run . -dev          # API + web UI at http://localhost:8080
```

Then run any script (from this folder):

```sh
cd "example/REST API"
./03_detect_yolo.sh
./run_all.sh           # or run every script in order
```

Output is pretty-printed with [`jq`](https://jqlang.github.io/jq/) if installed,
otherwise raw JSON.

### Configuration (env vars)

All scripts source [`_common.sh`](_common.sh); override via the environment:

| Var | Default | Meaning |
|-----|---------|---------|
| `BASE_URL` | `http://localhost:8080` | Server base URL |
| `CNNAIO_KEY` | _(unset)_ | Bearer token — only needed when auth is enabled |
| `IMAGE` | `../testdata/test.png` | Image sent by the scripts |
| `IMAGE_A`, `IMAGE_B` | `$IMAGE` | The two images for face comparison |

```sh
BASE_URL=http://10.0.0.5:8080 CNNAIO_KEY=cxn-... IMAGE=~/cat.jpg ./03_detect_yolo.sh
```

## Scripts

| Script | Endpoint | Model (default) |
|--------|----------|-----------------|
| [`01_classify_mobilenet.sh`](01_classify_mobilenet.sh) | `POST /v1/images/classifications` | `mobilenet-v2` |
| [`02_classify_yolo.sh`](02_classify_yolo.sh) | `POST /v1/images/classifications` | `yolo11n-cls` |
| [`03_detect_yolo.sh`](03_detect_yolo.sh) | `POST /v1/images/detections` | `yolo11n` |
| [`04_detect_nanodet.sh`](04_detect_nanodet.sh) | `POST /v1/images/detections` | `nanodet-plus-m` |
| [`05_segment.sh`](05_segment.sh) | `POST /v1/images/segmentations` | `yolo11n-seg` |
| [`06_pose.sh`](06_pose.sh) | `POST /v1/images/poses` | `yolo11n-pose` |
| [`07_oriented.sh`](07_oriented.sh) | `POST /v1/images/oriented` | `yolo11n-obb` |
| [`08_face_detect.sh`](08_face_detect.sh) | `POST /v1/faces/detections` | `ultraface-rfb-320` |
| [`09_face_landmarks.sh`](09_face_landmarks.sh) | `POST /v1/faces/landmarks` | `pfld` |
| [`10_face_embedding.sh`](10_face_embedding.sh) | `POST /v1/faces/embeddings` | `mbv2facenet` |
| [`11_face_comparison.sh`](11_face_comparison.sh) | `POST /v1/faces/comparisons` | `mbv2facenet` |
| [`12_gender.sh`](12_gender.sh) | `POST /v1/faces/gender` | `gender-mbv2-0.35` |
| [`13_vision_analyze.sh`](13_vision_analyze.sh) | `POST /v1/vision/analyze` | _(multiple tasks)_ |
| [`14_models_list.sh`](14_models_list.sh) | `GET /v1/models` | — |
| [`15_health.sh`](15_health.sh) | `GET /v1/health` | — |
| [`16_preview_png.sh`](16_preview_png.sh) | `POST /v1/images/detections?preview` | annotated PNG out |
| [`17_async_job.sh`](17_async_job.sh) | `POST … async=true` + `GET /v1/jobs/{id}` | submit + poll |

## Default option values

These are the per-task defaults the scripts populate (the server applies the
same defaults if you omit them):

| Field | Applies to | Default |
|-------|------------|---------|
| `top_k` | classification | `5` |
| `score_threshold` | detect / seg / pose / obb | `0.25` (nanodet `0.4`) |
| `nms_threshold` | detect / seg / pose / obb | `0.45` (nanodet `0.5`) |
| `score_threshold` | faces / landmarks | `0.7` |
| `nms_threshold` | faces / landmarks | `0.3` |
| `max_results` | detection-like | `100` |
| `cropped` | landmarks / embeddings / gender | `false` |
| `threshold` | face comparison | `0.5` |
| `render` | all | `false` |
| `async` | all | `false` |

> **JSON vs multipart.** Single-image scripts use multipart (`-F`) so each option
> is one readable form field. Face comparison and vision analyze carry multiple
> images / a tasks array, so they POST a JSON body (image as a base64 data URI).

Full reference: [`../../docs/API.md`](../../docs/API.md).

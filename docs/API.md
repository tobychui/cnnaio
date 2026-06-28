# cnnaio Vision Server — REST API

The HTTP surface lives in `main.go` + `mod/api`. It deliberately mirrors the
**OpenAI API** conventions so the server feels familiar and is easy to wrap with
existing client patterns:

- Versioned base path: `/v1/...`
- Bearer-token auth: `Authorization: Bearer <API_KEY>`
- JSON request/response, `snake_case` fields
- A `model` field selects the concrete model where several exist
- A consistent response envelope with `object`, `model`, `created`, `data`
- OpenAI-style error envelope
- Resource-style, task-specific endpoints

---

## 1. Concepts & capabilities

Every endpoint runs one of the bundled models on the shared pure-Go
wazero/ncnn runtime. Capabilities map 1:1 onto the `mod/*` packages:

| Capability            | Endpoint                          | Backing module(s)            |
|-----------------------|-----------------------------------|------------------------------|
| Image classification  | `POST /v1/images/classifications` | `mobilenet`, `yolo11` (cls)  |
| Object detection      | `POST /v1/images/detections`      | `yolo11`, `nanodet`          |
| Instance segmentation | `POST /v1/images/segmentations`   | `yolo11` (seg)               |
| Pose estimation       | `POST /v1/images/poses`           | `yolo11` (pose)              |
| Oriented detection    | `POST /v1/images/oriented`        | `yolo11` (obb)               |
| Face detection        | `POST /v1/faces/detections`       | `facedetector`               |
| Facial landmarks      | `POST /v1/faces/landmarks`        | `landmark` (PFLD)            |
| Face embedding        | `POST /v1/faces/embeddings`       | `facerecognition`            |
| Face comparison       | `POST /v1/faces/comparisons`      | `facerecognition`            |
| Gender classification | `POST /v1/faces/gender`           | `genderdetector`             |
| Combined analysis     | `POST /v1/vision/analyze`         | any of the above             |
| List models           | `GET  /v1/models`                 | —                            |
| Model detail          | `GET  /v1/models/{id}`            | —                            |
| Async job status      | `GET  /v1/jobs/{id}`             | —                            |
| Health                | `GET  /v1/health`                 | —                            |

---

## 2. Authentication

```
Authorization: Bearer <API_KEY>
```

Tokens are generated with the `-nt` flag and stored in `./token/tokens.json`
(each looks like `cxn-<hex>`). Missing/invalid keys return `401`. Auth can be
disabled wholesale by setting `"no_auth": true` in `conf/config.json`
(local/trusted deployments — the bundled config ships with this on). `GET
/v1/health` is always public. See [§12](#12-running-the-server).

---

## 3. Image input

Every recognition endpoint accepts the image in one of two encodings:

### 3a. JSON body (default)
`Content-Type: application/json`. The `image` field is one of:

| Form              | Example                                  |
|-------------------|------------------------------------------|
| Data URI (base64) | `"data:image/jpeg;base64,/9j/4AAQ..."`   |
| Bare base64       | `"/9j/4AAQSkZJRg..."`                     |

```json
{ "model": "yolo11n", "image": "data:image/jpeg;base64,...", "score_threshold": 0.25 }
```

> Remote image **URLs are not supported** (avoids SSRF); send base64 or upload
> the file via multipart.

### 3b. Multipart upload
`Content-Type: multipart/form-data` with an `image` file part; other parameters
are sent as form fields.

```sh
curl .../v1/images/detections -H "Authorization: Bearer $KEY" \
  -F image=@cat.jpg -F model=yolo11n -F score_threshold=0.25
```

Supported formats: JPEG, PNG, BMP, GIF (anything `stb_image` decodes), plus
**WebP** (lossy + lossless), transcoded to PNG server-side by a pure-Go decoder —
no ffmpeg/libwebp needed. (Animated/extended WebP the pure-Go decoder can't read
returns a `400`.) Max upload size is configurable (default 10 MB — see §12).

---

## 4. Common parameters

| Field             | Type    | Default          | Applies to        | Meaning                                  |
|-------------------|---------|------------------|-------------------|------------------------------------------|
| `model`           | string  | per-task default | all               | Which model to run (see `GET /v1/models`). |
| `score_threshold` | number  | task default     | detect/seg/pose/obb/faces/landmarks | Min confidence to keep.       |
| `nms_threshold`   | number  | task default     | detect/seg/pose/obb/faces/landmarks | IoU suppression threshold.    |
| `top_k`           | integer | `5`              | classification    | Number of ranked results.                |
| `max_results`     | integer | `100`            | detection-like    | Cap on returned items.                   |
| `cropped`         | boolean | `false`          | landmarks/embeddings/gender | Treat the whole image as one pre-cropped face. |
| `render`          | boolean | `false`          | all               | Also return an annotated PNG (see §6).   |
| `async`           | boolean | `false`          | all               | Run as a background job (see §8).         |

Per-task threshold defaults: detection/seg/pose/obb `score 0.25` / `nms 0.45`
(NanoDet `0.4` / `0.5`); faces & landmarks `score 0.7` / `nms 0.3`.

---

## 5. Common response envelope

```json
{
  "object": "image.detection",
  "model": "yolo11n",
  "created": 1719150000,
  "image": { "width": 1920, "height": 1080 },
  "timing_ms": 41,
  "data": [ /* task-specific items, see each endpoint */ ],
  "rendered_image": null
}
```

- `object` — `image.classification` | `image.detection` | `image.segmentation` |
  `image.pose` | `image.oriented` | `face.detection` | `face.landmarks` |
  `face.embedding` | `face.comparison` | `face.gender` | `vision.analysis`
- `image` — decoded source dimensions in pixels (coordinates below are in this space)
- `rendered_image` — present only when `render: true` (a `data:image/png;base64,...` URI)

All bounding boxes use absolute (integer) pixel coordinates in the original
image: `box = { "x1": .., "y1": .., "x2": .., "y2": .. }`, origin top-left.

---

## 6. Rendering & preview (optional)

Two ways to get an annotated image (boxes/labels, landmark or keypoint dots, pose
skeletons, oriented polygons, segmentation masks):

- **In JSON** — add `"render": true`; the response gains a `rendered_image`
  data URI alongside the normal `data`.
- **Raw PNG** — add `?preview` (or `?preview=true`, or header `Accept:
  image/png`) to get the annotated **PNG bytes** directly instead of JSON. Single
  image only, and not combinable with `async`.

```json
"rendered_image": "data:image/png;base64,iVBORw0KGgo..."
```

---

## 7. Endpoints

Every example uses a JSON body with the image as a base64 data URI (abbreviated
to `...`); the same calls work with a multipart `image` upload (see §3b). All
coordinates are absolute integer pixels in the source image.

### 7.1 Classification — `POST /v1/images/classifications`
Models: `mobilenet-v2` (ImageNet-1000, default), `yolo11n-cls` (ImageNet-1000).

**Request**
```json
{ "model": "mobilenet-v2", "image": "data:image/jpeg;base64,...", "top_k": 5 }
```
**Response** — `data[]` item: `{ "label": string, "index": int, "score": 0..1 }`.
```json
{
  "object": "image.classification",
  "model": "mobilenet-v2",
  "created": 1719150000,
  "image": { "width": 800, "height": 533 },
  "timing_ms": 385,
  "data": [
    { "label": "'n03770439 miniskirt, mini'", "index": 655, "score": 0.1956 },
    { "label": "'n03450230 gown'",            "index": 578, "score": 0.1418 }
  ]
}
```

### 7.2 Object detection — `POST /v1/images/detections`
Models: `yolo11n` (default, COCO-80), `nanodet-plus-m` (COCO-80, smaller).

**Request**
```json
{ "model": "yolo11n", "image": "data:image/jpeg;base64,...",
  "score_threshold": 0.25, "nms_threshold": 0.45, "max_results": 100 }
```
**Response** — `data[]` item: `{ label, class_id, score, box }`.
```json
{
  "object": "image.detection",
  "model": "yolo11n",
  "created": 1719150000,
  "image": { "width": 800, "height": 535 },
  "timing_ms": 41,
  "data": [
    { "label": "person", "class_id": 0,  "score": 0.81, "box": { "x1": 110, "y1": 85, "x2": 591, "y2": 535 } },
    { "label": "chair",  "class_id": 56, "score": 0.56, "box": { "x1": 0,   "y1": 332, "x2": 117, "y2": 472 } }
  ]
}
```

### 7.3 Segmentation — `POST /v1/images/segmentations`
Model: `yolo11n-seg`. Each item is a detection plus a **per-instance** mask: an
8-bit grayscale PNG (255 = object), box-cropped, plus its placement origin.

**Request**
```json
{ "model": "yolo11n-seg", "image": "data:image/jpeg;base64,...",
  "score_threshold": 0.25, "nms_threshold": 0.45 }
```
**Response**
```json
{
  "object": "image.segmentation",
  "model": "yolo11n-seg",
  "created": 1719150000,
  "image": { "width": 800, "height": 535 },
  "timing_ms": 2174,
  "data": [
    { "label": "person", "class_id": 0, "score": 0.86,
      "box":  { "x1": 114, "y1": 84, "x2": 590, "y2": 535 },
      "mask": { "encoding": "png", "width": 476, "height": 451,
                "origin": { "x": 114, "y": 84 }, "data": "iVBORw0KGgo..." } }
  ]
}
```

### 7.4 Pose — `POST /v1/images/poses`
Model: `yolo11n-pose`. 17 named COCO keypoints per person (keypoint = `{name,x,y}`).

**Request**
```json
{ "model": "yolo11n-pose", "image": "data:image/jpeg;base64,...",
  "score_threshold": 0.25, "nms_threshold": 0.45 }
```
**Response**
```json
{
  "object": "image.pose",
  "model": "yolo11n-pose",
  "created": 1719150000,
  "image": { "width": 800, "height": 535 },
  "timing_ms": 1779,
  "data": [
    { "score": 0.83,
      "box": { "x1": 88, "y1": 86, "x2": 694, "y2": 535 },
      "keypoints": [
        { "name": "nose",      "x": 268, "y": 196 },
        { "name": "left_eye",  "x": 252, "y": 180 }
        /* … 17 total: nose, left/right eye, ears, shoulders, elbows,
           wrists, hips, knees, ankles … */
      ] }
  ]
}
```

### 7.5 Oriented detection (OBB) — `POST /v1/images/oriented`
Model: `yolo11n-obb` (DOTA-15). **Aerial/top-down imagery only** (expect no
results on ordinary photos). Boxes are rotated, given as a 4-point polygon.

**Request**
```json
{ "model": "yolo11n-obb", "image": "data:image/jpeg;base64,...",
  "score_threshold": 0.25, "nms_threshold": 0.45 }
```
**Response**
```json
{
  "object": "image.oriented",
  "model": "yolo11n-obb",
  "created": 1719150000,
  "image": { "width": 1024, "height": 1024 },
  "timing_ms": 1692,
  "data": [
    { "label": "ship", "class_id": 1, "score": 0.77, "angle_rad": 0.31,
      "polygon": [ {"x":120,"y":80}, {"x":210,"y":110}, {"x":190,"y":175}, {"x":100,"y":145} ] }
  ]
}
```

### 7.6 Face detection — `POST /v1/faces/detections`
Models: `ultraface-rfb-320` (default), `ultraface-slim-320` (faster).

**Request**
```json
{ "model": "ultraface-rfb-320", "image": "data:image/jpeg;base64,...",
  "score_threshold": 0.7, "nms_threshold": 0.3 }
```
**Response** — `data[]` item: `{ score, box }`.
```json
{
  "object": "face.detection",
  "model": "ultraface-rfb-320",
  "created": 1719150000,
  "image": { "width": 800, "height": 535 },
  "timing_ms": 116,
  "data": [
    { "score": 0.83, "box": { "x1": 161, "y1": 159, "x2": 368, "y2": 351 } }
  ]
}
```

### 7.7 Facial landmarks — `POST /v1/faces/landmarks`
Model: `pfld`. Detects faces, returns 98 points each. To run on a known crop,
set `"cropped": true` and the whole image is treated as one face.

**Request**
```json
{ "model": "pfld", "image": "data:image/jpeg;base64,...", "cropped": false }
```
**Response**
```json
{
  "object": "face.landmarks",
  "model": "pfld",
  "created": 1719150000,
  "image": { "width": 800, "height": 535 },
  "timing_ms": 322,
  "data": [
    { "box": { "x1": 161, "y1": 159, "x2": 368, "y2": 351 },
      "points": [ { "x": 167, "y": 223 }, { "x": 169, "y": 240 } /* … 98 total */ ] }
  ]
}
```

### 7.8 Face embedding — `POST /v1/faces/embeddings`
Model: `mbv2facenet`. Returns an L2-normalized 128-d vector per face. By default
embeds the largest face in a photo; `"cropped": true` treats the input as a face.

**Request**
```json
{ "model": "mbv2facenet", "image": "data:image/jpeg;base64,...", "cropped": false }
```
**Response**
```json
{
  "object": "face.embedding",
  "model": "mbv2facenet",
  "created": 1719150000,
  "image": { "width": 800, "height": 535 },
  "timing_ms": 499,
  "data": [
    { "box": { "x1": 161, "y1": 159, "x2": 368, "y2": 351 },
      "embedding": [ -0.0233, 0.0481, -0.0117 /* … 128 floats */ ], "dim": 128 }
  ]
}
```

### 7.9 Face comparison — `POST /v1/faces/comparisons`
JSON only (it carries two images). Returns cosine similarity. Each image may be a
full photo (auto-cropped to its largest face) or an already-cropped face.

**Request**
```json
{ "model": "mbv2facenet",
  "image_a": "data:image/jpeg;base64,...", "image_b": "data:image/jpeg;base64,...",
  "a_cropped": false, "b_cropped": false, "threshold": 0.5 }
```
**Response**
```json
{ "object": "face.comparison", "model": "mbv2facenet", "created": 1719150000,
  "similarity": 0.62, "same": true, "threshold": 0.5,
  "box_a": { "x1": 161, "y1": 159, "x2": 368, "y2": 351 },
  "box_b": { "x1": 88,  "y1": 102, "x2": 274, "y2": 300 } }
```

### 7.10 Gender classification — `POST /v1/faces/gender`
Model: `gender-mbv2-0.35`. Classifies the largest face (or `"cropped": true`).

**Request**
```json
{ "model": "gender-mbv2-0.35", "image": "data:image/jpeg;base64,...", "cropped": false }
```
**Response**
```json
{
  "object": "face.gender",
  "model": "gender-mbv2-0.35",
  "created": 1719150000,
  "image": { "width": 800, "height": 535 },
  "timing_ms": 156,
  "data": [
    { "box": { "x1": 161, "y1": 159, "x2": 368, "y2": 351 },
      "gender": { "label": "female", "confidence": 0.97,
                  "scores": { "female": 0.976, "male": 0.024 } } }
  ]
}
```

### 7.11 Combined analysis — `POST /v1/vision/analyze`
JSON only. Run several tasks over one image in a single round trip (decode once).
Per-task parameters go in `options` keyed by task name.

**Request**
```json
{ "image": "data:image/jpeg;base64,...",
  "tasks": [ "detect", "faces", "gender" ],
  "options": { "detect": { "model": "yolo11n", "score_threshold": 0.3 },
               "gender": {} },
  "render": true }
```
**Response**
```json
{ "object": "vision.analysis", "created": 1719150000,
  "image": { "width": 800, "height": 535 },
  "results": {
     "detect": { "object": "image.detection", "model": "yolo11n",
                 "data": [ { "label": "person", "class_id": 0, "score": 0.81,
                             "box": { "x1": 110, "y1": 85, "x2": 591, "y2": 535 } } ] },
     "faces":  { "object": "face.detection", "model": "ultraface-rfb-320",
                 "data": [ { "score": 0.83, "box": { "x1": 161, "y1": 159, "x2": 368, "y2": 351 } } ] },
     "gender": { "object": "face.gender", "model": "gender-mbv2-0.35",
                 "data": [ { "box": { "x1": 161, "y1": 159, "x2": 368, "y2": 351 },
                             "gender": { "label": "female", "confidence": 0.97,
                                         "scores": { "female": 0.976, "male": 0.024 } } } ] }
  },
  "rendered_image": "data:image/png;base64,iVBORw0KGgo..." }
```
Valid `tasks`: `classify`, `detect`, `segment`, `pose`, `oriented`, `faces`,
`landmarks`, `gender` (`attributes` is accepted as an alias for `gender`).
`embedding`/`comparison` are excluded — they need their own request shapes.

### 7.12 Models — `GET /v1/models`
OpenAI-style list of available models.

**Request**
```
GET /v1/models
Authorization: Bearer <API_KEY>
```
**Response**
```json
{ "object": "list",
  "data": [
    { "id": "yolo11n",          "object": "model", "task": "detection",      "classes": 80,   "input": 640 },
    { "id": "mobilenet-v2",     "object": "model", "task": "classification", "classes": 1000, "input": 224 },
    { "id": "gender-mbv2-0.35", "object": "model", "task": "gender",         "classes": 2,    "input": 64 }
  ] }
```
`GET /v1/models/{id}` returns one entry (`404` if unknown):
```json
{ "id": "yolo11n", "object": "model", "task": "detection", "classes": 80, "input": 640 }
```

### 7.13 Health — `GET /v1/health`
Always public (no auth).

**Request**
```
GET /v1/health
```
**Response**
```json
{ "status": "ok", "version": "0.1.0", "models_loaded": 12, "sessions": 4, "uptime_s": 1234 }
```

---

## 8. Batch & async

### Batch
Send `"images": [ ... ]` (JSON) or multiple `image` parts (multipart) to get an
`<object>.batch` envelope whose `data` is an array of per-image result envelopes.

**Request**
```json
{ "model": "yolo11n", "images": [ "data:image/jpeg;base64,...", "data:image/jpeg;base64,..." ] }
```
**Response**
```json
{ "object": "image.detection.batch", "created": 1719150000,
  "data": [
    { "object": "image.detection", "model": "yolo11n", "image": { "width": 800, "height": 535 }, "data": [ /* … */ ] },
    { "object": "image.detection", "model": "yolo11n", "image": { "width": 640, "height": 480 }, "data": [ /* … */ ] }
  ] }
```

### Async — submit a job, then poll
Add `"async": true` (or form field `async=true`) to any recognition request to
get `202` + a **job object**.

**Submit request**
```json
{ "model": "yolo11n", "image": "data:image/jpeg;base64,...", "async": true }
```
**Submit response** (`202 Accepted`)
```json
{ "id": "job-1a2b3c...", "object": "job", "status": "queued", "created": 1719150000 }
```

Then poll the job until it finishes — `GET /v1/jobs/{id}`:

**Request**
```
GET /v1/jobs/job-1a2b3c...
Authorization: Bearer <API_KEY>
```
**Response** (still running)
```json
{ "id": "job-1a2b3c...", "object": "job", "status": "running", "created": 1719150000 }
```
**Response** (on completion the job carries the `result`, or an `error`)
```json
{ "id": "job-1a2b3c...", "object": "job", "status": "succeeded", "created": 1719150000,
  "result": { "object": "image.detection", "model": "yolo11n", "data": [ /* … */ ] } }
```
Job statuses: `queued` → `running` → `succeeded` | `failed`. Jobs are in-memory
(lost on restart) — fine for local single-process use.

---

## 9. Errors

OpenAI-style envelope; HTTP status matches the error type.
```json
{ "error": {
    "message": "model 'yolo99' is not available",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found" } }
```
| HTTP | `type`                  | When                                   |
|------|-------------------------|----------------------------------------|
| 400  | `invalid_request_error` | bad params, undecodable image          |
| 401  | `authentication_error`  | missing/invalid API key                |
| 404  | `not_found_error`       | unknown model / job / endpoint         |
| 413  | `payload_too_large`     | image exceeds size limit               |
| 422  | `unprocessable_entity`  | e.g. no face found for a face endpoint  |
| 429  | `rate_limit_error`      | throttled (when `rate_limit_per_minute` > 0) |
| 500  | `server_error`          | inference/internal failure             |

---

## 10. Usage examples

### curl — detection (base64 JSON)
```sh
curl http://localhost:8080/v1/images/detections \
  -H "Authorization: Bearer $CNNAIO_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "model":"yolo11n", "score_threshold":0.25,
        "image":"data:image/jpeg;base64,/9j/4AAQ..." }'
```

### curl — multipart file + raw PNG preview
```sh
curl "http://localhost:8080/v1/images/poses?preview" \
  -H "Authorization: Bearer $CNNAIO_KEY" \
  -F image=@person.jpg -o pose.png
```

### Python (requests)
```python
import base64, requests

def b64(path):
    return "data:image/jpeg;base64," + base64.b64encode(open(path,"rb").read()).decode()

r = requests.post(
    "http://localhost:8080/v1/faces/comparisons",
    headers={"Authorization": f"Bearer {KEY}"},
    json={"image_a": b64("a.jpg"), "image_b": b64("b.jpg"), "threshold": 0.5},
)
res = r.json()
print(res["similarity"], "same" if res["same"] else "different")
```

### JavaScript (fetch)
```js
const res = await fetch(`${BASE}/v1/vision/analyze`, {
  method: "POST",
  headers: { "Authorization": `Bearer ${KEY}`, "Content-Type": "application/json" },
  body: JSON.stringify({
    image: dataUri,
    tasks: ["detect", "faces", "gender"],
    render: true,
  }),
});
const out = await res.json();
console.log(out.results.detect.data);                   // boxes
document.querySelector("img").src = out.rendered_image; // annotated preview
```

A full set of ready-to-run curl scripts (one per endpoint, all option fields
populated) lives in [`../example/REST API/`](../example/REST%20API/).

---

## 11. Server layout

```
main.go                   HTTP bootstrap + flags (-nt, -j, -addr, -config, -dev)
mod/api/config.go         conf/config.json load/defaults
mod/api/token.go          token generate/store/load (./token/tokens.json)
mod/api/pool.go           pool of -j ncnn.Sessions (inference concurrency)
mod/api/server.go         router + middleware (auth, CORS, rate limit)
mod/api/image.go          image-input decoding (base64/data-URI/multipart)
mod/api/respond.go        envelope + error helpers
mod/api/jobs.go           in-memory async job store
mod/api/models.go         model registry + resolution
mod/api/handlers_common.go  shared single-image pipeline (batch/async/preview)
mod/api/handlers_*.go     one file per capability group, calling the mod/* packages
mod/api/devui.go          optional developer web UI (-dev)
```
The model packages are unchanged; `mod/api` is a thin HTTP layer over a pool of
shared `ncnn.Session`s (each compiles the embedded wasm once).

---

## 12. Running the server

```sh
go run . -nt            # generate an API token (stored in ./token/), print it, exit
go run .                # start the server (conf/config.json, auto-created on first run)
go run . -j 4           # 4 inference sessions = up to 4 concurrent inferences
go run . -addr :9000    # override the listen address
go run . -config x.json # use a different config file
go run . -dev           # also serve the developer web UI at /
```

### Flags
| Flag       | Default              | Meaning                                                        |
|------------|----------------------|----------------------------------------------------------------|
| `-nt`      | —                    | Generate a new token, append to `./token/tokens.json`, print it, exit. |
| `-j`       | `1`                  | Number of ncnn inference sessions (pool size = concurrency). Each compiles the wasm once. |
| `-addr`    | (from config)        | Listen address override.                                       |
| `-config`  | `conf/config.json`   | Config file path (created with defaults if missing).           |
| `-dev`     | off                  | Serve the developer web UI (API tester + code generator) from `./web` at `/`. |
| `-webdir`  | `web`                | Directory served as the web UI in `-dev` mode.                 |

### Developer web UI (`-dev`)
`go run . -dev` serves an interactive tester at `http://localhost:<port>/`: pick a
task + model + parameters, upload a local image, and see the annotated PNG and JSON
response. It also live-generates **curl / fetch / jQuery** snippets for the current
request. The `/v1/*` API is served alongside it.

### `conf/config.json`
```json
{
  "listen": ":8080",
  "no_auth": false,
  "max_image_bytes": 10485760,
  "max_results": 100,
  "request_timeout_seconds": 30,
  "rate_limit_per_minute": 0,
  "cors_origins": ["*"],
  "default_models": {
    "classification": "mobilenet-v2",
    "detection": "yolo11n",
    "face_detection": "ultraface-rfb-320"
  }
}
```
- `no_auth: true` disables Bearer-token checks entirely (local/trusted use).
- `rate_limit_per_minute: 0` means unlimited (fixed-window global limiter otherwise).
- `cors_origins`: `["*"]` allows any origin; otherwise list specific origins.

### Tokens
`-nt` writes tokens to `./token/tokens.json` (a JSON array). Run it multiple
times to add more; any listed token is accepted. Treat the file as a secret.

### Notes
- Serves plain HTTP (no TLS yet — terminate TLS with a reverse proxy).
- Async jobs are in-memory (lost on restart); fine for local single-process use.

---

## 13. Conventions quick reference

- Coordinates: absolute (integer) pixels in the original image, origin top-left.
- Scores/similarities: floats in `[0,1]` (cosine similarity in `[-1,1]`).
- Times: `timing_ms` is wall-clock inference time for the request.
- Versioning: breaking changes bump the path prefix (`/v2`).
- All responses include `object` (+ `model` where applicable) so clients can
  branch generically.

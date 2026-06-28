# Full pipeline — every model on one session

The kitchen-sink demo. It runs **every** cnnaio model package over a single
image on one shared `*ncnn.Session` and renders annotated PNGs to `./output/`:

- `mobilenet` / `yolo11`(cls) — image classification
- `yolo11` / `nanodet` — object detection (COCO-80)
- `yolo11`(seg/pose/obb) — segmentation, pose, oriented boxes
- `facedetector` — face detection (Ultra-Light)
- `landmark` — 98-point facial landmarks (PFLD)
- `facerecognition` — face embedding + matching
- `genderdetector` — gender classification

## Run

```sh
# from the repo root
go run ./example/full-pipeline                # bundled example/testdata/test.png
go run ./example/full-pipeline my.jpg         # one image, full pipeline
go run ./example/full-pipeline a.jpg b.jpg    # two images -> face matching mode
```

Annotated results are written to `./output/<name>_<task>.png` (objects, yolo,
faces, pose, obb, seg).

## The one rule: share a Session

Every model runs on an `*ncnn.Session`, which owns the wazero runtime and the
compiled wasm. **Create one and reuse it** across models and calls — building a
session compiles the wasm (~a few hundred ms), so you don't want one per call. A
session is used serially; for concurrency create several (that's what the
server's `-j` pool does).

```go
ctx := context.Background()
session, err := ncnn.NewNcnnSession()
if err != nil { log.Fatal(err) }
defer session.Close(ctx)
```

## Minimal per-task snippets

### Image classification

```go
clf, _ := mobilenet.NewMobileNetClassifier(session)
res, _ := clf.Classify(ctx, mobilenet.V2, imageBytes, 5) // top-5
for _, p := range res.Predictions {
    fmt.Printf("%6.2f%%  %s\n", p.Score*100, p.Label)
}
```

### Object detection (YOLO11 or NanoDet)

```go
det, _ := yolo11.New(session)                       // or nanodet.New(session)
dets, _ := det.Detect(ctx, imageBytes, 0.25, 0.45)  // score, NMS thresholds
for _, d := range dets {
    fmt.Printf("%-12s %4.0f%%  [%.0f,%.0f %.0f,%.0f]\n",
        d.Label, d.Score*100, d.Box.X1, d.Box.Y1, d.Box.X2, d.Box.Y2)
}
```

`yolo11` also has `NewClassifier`, `NewSegmenter`, `NewPoseDetector`,
`NewOBBDetector` — same shape: construct with the session, call `Detect`/`Classify`.

### Faces: detect → landmarks / recognize / gender

```go
fd, _    := facedetector.New(session)
faces, _ := fd.Detect(ctx, imageBytes, 0.7, 0.3)    // []detect.Detection

lm, _    := landmark.New(session)
pts, _   := lm.Detect(ctx, imageBytes, faces[0].Box) // 98 []detect.Point

// recognition: embed + compare (cropped face OR full photo auto-cropped)
rec, _ := facerecognition.New(session)
r, _   := rec.Match(ctx,
    facerecognition.Sample{Image: photoA},
    facerecognition.Sample{Image: photoB},
    facerecognition.DefaultThreshold)
fmt.Printf("similarity %.2f same=%v\n", r.Similarity, r.Same)

// gender
gd, _ := genderdetector.New(session)
g, _  := gd.ClassifyPhoto(ctx, imageBytes)
fmt.Println(g.Label, g.Confidence)
```

### Visualizing results

`mod/render` draws boxes/labels/landmarks/skeletons/polygons/masks and saves PNGs:

```go
img, _, _ := image.Decode(bytes.NewReader(imageBytes))
canvas := render.Render(img, render.Overlay{Detections: dets})
render.SavePNG("output/result.png", canvas)
```

### Shared types (`mod/detect`)

All detectors return `detect.Box{X1,Y1,X2,Y2}` / `detect.Detection{Box,Score,ClassID,Label}`
/ `detect.Point{X,Y}` in original-image pixel coordinates, plus helpers `IoU`,
`NMS`, `SquareROI`, and `COCOLabel`.

## Prefer the HTTP server?

If you'd rather call over HTTP than link the module, run the server
(`go run . -dev`) and use the REST API — see [../../docs/API.md](../../docs/API.md).

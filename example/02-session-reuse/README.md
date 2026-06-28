# Example 2 — Session reuse & a conditional pipeline

This example shows the **core idea** of cnnaio's library API: one
`*ncnn.Session` drives many models. Building the session compiles the embedded
wasm (a few hundred ms); reusing it across models makes every subsequent model
essentially free to attach.

It runs a small conditional pipeline and writes the result to `output.json`:

```
image ─▶ object detection (YOLO11)        always run
      ─▶ face detection   (Ultra-Light)   always run
          └─ if any face ─▶ landmarks (PFLD, 98 points)   only when a face exists
```

…then **explicitly closes the session** so the wazero runtime is released
cleanly.

## Run

```sh
# from the repo root
go run ./example/02-session-reuse                 # uses example/testdata/test.png
go run ./example/02-session-reuse path/to/img.jpg # your own image
```

## Output

Console:

```
objects: 5
faces:   1
  face 1: 98 landmarks
wrote 5 object(s) and 1 face(s) to output.json
```

`output.json`:

```json
{
  "image": "example/testdata/test.png",
  "objects": [
    { "label": "person", "score": 0.91, "box": { "X1": 110, "Y1": 60, "X2": 640, "Y2": 533 } }
  ],
  "has_face": true,
  "faces": [
    { "score": 0.99,
      "box": { "X1": 150, "Y1": 90, "X2": 360, "Y2": 340 },
      "landmarks": [ { "X": 171, "Y": 225 }, "… 98 points" ] }
  ]
}
```

## What to notice

- **One session, three models.** `yolo11.New(s)`, `facedetector.New(s)` and
  `landmark.New(s)` all take the *same* `s`. None of them owns the runtime, so
  there is no per-model `Close` — you close the session once.
- **Conditional work.** The landmark model is only constructed and run when face
  detection returned at least one face (`has_face`).
- **Clean shutdown.** `session.Close(ctx)` is called explicitly before the
  program writes its output and exits, releasing the wazero runtime.

> A session is used **serially**. For concurrent inference, create several
> sessions — that is exactly what the server's `-j` pool does.

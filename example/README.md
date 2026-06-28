# cnnaio examples

Runnable examples of using cnnaio — both **as a Go library** (the numbered
programs + `full-pipeline`) and **over the HTTP API** (the [`REST API`](REST%20API/)
curl scripts). The library path is pure Go (wazero + embedded ncnn wasm +
embedded models), no cgo, no system libraries, no model files to ship.

```sh
go get cnnaio
```

The Go examples are each a self-contained `package main`. Run them from the repo
root:

| # | Example | What it shows | Output |
|---|---------|---------------|--------|
| 1 | [`01-basic-classification`](01-basic-classification/) | Smallest program: classify an image, print top-5 | STDOUT |
| 2 | [`02-session-reuse`](02-session-reuse/) | One session → detection → faces → *conditional* landmarks; clean shutdown | `output.json` |
| 3 | [`03-render-detection`](03-render-detection/) | `mod/render`: draw detection boxes onto the image | `output.png` |
| — | [`full-pipeline`](full-pipeline/) | Every model in one program + all renderers | `output/*.png` |
| — | [`REST API`](REST%20API/) | curl scripts hitting every HTTP endpoint (not the Go library) | JSON / PNG |

```sh
go run ./example/01-basic-classification
go run ./example/02-session-reuse
go run ./example/03-render-detection
go run ./example/full-pipeline
```

All four default to the bundled sample image
[`testdata/test.png`](testdata/test.png); pass a path to use your own:

```sh
go run ./example/03-render-detection path/to/my.jpg
```

## The one rule: share a Session

Every model runs on an `*ncnn.Session`, which owns the wazero runtime and the
compiled wasm. **Create one and reuse it** across models and calls — building a
session compiles the wasm (~a few hundred ms), so you don't want one per call.
A session is used serially; for concurrency create several (that's what the
server's `-j` pool does).

```go
ctx := context.Background()
session, err := ncnn.NewNcnnSession()
if err != nil { log.Fatal(err) }
defer session.Close(ctx)
```

See [`full-pipeline/README.md`](full-pipeline/README.md) for minimal,
copy-pasteable snippets of every task (classification, detection, segmentation,
pose, OBB, faces, landmarks, recognition, gender, rendering).

## Prefer the HTTP server?

If you'd rather call over HTTP than link the module, run the server
(`go run . -dev`) and use the OpenAI-style REST API — see
[../docs/API.md](../docs/API.md).

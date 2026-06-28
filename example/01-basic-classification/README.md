# Example 1 — Basic classification

The smallest useful cnnaio program. It loads an image, runs **MobileNetV2**
image classification on the embedded ncnn/wazero runtime, and prints the top-5
ImageNet labels to STDOUT.

No HTTP server, no model downloads, no cgo — the model weights and the inference
runtime are both compiled into the binary.

## Run

```sh
# from the repo root
go run ./example/01-basic-classification                 # uses example/testdata/test.png
go run ./example/01-basic-classification path/to/img.jpg # your own image
```

## Expected output

```
Classification of example/testdata/test.png  (MobileNetV2 / ImageNet-1000)
 1.  41.83%  [...] <label>
 2.  12.07%  [...] <label>
 ...

inference round-trip: 120ms
```

## The three steps

```go
session, _ := ncnn.NewNcnnSession()          // 1. one shared runtime (Close it)
defer session.Close(ctx)

clf, _ := mobilenet.NewMobileNetClassifier(session) // 2. attach a model
res, _ := clf.Classify(ctx, mobilenet.V2, img, 5)   // 3. run it (top-5)
```

`res.Predictions` is a ranked `[]mobilenet.Prediction{Index, Score, Label}`.

> **Tip:** building the session compiles the wasm, so create it **once** and
> reuse it across every model and image — see
> [Example 2](../02-session-reuse/) for a multi-model pipeline on one session.

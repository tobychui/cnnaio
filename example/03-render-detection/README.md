# Example 3 — Rendering detections to an image

Run object detection (**YOLO11**, COCO-80) and use the `mod/render` package to
draw the bounding boxes and labels back onto the photo, saving the annotated
result to `output.png`.

`mod/render` is a small, dependency-free drawing helper (boxes, labels,
landmark/keypoint dots, pose skeletons, oriented polygons, segmentation masks)
layered on top of the shared `mod/detect` types. Model packages don't depend on
it — it's a pure visualization layer.

## Run

```sh
# from the repo root
go run ./example/03-render-detection                 # uses example/testdata/test.png
go run ./example/03-render-detection path/to/img.jpg # your own image
```

## Output

Console:

```
person          91.2%  [110,60 640,533]
potted plant    74.0%  [600,360 720,533]
chair           62.5%  [380,150 660,520]

drew 3 box(es) -> output.png
```

…and an annotated `output.png` with green boxes + `label NN%` captions.

## The render call

```go
img, _, _ := image.Decode(bytes.NewReader(imgBytes))      // original picture
canvas := render.Render(img, render.Overlay{Detections: dets}) // boxes + labels
render.SavePNG("output.png", canvas)                      // write PNG
```

`render.Render` uses `AutoStyle` to scale line thickness and font size to the
image so overlays stay legible on both small and large pictures.

### Lower-level drawing

For more control, draw onto a canvas yourself:

```go
canvas := render.ToRGBA(img)
st := render.AutoStyle(img)
for _, d := range dets {
    render.DrawBox(canvas, d.Box, st.BoxColor, st.Thickness)
}
```

`mod/render` also exposes `DrawLandmarks`, `DrawSkeleton`, `DrawPolygon` and
`DrawMask` for the other model outputs — see
[`full-pipeline`](../full-pipeline/) for every renderer in one program.

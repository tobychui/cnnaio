// Example 3 — Rendering detections to an image.
//
// Run object detection (YOLO11, COCO-80) on an image and use the mod/render
// package to draw the bounding boxes + labels onto the picture, saving the
// annotated result to ./output.png.
//
// mod/render is a small, dependency-free drawing helper (boxes, labels,
// landmarks, skeletons, polygons, masks) layered on top of mod/detect — handy
// for debugging or any app that just wants a picture back.
//
// Run it:
//
//	go run ./example/03-render-detection                 # uses the bundled test.png
//	go run ./example/03-render-detection path/to/img.jpg # your own image
package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	_ "image/jpeg" // register decoders so image.Decode handles jpg/png input
	_ "image/png"
	"os"

	"cnnaio/mod/ncnn"
	"cnnaio/mod/render"
	"cnnaio/mod/yolo11"
)

const (
	defaultImage = "example/testdata/test.png"
	outputFile   = "output.png"
)

func main() {
	imagePath := defaultImage
	if len(os.Args) > 1 {
		imagePath = os.Args[1]
	}
	imgBytes, err := os.ReadFile(imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot read image %q: %v\n", imagePath, err)
		os.Exit(1)
	}

	ctx := context.Background()

	session, err := ncnn.NewNcnnSession()
	if err != nil {
		fmt.Fprintf(os.Stderr, "init ncnn session: %v\n", err)
		os.Exit(1)
	}
	defer session.Close(ctx)

	// 1. Detect objects.
	det, err := yolo11.New(session)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init yolo11: %v\n", err)
		os.Exit(1)
	}
	dets, err := det.Detect(ctx, imgBytes, 0.25, 0.45) // score, NMS thresholds
	if err != nil {
		fmt.Fprintf(os.Stderr, "detect: %v\n", err)
		os.Exit(1)
	}
	for _, d := range dets {
		fmt.Printf("%-14s %5.1f%%  [%.0f,%.0f %.0f,%.0f]\n",
			d.Label, d.Score*100, d.Box.X1, d.Box.Y1, d.Box.X2, d.Box.Y2)
	}

	// 2. Decode the original image so we can draw on it.
	img, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		fmt.Fprintf(os.Stderr, "decode image: %v\n", err)
		os.Exit(1)
	}

	// 3. Render the boxes + labels. render.Render auto-sizes line thickness and
	//    font to the image, then returns an annotated canvas.
	canvas := render.Render(img, render.Overlay{Detections: dets})

	// 4. Save the annotated PNG (parent dirs are created if needed).
	if err := render.SavePNG(outputFile, canvas); err != nil {
		fmt.Fprintf(os.Stderr, "save %s: %v\n", outputFile, err)
		os.Exit(1)
	}
	fmt.Printf("\ndrew %d box(es) -> %s\n", len(dets), outputFile)
}

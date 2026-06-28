// Package landmark runs the PFLD facial-landmark model (98 points) on a face
// crop, using a shared ncnn.Session. Given an image and a face box (e.g. from
// mod/facedetector), it returns the 98 landmark points in original-image pixel
// coordinates. The face region is fed to the net via the wasm "infer" command's
// ROI cropping, so nothing is decoded on the Go side except the image header.
package landmark

import (
	"bytes"
	"context"
	"embed"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io/fs"
	"strconv"

	"cnnaio/mod/detect"
	"cnnaio/mod/ncnn"
)

//go:embed models/pfld-sim.param models/pfld-sim.bin
var models embed.FS

const (
	inputSize = 112 // PFLD square input
	numPoints = 98  // WFLW 98-point landmarks (output = 98*2 = 196 values)
	margin    = 0.15
)

// PFLD preprocessing: RGB, scaled to [0,1] (no mean subtraction).
var norm = [3]float64{1.0 / 255, 1.0 / 255, 1.0 / 255}

// Detector runs PFLD on a shared ncnn.Session.
type Detector struct {
	session  *ncnn.Session
	modelsFS fs.FS
}

// New attaches a landmark detector to an existing ncnn.Session.
func New(session *ncnn.Session) (*Detector, error) {
	if session == nil {
		return nil, fmt.Errorf("nil ncnn.Session")
	}
	sub, err := fs.Sub(models, "models")
	if err != nil {
		return nil, fmt.Errorf("sub models fs: %w", err)
	}
	return &Detector{session: session, modelsFS: sub}, nil
}

// Detect returns the 98 facial landmarks for the given face box, in original
// image pixel coordinates.
func (d *Detector) Detect(ctx context.Context, img []byte, box detect.Box) ([]detect.Point, error) {
	imgW, imgH, err := imageSize(img)
	if err != nil {
		return nil, fmt.Errorf("read image size: %w", err)
	}
	rx, ry, rw, rh := detect.SquareROI(box, imgW, imgH, margin)
	if rw <= 0 || rh <= 0 {
		return nil, fmt.Errorf("empty face roi")
	}

	args := []string{
		"ncnn", "infer",
		"/models/pfld-sim.param",
		"/models/pfld-sim.bin",
		"1", // text param
		ncnn.ImageGuestPath + "/image",
		strconv.Itoa(inputSize), strconv.Itoa(inputSize),
		"0",           // keep RGB
		"0", "0", "0", // mean
		ftoa(norm[0]), ftoa(norm[1]), ftoa(norm[2]),
		strconv.Itoa(rx), strconv.Itoa(ry), strconv.Itoa(rw), strconv.Itoa(rh), // roi
		"input_1", // input blob
		"1",       // one output
		"415",
	}

	run, err := d.session.Run(ctx, ncnn.RunRequest{
		Args: args,
		Mounts: map[string]fs.FS{
			"/models":           d.modelsFS,
			ncnn.ImageGuestPath: ncnn.ImageFS(img),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("landmark inference failed: %w\nstderr: %s", err, run.Stderr)
	}

	infer, err := ncnn.ParseInferOutput(run.Stdout)
	if err != nil {
		return nil, fmt.Errorf("parse infer output: %w", err)
	}
	out, ok := infer.Tensor("415")
	if !ok {
		return nil, fmt.Errorf("missing '415' tensor")
	}
	if len(out.Data) < numPoints*2 {
		return nil, fmt.Errorf("landmark tensor too small: %d (want %d)", len(out.Data), numPoints*2)
	}

	// PFLD emits interleaved (x,y) normalized to [0,1] over the crop; map back
	// onto the ROI in original-image coordinates.
	pts := make([]detect.Point, numPoints)
	for i := 0; i < numPoints; i++ {
		lx := out.Data[i*2]
		ly := out.Data[i*2+1]
		pts[i] = detect.Point{
			X: float32(rx) + lx*float32(rw),
			Y: float32(ry) + ly*float32(rh),
		}
	}
	return pts, nil
}

func imageSize(data []byte) (int, int, error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return 0, 0, err
	}
	return cfg.Width, cfg.Height, nil
}

func ftoa(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }

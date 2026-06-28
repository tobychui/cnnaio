// Package facedetector runs the Ultra-Light-Fast-Generic-Face-Detector
// (version-RFB / version-slim, 320x240) on a shared ncnn.Session. The models are
// embedded via go:embed; SSD prior generation, box decoding and NMS happen here
// in Go from the raw "scores"/"boxes" tensors dumped by the wasm "infer" command.
package facedetector

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"math"
	"strconv"

	"cnnaio/mod/detect"
	"cnnaio/mod/ncnn"
)

//go:embed models/RFB-320.param models/RFB-320.bin models/slim_320.param models/slim_320.bin
var models embed.FS

// Variant selects which backbone to use.
type Variant int

const (
	RFB  Variant = iota // version-RFB-320 (more accurate)
	Slim                // version-slim-320 (faster)
)

const (
	inputW = 320
	inputH = 240
)

// Ultra-Light preprocessing: RGB input, (x-127)/128.
var (
	mean = [3]float64{127, 127, 127}
	norm = [3]float64{1.0 / 128, 1.0 / 128, 1.0 / 128}
)

// SSD prior config (matches the Linzaer reference).
var (
	strides  = []int{8, 16, 32, 64}
	minBoxes = [][]float64{{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}}
)

const (
	centerVariance = 0.1
	sizeVariance   = 0.2
)

// prior is a normalized [cx, cy, w, h] box in [0,1].
type prior struct{ cx, cy, w, h float64 }

// Detector runs the face detector on a shared ncnn.Session.
type Detector struct {
	session   *ncnn.Session
	modelsFS  fs.FS
	paramName string
	modelName string
	priors    []prior
}

// New attaches a face detector (default RFB) to an existing ncnn.Session.
func New(session *ncnn.Session) (*Detector, error) {
	return NewVariant(session, RFB)
}

// NewVariant attaches a face detector using the chosen backbone.
func NewVariant(session *ncnn.Session, v Variant) (*Detector, error) {
	if session == nil {
		return nil, fmt.Errorf("nil ncnn.Session")
	}
	sub, err := fs.Sub(models, "models")
	if err != nil {
		return nil, fmt.Errorf("sub models fs: %w", err)
	}
	param, model := "RFB-320.param", "RFB-320.bin"
	if v == Slim {
		param, model = "slim_320.param", "slim_320.bin"
	}
	return &Detector{
		session:   session,
		modelsFS:  sub,
		paramName: param,
		modelName: model,
		priors:    generatePriors(),
	}, nil
}

// Detect returns face boxes in original-image pixel coordinates. scoreThresh is
// the minimum face probability (~0.7 works well); nmsThresh is the IoU threshold.
func (d *Detector) Detect(ctx context.Context, image []byte, scoreThresh, nmsThresh float32) ([]detect.Detection, error) {
	args := []string{
		"ncnn", "infer",
		"/models/" + d.paramName,
		"/models/" + d.modelName,
		"1", // text param
		ncnn.ImageGuestPath + "/image",
		strconv.Itoa(inputW), strconv.Itoa(inputH),
		"0", // keep RGB
		ftoa(mean[0]), ftoa(mean[1]), ftoa(mean[2]),
		ftoa(norm[0]), ftoa(norm[1]), ftoa(norm[2]),
		"0", "0", "0", "0", // roi: whole image
		"input", // input blob name
		"2",     // two outputs
		"scores", "boxes",
	}

	run, err := d.session.Run(ctx, ncnn.RunRequest{
		Args: args,
		Mounts: map[string]fs.FS{
			"/models":           d.modelsFS,
			ncnn.ImageGuestPath: ncnn.ImageFS(image),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("facedetector inference failed: %w\nstderr: %s", err, run.Stderr)
	}

	infer, err := ncnn.ParseInferOutput(run.Stdout)
	if err != nil {
		return nil, fmt.Errorf("parse infer output: %w", err)
	}
	scores, ok := infer.Tensor("scores")
	if !ok {
		return nil, fmt.Errorf("missing 'scores' tensor")
	}
	boxes, ok := infer.Tensor("boxes")
	if !ok {
		return nil, fmt.Errorf("missing 'boxes' tensor")
	}

	dets := d.decode(scores, boxes, infer.OrigW, infer.OrigH, scoreThresh)
	return detect.NMS(dets, nmsThresh), nil
}

func (d *Detector) decode(scores, boxes ncnn.Tensor, origW, origH int, scoreThresh float32) []detect.Detection {
	// scores: [N,2] (background, face); boxes: [N,4] (dx,dy,dw,dh vs prior).
	n := len(d.priors)
	sc := rowAccessor(scores, 2)
	bx := rowAccessor(boxes, 4)

	var dets []detect.Detection
	for i := 0; i < n; i++ {
		face := sc(i, 1)
		if face < scoreThresh {
			continue
		}
		p := d.priors[i]
		cx := float64(bx(i, 0))*centerVariance*p.w + p.cx
		cy := float64(bx(i, 1))*centerVariance*p.h + p.cy
		w := math.Exp(float64(bx(i, 2))*sizeVariance) * p.w
		h := math.Exp(float64(bx(i, 3))*sizeVariance) * p.h

		b := detect.Box{
			X1: clamp(float32((cx-w/2)*float64(origW)), 0, float32(origW)),
			Y1: clamp(float32((cy-h/2)*float64(origH)), 0, float32(origH)),
			X2: clamp(float32((cx+w/2)*float64(origW)), 0, float32(origW)),
			Y2: clamp(float32((cy+h/2)*float64(origH)), 0, float32(origH)),
		}
		dets = append(dets, detect.Detection{Box: b, Score: face, ClassID: 0, Label: "face"})
	}
	return dets
}

// generatePriors builds the SSD priors for the 320x240 input, in the same order
// the network concatenates its head outputs.
func generatePriors() []prior {
	var priors []prior
	for k, stride := range strides {
		fw := ceilDiv(inputW, stride)
		fh := ceilDiv(inputH, stride)
		scaleW := float64(inputW) / float64(stride)
		scaleH := float64(inputH) / float64(stride)
		for j := 0; j < fh; j++ {
			for i := 0; i < fw; i++ {
				xc := (float64(i) + 0.5) / scaleW
				yc := (float64(j) + 0.5) / scaleH
				for _, mb := range minBoxes[k] {
					priors = append(priors, prior{
						cx: clampf(xc, 0, 1),
						cy: clampf(yc, 0, 1),
						w:  mb / float64(inputW),
						h:  mb / float64(inputH),
					})
				}
			}
		}
	}
	return priors
}

// rowAccessor returns data[i*cols + c], handling the transposed orientation
// (W=N instead of W=cols) just in case.
func rowAccessor(t ncnn.Tensor, cols int) func(i, c int) float32 {
	if t.W == cols {
		return func(i, c int) float32 { return t.Data[i*cols+c] }
	}
	return func(i, c int) float32 { return t.Data[c*t.W+i] }
}

func ceilDiv(a, b int) int { return (a + b - 1) / b }

func clamp(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampf(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func ftoa(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }

// Package nanodet runs NanoDet-Plus object detection (COCO, 80 classes) on top
// of a shared ncnn.Session. The model is embedded via go:embed; decoding of the
// anchor-free GFL output (box distribution + per-class scores) and NMS happen
// here in Go using the raw tensor dumped by the wasm "infer" command.
package nanodet

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

//go:embed models/nanodet-plus-m_416.param models/nanodet-plus-m_416.bin
var models embed.FS

const (
	inputSize = 416 // square input the net expects
	numClass  = 80
	regMax    = 7 // box distribution has regMax+1 = 8 bins per side
	channels  = numClass + 4*(regMax+1)
)

// strides of the four detection heads (anchor-free centers).
var strides = []int{8, 16, 32, 64}

// NanoDet-Plus preprocessing (BGR input). mean/std are the ImageNet-BGR values
// used by RangiLyu's reference demo.
var (
	mean = [3]float64{103.53, 116.28, 123.675}
	norm = [3]float64{1.0 / 57.375, 1.0 / 57.12, 1.0 / 58.395}
)

// Detector runs NanoDet-Plus on a shared ncnn.Session.
type Detector struct {
	session   *ncnn.Session
	param     fs.FS
	paramName string
	modelName string
}

// New attaches a NanoDet detector to an existing ncnn.Session.
func New(session *ncnn.Session) (*Detector, error) {
	if session == nil {
		return nil, fmt.Errorf("nil ncnn.Session")
	}
	sub, err := fs.Sub(models, "models")
	if err != nil {
		return nil, fmt.Errorf("sub models fs: %w", err)
	}
	return &Detector{
		session:   session,
		param:     sub,
		paramName: "nanodet-plus-m_416.param",
		modelName: "nanodet-plus-m_416.bin",
	}, nil
}

// Detect runs detection on a decoded image's raw bytes and returns boxes in the
// original image's pixel coordinates. scoreThresh filters low-confidence
// detections before NMS; nmsThresh is the IoU suppression threshold.
func (d *Detector) Detect(ctx context.Context, image []byte, scoreThresh, nmsThresh float32) ([]detect.Detection, error) {
	args := []string{
		"ncnn", "infer",
		"/models/" + d.paramName,
		"/models/" + d.modelName,
		"1", // param is text
		ncnn.ImageGuestPath + "/image",
		strconv.Itoa(inputSize), strconv.Itoa(inputSize),
		"1", // to BGR
		ftoa(mean[0]), ftoa(mean[1]), ftoa(mean[2]),
		ftoa(norm[0]), ftoa(norm[1]), ftoa(norm[2]),
		"0", "0", "0", "0", // roi: whole image
		"data", // input blob name
		"1",    // one output
		"output",
	}

	run, err := d.session.Run(ctx, ncnn.RunRequest{
		Args: args,
		Mounts: map[string]fs.FS{
			"/models":           d.param,
			ncnn.ImageGuestPath: ncnn.ImageFS(image),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("nanodet inference failed: %w\nstderr: %s", err, run.Stderr)
	}

	infer, err := ncnn.ParseInferOutput(run.Stdout)
	if err != nil {
		return nil, fmt.Errorf("parse infer output: %w", err)
	}
	out, ok := infer.Tensor("output")
	if !ok {
		return nil, fmt.Errorf("missing 'output' tensor")
	}

	dets := decode(out, infer.OrigW, infer.OrigH, scoreThresh)
	return detect.NMS(dets, nmsThresh), nil
}

// rowAt returns the per-anchor row accessor independent of tensor orientation.
// The model output is [numAnchors, channels]; depending on the final Permute it
// may come back as W=channels (row-major) or W=numAnchors (transposed).
func rowAccessor(t ncnn.Tensor) (numAnchors int, at func(anchor, ch int) float32) {
	if t.W == channels {
		return t.H, func(a, c int) float32 { return t.Data[a*channels+c] }
	}
	// transposed: data[c*numAnchors + a]
	return t.W, func(a, c int) float32 { return t.Data[c*t.W+a] }
}

func decode(out ncnn.Tensor, origW, origH int, scoreThresh float32) []detect.Detection {
	numAnchors, at := rowAccessor(out)
	scaleX := float32(origW) / inputSize
	scaleY := float32(origH) / inputSize

	var dets []detect.Detection
	anchor := 0
	for _, stride := range strides {
		fw := (inputSize + stride - 1) / stride // ceil
		fh := (inputSize + stride - 1) / stride
		for y := 0; y < fh; y++ {
			for x := 0; x < fw; x++ {
				if anchor >= numAnchors {
					break
				}
				a := anchor
				anchor++

				// best class (scores are already sigmoid'd in-network)
				bestC, bestS := 0, at(a, 0)
				for c := 1; c < numClass; c++ {
					if s := at(a, c); s > bestS {
						bestS, bestC = s, c
					}
				}
				if bestS < scoreThresh {
					continue
				}

				// box distribution -> distances (left, top, right, bottom).
				// NanoDet-Plus anchor centers are at (x*stride, y*stride) — no
				// half-cell offset (matches RangiLyu's reference demo).
				cx := float32(x * stride)
				cy := float32(y * stride)
				var dist [4]float32
				for side := 0; side < 4; side++ {
					dist[side] = integral(at, a, numClass+side*(regMax+1)) * float32(stride)
				}

				b := detect.Box{
					X1: clamp((cx-dist[0])*scaleX, 0, float32(origW)),
					Y1: clamp((cy-dist[1])*scaleY, 0, float32(origH)),
					X2: clamp((cx+dist[2])*scaleX, 0, float32(origW)),
					Y2: clamp((cy+dist[3])*scaleY, 0, float32(origH)),
				}
				dets = append(dets, detect.Detection{
					Box:     b,
					Score:   bestS,
					ClassID: bestC,
					Label:   detect.COCOLabel(bestC),
				})
			}
		}
	}
	return dets
}

// integral computes the softmax-weighted expected value over regMax+1 bins
// starting at channel base — the "distribution focal loss" decode of one side.
func integral(at func(a, c int) float32, a, base int) float32 {
	var maxv float32 = at(a, base)
	for j := 1; j <= regMax; j++ {
		if v := at(a, base+j); v > maxv {
			maxv = v
		}
	}
	var sum, acc float64
	bins := make([]float64, regMax+1)
	for j := 0; j <= regMax; j++ {
		bins[j] = math.Exp(float64(at(a, base+j) - maxv))
		sum += bins[j]
	}
	for j := 0; j <= regMax; j++ {
		acc += float64(j) * (bins[j] / sum)
	}
	return float32(acc)
}

func clamp(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func ftoa(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }

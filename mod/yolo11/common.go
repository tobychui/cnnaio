// Package yolo11 wires up the Ultralytics YOLO11 model family (exported to ncnn)
// on a shared ncnn.Session: detection, classification, pose, oriented boxes (OBB)
// and instance segmentation. Each task is a separate type (Detector, Classifier,
// PoseDetector, OBBDetector, Segmenter); they share the embedded models and the
// decoding helpers in this file.
package yolo11

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"math"
	"strconv"

	"cnnaio/mod/ncnn"
)

//go:embed models/*.ncnn.param models/*.ncnn.bin
var models embed.FS

const (
	detSize = 640 // input size for detect/pose/obb/seg
	regMax  = 16  // DFL bins per box side (box channels = 4*regMax = 64)
)

// strides of the three detection heads (anchor centers at (x+0.5, y+0.5)*stride).
var strides = []int{8, 16, 32}

// norm scales raw RGB pixels to [0,1]; YOLO11 ncnn exports have no built-in
// normalization, so the host applies it. (mean is zero.)
var norm = [3]float64{1.0 / 255, 1.0 / 255, 1.0 / 255}

func modelsFS() (fs.FS, error) { return fs.Sub(models, "models") }

// runInfer runs the wasm "infer" command over the whole image (no ROI) for the
// given YOLO model and returns the parsed output tensors.
func runInfer(ctx context.Context, s *ncnn.Session, param fs.FS, paramName, modelName string,
	target int, inName string, outNames []string, image []byte) (*ncnn.InferOutput, error) {
	args := []string{
		"ncnn", "infer",
		"/models/" + paramName,
		"/models/" + modelName,
		"1", // text param
		ncnn.ImageGuestPath + "/image",
		strconv.Itoa(target), strconv.Itoa(target),
		"0",           // keep RGB
		"0", "0", "0", // mean
		ftoa(norm[0]), ftoa(norm[1]), ftoa(norm[2]),
		"0", "0", "0", "0", // roi: whole image
		inName,
		strconv.Itoa(len(outNames)),
	}
	args = append(args, outNames...)

	run, err := s.Run(ctx, ncnn.RunRequest{
		Args: args,
		Mounts: map[string]fs.FS{
			"/models":           param,
			ncnn.ImageGuestPath: ncnn.ImageFS(image),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("yolo11 inference failed: %w\nstderr: %s", err, run.Stderr)
	}
	return ncnn.ParseInferOutput(run.Stdout)
}

// accessor returns a per-anchor channel accessor independent of the output
// orientation: a [numAnchors, channels] tensor may come back as W=channels
// (row-major) or W=numAnchors (transposed).
func accessor(t ncnn.Tensor, channels int) (numAnchors int, at func(anchor, ch int) float32) {
	if t.W == channels {
		return t.H, func(a, c int) float32 { return t.Data[a*channels+c] }
	}
	return t.W, func(a, c int) float32 { return t.Data[c*t.W+a] }
}

// dflExpect computes the softmax-weighted expected value over regMax bins starting
// at channel base — the distribution-focal decode of one box side, in grid units.
func dflExpect(at func(a, c int) float32, a, base int) float32 {
	maxv := at(a, base)
	for j := 1; j < regMax; j++ {
		if v := at(a, base+j); v > maxv {
			maxv = v
		}
	}
	var sum, acc float64
	bins := make([]float64, regMax)
	for j := 0; j < regMax; j++ {
		bins[j] = math.Exp(float64(at(a, base+j) - maxv))
		sum += bins[j]
	}
	for j := 0; j < regMax; j++ {
		acc += float64(j) * (bins[j] / sum)
	}
	return float32(acc)
}

func sigmoid(x float32) float32 { return float32(1.0 / (1.0 + math.Exp(float64(-x)))) }

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

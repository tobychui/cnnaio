package yolo11

import (
	"context"
	"fmt"
	"io/fs"

	"cnnaio/mod/detect"
	"cnnaio/mod/ncnn"
)

const (
	detNumClass = 80                     // COCO
	detChannels = 4*regMax + detNumClass // 64 box + 80 class = 144
)

// Detector runs YOLO11 object detection (COCO, 80 classes) on a shared session.
type Detector struct {
	session *ncnn.Session
	param   fs.FS
}

// New attaches a YOLO11 detector to an existing ncnn.Session.
func New(session *ncnn.Session) (*Detector, error) {
	if session == nil {
		return nil, fmt.Errorf("nil ncnn.Session")
	}
	sub, err := modelsFS()
	if err != nil {
		return nil, err
	}
	return &Detector{session: session, param: sub}, nil
}

// Detect returns COCO object boxes in original-image pixel coordinates.
func (d *Detector) Detect(ctx context.Context, image []byte, scoreThresh, nmsThresh float32) ([]detect.Detection, error) {
	infer, err := runInfer(ctx, d.session, d.param, "yolo11n.ncnn.param", "yolo11n.ncnn.bin",
		detSize, "in0", []string{"out0"}, image)
	if err != nil {
		return nil, err
	}
	out, ok := infer.Tensor("out0")
	if !ok {
		return nil, fmt.Errorf("missing 'out0' tensor")
	}
	dets := decodeDetect(out, infer.OrigW, infer.OrigH, scoreThresh)
	return detect.NMS(dets, nmsThresh), nil
}

// decodeDetect turns the [anchors, 144] head output into boxes (DFL box decode +
// per-class sigmoid scores), in original-image coordinates.
func decodeDetect(out ncnn.Tensor, origW, origH int, scoreThresh float32) []detect.Detection {
	numAnchors, at := accessor(out, detChannels)
	scaleX := float32(origW) / detSize
	scaleY := float32(origH) / detSize

	var dets []detect.Detection
	anchor := 0
	for _, stride := range strides {
		fw := (detSize + stride - 1) / stride
		fh := (detSize + stride - 1) / stride
		for y := 0; y < fh; y++ {
			for x := 0; x < fw; x++ {
				if anchor >= numAnchors {
					break
				}
				a := anchor
				anchor++

				bestC, bestLogit := 0, at(a, 4*regMax)
				for c := 1; c < detNumClass; c++ {
					if l := at(a, 4*regMax+c); l > bestLogit {
						bestLogit, bestC = l, c
					}
				}
				score := sigmoid(bestLogit)
				if score < scoreThresh {
					continue
				}

				cx := (float32(x) + 0.5) * float32(stride)
				cy := (float32(y) + 0.5) * float32(stride)
				dl := dflExpect(at, a, 0*regMax) * float32(stride)
				dt := dflExpect(at, a, 1*regMax) * float32(stride)
				dr := dflExpect(at, a, 2*regMax) * float32(stride)
				db := dflExpect(at, a, 3*regMax) * float32(stride)

				dets = append(dets, detect.Detection{
					Box: detect.Box{
						X1: clamp((cx-dl)*scaleX, 0, float32(origW)),
						Y1: clamp((cy-dt)*scaleY, 0, float32(origH)),
						X2: clamp((cx+dr)*scaleX, 0, float32(origW)),
						Y2: clamp((cy+db)*scaleY, 0, float32(origH)),
					},
					Score:   score,
					ClassID: bestC,
					Label:   detect.COCOLabel(bestC),
				})
			}
		}
	}
	return dets
}

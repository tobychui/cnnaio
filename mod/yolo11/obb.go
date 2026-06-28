package yolo11

import (
	"context"
	"fmt"
	"io/fs"
	"math"
	"sort"

	"cnnaio/mod/detect"
	"cnnaio/mod/ncnn"
)

const (
	obbNumClass = 15                     // DOTA v1
	obbChannels = 4*regMax + obbNumClass // out0: 64 box + 15 class = 79 ; out1: 1 angle
)

// dotaLabels are the 15 DOTA-v1 classes in ultralytics order.
var dotaLabels = []string{
	"plane", "ship", "storage tank", "baseball diamond", "tennis court",
	"basketball court", "ground track field", "harbor", "bridge", "large vehicle",
	"small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool",
}

// OBBDetection is one oriented (rotated) box, given as its 4 corner points in
// original-image coordinates.
type OBBDetection struct {
	Poly    [4]detect.Point
	Score   float32
	ClassID int
	Label   string
}

// OBBDetector runs YOLO11 oriented-bounding-box detection (DOTA, 15 classes).
type OBBDetector struct {
	session *ncnn.Session
	param   fs.FS
}

// NewOBBDetector attaches a YOLO11 OBB detector to an existing ncnn.Session.
func NewOBBDetector(session *ncnn.Session) (*OBBDetector, error) {
	if session == nil {
		return nil, fmt.Errorf("nil ncnn.Session")
	}
	sub, err := modelsFS()
	if err != nil {
		return nil, err
	}
	return &OBBDetector{session: session, param: sub}, nil
}

// Detect returns oriented boxes in original-image coordinates.
func (d *OBBDetector) Detect(ctx context.Context, image []byte, scoreThresh, nmsThresh float32) ([]OBBDetection, error) {
	infer, err := runInfer(ctx, d.session, d.param, "yolo11n_obb.ncnn.param", "yolo11n_obb.ncnn.bin",
		detSize, "in0", []string{"out0", "out1"}, image)
	if err != nil {
		return nil, err
	}
	box, ok := infer.Tensor("out0")
	if !ok {
		return nil, fmt.Errorf("missing 'out0' tensor")
	}
	ang, ok := infer.Tensor("out1")
	if !ok {
		return nil, fmt.Errorf("missing 'out1' tensor")
	}
	dets := decodeOBB(box, ang, infer.OrigW, infer.OrigH, scoreThresh)
	return nmsOBB(dets, nmsThresh), nil
}

func decodeOBB(box, ang ncnn.Tensor, origW, origH int, scoreThresh float32) []OBBDetection {
	numAnchors, atB := accessor(box, obbChannels)
	_, atA := accessor(ang, 1)
	scaleX := float32(origW) / detSize
	scaleY := float32(origH) / detSize

	var dets []OBBDetection
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

				bestC, bestLogit := 0, atB(a, 4*regMax)
				for c := 1; c < obbNumClass; c++ {
					if l := atB(a, 4*regMax+c); l > bestLogit {
						bestLogit, bestC = l, c
					}
				}
				score := sigmoid(bestLogit)
				if score < scoreThresh {
					continue
				}

				// dist2rbox: rotate the (l,t,r,b) offsets by the predicted angle.
				angle := (float64(sigmoid(atA(a, 0))) - 0.25) * math.Pi
				l := dflExpect(atB, a, 0*regMax)
				t := dflExpect(atB, a, 1*regMax)
				r := dflExpect(atB, a, 2*regMax)
				b := dflExpect(atB, a, 3*regMax)
				cosA, sinA := math.Cos(angle), math.Sin(angle)
				xf := float64(r-l) / 2
				yf := float64(b-t) / 2
				ox := xf*cosA - yf*sinA
				oy := xf*sinA + yf*cosA
				cx := (float64(x) + 0.5 + ox) * float64(stride)
				cy := (float64(y) + 0.5 + oy) * float64(stride)
				w := float64(l+r) * float64(stride)
				h := float64(t+b) * float64(stride)

				var poly [4]detect.Point
				for i, s := range [4][2]float64{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}} {
					cxo := s[0] * w / 2
					cyo := s[1] * h / 2
					px := cx + cxo*cosA - cyo*sinA
					py := cy + cxo*sinA + cyo*cosA
					poly[i] = detect.Point{X: float32(px) * scaleX, Y: float32(py) * scaleY}
				}

				dets = append(dets, OBBDetection{Poly: poly, Score: score, ClassID: bestC, Label: dotaLabels[bestC]})
			}
		}
	}
	return dets
}

// nmsOBB does greedy NMS using the axis-aligned bounding box of each polygon as
// an approximation of rotated IoU (adequate for visualization).
func nmsOBB(dets []OBBDetection, iouThresh float32) []OBBDetection {
	sort.SliceStable(dets, func(i, j int) bool { return dets[i].Score > dets[j].Score })
	kept := make([]OBBDetection, 0, len(dets))
	dead := make([]bool, len(dets))
	for i := range dets {
		if dead[i] {
			continue
		}
		kept = append(kept, dets[i])
		for j := i + 1; j < len(dets); j++ {
			if !dead[j] && dets[j].ClassID == dets[i].ClassID &&
				detect.IoU(aabb(dets[i].Poly), aabb(dets[j].Poly)) > iouThresh {
				dead[j] = true
			}
		}
	}
	return kept
}

func aabb(p [4]detect.Point) detect.Box {
	x1, y1 := p[0].X, p[0].Y
	x2, y2 := p[0].X, p[0].Y
	for _, q := range p[1:] {
		x1 = minf(x1, q.X)
		y1 = minf(y1, q.Y)
		x2 = maxf(x2, q.X)
		y2 = maxf(y2, q.Y)
	}
	return detect.Box{X1: x1, Y1: y1, X2: x2, Y2: y2}
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

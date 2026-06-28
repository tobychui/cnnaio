// Package detect holds shared types and helpers for object detectors built on
// the ncnn runtime (bounding boxes, IoU, non-maximum suppression). It is model
// agnostic: nanodet, facedetector, etc. all produce []Detection and reuse NMS.
package detect

import "sort"

// Box is an axis-aligned bounding box in pixel coordinates of the original image.
type Box struct {
	X1, Y1, X2, Y2 float32
}

func (b Box) Width() float32  { return b.X2 - b.X1 }
func (b Box) Height() float32 { return b.Y2 - b.Y1 }
func (b Box) Area() float32 {
	w, h := b.Width(), b.Height()
	if w <= 0 || h <= 0 {
		return 0
	}
	return w * h
}

// SquareROI expands a box to a square (by its larger side) with a relative
// margin and clamps it to the image. Returns integer x,y,w,h suitable for the
// wasm "infer" ROI; w/h may differ slightly from square at the image edges.
// Used to feed a face crop to landmark / recognition models.
func SquareROI(b Box, imgW, imgH int, margin float32) (x, y, w, h int) {
	cx := (b.X1 + b.X2) / 2
	cy := (b.Y1 + b.Y2) / 2
	side := b.Width()
	if b.Height() > side {
		side = b.Height()
	}
	side *= 1 + margin
	clamp := func(v, hi float32) float32 {
		if v < 0 {
			return 0
		}
		if v > hi {
			return hi
		}
		return v
	}
	x1 := clamp(cx-side/2, float32(imgW))
	y1 := clamp(cy-side/2, float32(imgH))
	x2 := clamp(cx+side/2, float32(imgW))
	y2 := clamp(cy+side/2, float32(imgH))
	return int(x1), int(y1), int(x2 - x1), int(y2 - y1)
}

// Detection is one detected object.
type Detection struct {
	Box     Box
	Score   float32
	ClassID int
	Label   string
}

// Point is a 2D point in image pixel coordinates (landmarks, keypoints, polygon
// corners).
type Point struct {
	X, Y float32
}

// IoU returns the intersection-over-union of two boxes.
func IoU(a, b Box) float32 {
	x1 := maxf(a.X1, b.X1)
	y1 := maxf(a.Y1, b.Y1)
	x2 := minf(a.X2, b.X2)
	y2 := minf(a.Y2, b.Y2)
	iw, ih := x2-x1, y2-y1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	union := a.Area() + b.Area() - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// NMS performs greedy non-maximum suppression: it sorts detections by score
// (descending) and drops any box overlapping a kept higher-scoring box of the
// same class by more than iouThresh. Returns the kept detections, score order.
func NMS(dets []Detection, iouThresh float32) []Detection {
	sort.SliceStable(dets, func(i, j int) bool { return dets[i].Score > dets[j].Score })

	kept := make([]Detection, 0, len(dets))
	suppressed := make([]bool, len(dets))
	for i := range dets {
		if suppressed[i] {
			continue
		}
		kept = append(kept, dets[i])
		for j := i + 1; j < len(dets); j++ {
			if suppressed[j] || dets[j].ClassID != dets[i].ClassID {
				continue
			}
			if IoU(dets[i].Box, dets[j].Box) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return kept
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

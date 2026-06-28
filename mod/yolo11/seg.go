package yolo11

import (
	"context"
	"fmt"
	"io/fs"
	"sort"

	"cnnaio/mod/detect"
	"cnnaio/mod/ncnn"
)

const (
	segNumClass  = 80                     // COCO
	segMaskCoeff = 32                     // mask coefficients per anchor (out1)
	segChannels  = 4*regMax + segNumClass // out0: 64 box + 80 class = 144
)

// SegResult is the segmentation output. Mask is a combined (union) instance mask
// at original-image resolution (Mask[y*W+x] true where any object is); Instances
// holds the same objects with individual box-cropped masks.
type SegResult struct {
	Detections []detect.Detection
	Mask       []bool
	W, H       int
	Instances  []SegInstance
}

// SegInstance is one segmented object with a mask cropped to its box. Mask is
// row-major of size W*H, where (W,H) is the box size and (OriginX,OriginY) is the
// box's top-left in the original image.
type SegInstance struct {
	Detection        detect.Detection
	Mask             []bool
	W, H             int
	OriginX, OriginY int
}

// Segmenter runs YOLO11 instance segmentation (COCO, 80 classes).
type Segmenter struct {
	session *ncnn.Session
	param   fs.FS
}

// NewSegmenter attaches a YOLO11 segmenter to an existing ncnn.Session.
func NewSegmenter(session *ncnn.Session) (*Segmenter, error) {
	if session == nil {
		return nil, fmt.Errorf("nil ncnn.Session")
	}
	sub, err := modelsFS()
	if err != nil {
		return nil, err
	}
	return &Segmenter{session: session, param: sub}, nil
}

type segCand struct {
	det    detect.Detection
	coeffs [segMaskCoeff]float32
}

// Detect returns boxes and a combined instance mask in original-image resolution.
func (s *Segmenter) Detect(ctx context.Context, image []byte, scoreThresh, nmsThresh float32) (*SegResult, error) {
	infer, err := runInfer(ctx, s.session, s.param, "yolo11n_seg.ncnn.param", "yolo11n_seg.ncnn.bin",
		detSize, "in0", []string{"out0", "out1", "out2"}, image)
	if err != nil {
		return nil, err
	}
	det, ok := infer.Tensor("out0")
	if !ok {
		return nil, fmt.Errorf("missing 'out0' tensor")
	}
	coef, ok := infer.Tensor("out1")
	if !ok {
		return nil, fmt.Errorf("missing 'out1' tensor")
	}
	proto, ok := infer.Tensor("out2")
	if !ok {
		return nil, fmt.Errorf("missing 'out2' tensor")
	}

	cands := decodeSegCands(det, coef, infer.OrigW, infer.OrigH, scoreThresh)
	cands = nmsSeg(cands, nmsThresh)

	res := &SegResult{W: infer.OrigW, H: infer.OrigH, Mask: make([]bool, infer.OrigW*infer.OrigH)}
	for _, c := range cands {
		res.Detections = append(res.Detections, c.det)
		inst := instanceMask(c, proto, infer.OrigW, infer.OrigH)
		res.Instances = append(res.Instances, inst)
		// OR the instance into the union mask (original-image space).
		for y := 0; y < inst.H; y++ {
			for x := 0; x < inst.W; x++ {
				if inst.Mask[y*inst.W+x] {
					res.Mask[(inst.OriginY+y)*res.W+(inst.OriginX+x)] = true
				}
			}
		}
	}
	return res, nil
}

func decodeSegCands(det, coef ncnn.Tensor, origW, origH int, scoreThresh float32) []segCand {
	numAnchors, atD := accessor(det, segChannels)
	_, atC := accessor(coef, segMaskCoeff)
	scaleX := float32(origW) / detSize
	scaleY := float32(origH) / detSize

	var cands []segCand
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

				bestC, bestLogit := 0, atD(a, 4*regMax)
				for c := 1; c < segNumClass; c++ {
					if l := atD(a, 4*regMax+c); l > bestLogit {
						bestLogit, bestC = l, c
					}
				}
				score := sigmoid(bestLogit)
				if score < scoreThresh {
					continue
				}

				cx := (float32(x) + 0.5) * float32(stride)
				cy := (float32(y) + 0.5) * float32(stride)
				dl := dflExpect(atD, a, 0*regMax) * float32(stride)
				dt := dflExpect(atD, a, 1*regMax) * float32(stride)
				dr := dflExpect(atD, a, 2*regMax) * float32(stride)
				db := dflExpect(atD, a, 3*regMax) * float32(stride)

				var cd segCand
				cd.det = detect.Detection{
					Box: detect.Box{
						X1: clamp((cx-dl)*scaleX, 0, float32(origW)),
						Y1: clamp((cy-dt)*scaleY, 0, float32(origH)),
						X2: clamp((cx+dr)*scaleX, 0, float32(origW)),
						Y2: clamp((cy+db)*scaleY, 0, float32(origH)),
					},
					Score:   score,
					ClassID: bestC,
					Label:   detect.COCOLabel(bestC),
				}
				for c := 0; c < segMaskCoeff; c++ {
					cd.coeffs[c] = atC(a, c)
				}
				cands = append(cands, cd)
			}
		}
	}
	return cands
}

// instanceMask computes one object's mask from the prototype tensor, cropped to
// its box (nearest-neighbour upsample from proto resolution).
func instanceMask(c segCand, proto ncnn.Tensor, origW, origH int) SegInstance {
	mw, mh, mc := proto.W, proto.H, proto.C
	if mc > segMaskCoeff {
		mc = segMaskCoeff
	}
	b := c.det.Box
	x1, y1 := clampInt(int(b.X1), 0, origW), clampInt(int(b.Y1), 0, origH)
	x2, y2 := clampInt(int(b.X2), 0, origW), clampInt(int(b.Y2), 0, origH)
	bw, bh := x2-x1, y2-y1
	inst := SegInstance{Detection: c.det, W: bw, H: bh, OriginX: x1, OriginY: y1}
	if bw <= 0 || bh <= 0 {
		return inst
	}
	inst.Mask = make([]bool, bw*bh)
	for oy := y1; oy < y2; oy++ {
		py := oy * mh / origH
		if py >= mh {
			py = mh - 1
		}
		for ox := x1; ox < x2; ox++ {
			px := ox * mw / origW
			if px >= mw {
				px = mw - 1
			}
			var logit float32
			for ch := 0; ch < mc; ch++ {
				logit += c.coeffs[ch] * proto.Data[(ch*mh+py)*mw+px]
			}
			if sigmoid(logit) > 0.5 {
				inst.Mask[(oy-y1)*bw+(ox-x1)] = true
			}
		}
	}
	return inst
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func nmsSeg(cands []segCand, iouThresh float32) []segCand {
	sort.SliceStable(cands, func(i, j int) bool { return cands[i].det.Score > cands[j].det.Score })
	kept := make([]segCand, 0, len(cands))
	dead := make([]bool, len(cands))
	for i := range cands {
		if dead[i] {
			continue
		}
		kept = append(kept, cands[i])
		for j := i + 1; j < len(cands); j++ {
			if !dead[j] && cands[j].det.ClassID == cands[i].det.ClassID &&
				detect.IoU(cands[i].det.Box, cands[j].det.Box) > iouThresh {
				dead[j] = true
			}
		}
	}
	return kept
}

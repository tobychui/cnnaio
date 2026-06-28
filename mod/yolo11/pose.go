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
	poseNumKpt   = 17             // COCO person keypoints
	poseChannels = 4*regMax + 1   // out0: 64 box + 1 person class = 65
	poseKptChan  = poseNumKpt * 3 // out1: 17 * (x,y,score) = 51
)

// PoseSkeleton is the COCO 17-keypoint skeleton (pairs of keypoint indices), for
// drawing limbs with render.DrawSkeleton.
var PoseSkeleton = [][2]int{
	{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6},
	{5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4},
	{3, 5}, {4, 6},
}

// PoseDetection is one detected person with its 17 keypoints (image coordinates).
type PoseDetection struct {
	Box       detect.Box
	Score     float32
	Keypoints []detect.Point
}

// PoseDetector runs YOLO11 pose estimation (person + 17 keypoints).
type PoseDetector struct {
	session *ncnn.Session
	param   fs.FS
}

// NewPoseDetector attaches a YOLO11 pose detector to an existing ncnn.Session.
func NewPoseDetector(session *ncnn.Session) (*PoseDetector, error) {
	if session == nil {
		return nil, fmt.Errorf("nil ncnn.Session")
	}
	sub, err := modelsFS()
	if err != nil {
		return nil, err
	}
	return &PoseDetector{session: session, param: sub}, nil
}

// Detect returns detected people with keypoints, in original-image coordinates.
func (d *PoseDetector) Detect(ctx context.Context, image []byte, scoreThresh, nmsThresh float32) ([]PoseDetection, error) {
	infer, err := runInfer(ctx, d.session, d.param, "yolo11n_pose.ncnn.param", "yolo11n_pose.ncnn.bin",
		detSize, "in0", []string{"out0", "out1"}, image)
	if err != nil {
		return nil, err
	}
	box, ok := infer.Tensor("out0")
	if !ok {
		return nil, fmt.Errorf("missing 'out0' tensor")
	}
	kpt, ok := infer.Tensor("out1")
	if !ok {
		return nil, fmt.Errorf("missing 'out1' tensor")
	}

	dets := decodePose(box, kpt, infer.OrigW, infer.OrigH, scoreThresh)
	return nmsPose(dets, nmsThresh), nil
}

func decodePose(box, kpt ncnn.Tensor, origW, origH int, scoreThresh float32) []PoseDetection {
	numAnchors, atB := accessor(box, poseChannels)
	_, atK := accessor(kpt, poseKptChan)
	scaleX := float32(origW) / detSize
	scaleY := float32(origH) / detSize

	var dets []PoseDetection
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

				score := sigmoid(atB(a, 4*regMax)) // single "person" class
				if score < scoreThresh {
					continue
				}

				cx := (float32(x) + 0.5) * float32(stride)
				cy := (float32(y) + 0.5) * float32(stride)
				dl := dflExpect(atB, a, 0*regMax) * float32(stride)
				dt := dflExpect(atB, a, 1*regMax) * float32(stride)
				dr := dflExpect(atB, a, 2*regMax) * float32(stride)
				db := dflExpect(atB, a, 3*regMax) * float32(stride)

				pts := make([]detect.Point, poseNumKpt)
				for k := 0; k < poseNumKpt; k++ {
					// kpt_xy = (raw*2 + grid) * stride  (ultralytics decode)
					kx := (atK(a, k*3)*2 + float32(x)) * float32(stride)
					ky := (atK(a, k*3+1)*2 + float32(y)) * float32(stride)
					pts[k] = detect.Point{X: kx * scaleX, Y: ky * scaleY}
				}

				dets = append(dets, PoseDetection{
					Box: detect.Box{
						X1: clamp((cx-dl)*scaleX, 0, float32(origW)),
						Y1: clamp((cy-dt)*scaleY, 0, float32(origH)),
						X2: clamp((cx+dr)*scaleX, 0, float32(origW)),
						Y2: clamp((cy+db)*scaleY, 0, float32(origH)),
					},
					Score:     score,
					Keypoints: pts,
				})
			}
		}
	}
	return dets
}

// nmsPose is greedy NMS over PoseDetection boxes.
func nmsPose(dets []PoseDetection, iouThresh float32) []PoseDetection {
	sort.SliceStable(dets, func(i, j int) bool { return dets[i].Score > dets[j].Score })
	kept := make([]PoseDetection, 0, len(dets))
	dead := make([]bool, len(dets))
	for i := range dets {
		if dead[i] {
			continue
		}
		kept = append(kept, dets[i])
		for j := i + 1; j < len(dets); j++ {
			if !dead[j] && detect.IoU(dets[i].Box, dets[j].Box) > iouThresh {
				dead[j] = true
			}
		}
	}
	return kept
}

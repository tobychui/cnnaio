// Package genderdetector classifies the gender of a face (male / female) using a
// MobileNetV2-0.35 model (Seeed sscma model zoo) on a shared ncnn.Session.
//
// Like mod/facerecognition it accepts either an already-cropped face image or a
// full photo (auto-cropping the largest face via mod/facedetector). The model
// takes a 64x64 RGB face crop and outputs two probabilities (female, male).
package genderdetector

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
	"cnnaio/mod/facedetector"
	"cnnaio/mod/ncnn"
)

//go:embed models/mbv2_0.35_rep_gender.param models/mbv2_0.35_rep_gender.bin
var models embed.FS

const (
	inputSize  = 64
	cropMargin = 0.2 // include a little context around the face box
)

// classes are the two output indices, in the model's order (female, male).
var classes = [2]string{"female", "male"}

// RGB input normalization applied by the host (model has no built-in norm).
// substract_mean_normalize computes (pixel-mean)*norm. ImageNet mean/std (RGB)
// is what this model expects — it was verified empirically: with /255 the model
// labels every face "female", while ImageNet normalization correctly separates
// male and female faces.
var (
	mean = [3]float64{123.675, 116.28, 103.53}
	norm = [3]float64{1.0 / 58.395, 1.0 / 57.12, 1.0 / 57.375}
)

// Result is one gender prediction.
type Result struct {
	Label      string     // "male" or "female"
	Confidence float32    // probability of the predicted class (normalized to [0,1])
	Female     float32    // raw model probability for "female"
	Male       float32    // raw model probability for "male"
	Box        detect.Box // face box used (zero for a cropped input)
}

// Sample is one input. If Cropped is true the whole image is treated as a face
// crop; otherwise it is a full photo and the largest detected face is used.
type Sample struct {
	Image   []byte
	Cropped bool
}

// Detector runs gender classification on a shared ncnn.Session.
type Detector struct {
	session  *ncnn.Session
	modelsFS fs.FS
	faces    *facedetector.Detector
}

// New attaches a gender detector to an existing ncnn.Session. It also builds a
// facedetector on the same session for auto-cropping full photos.
func New(session *ncnn.Session) (*Detector, error) {
	if session == nil {
		return nil, fmt.Errorf("nil ncnn.Session")
	}
	sub, err := fs.Sub(models, "models")
	if err != nil {
		return nil, fmt.Errorf("sub models fs: %w", err)
	}
	fd, err := facedetector.New(session)
	if err != nil {
		return nil, fmt.Errorf("init facedetector: %w", err)
	}
	return &Detector{session: session, modelsFS: sub, faces: fd}, nil
}

// Classify predicts gender for a sample (full photo -> auto-crop largest face,
// or an already-cropped face).
func (d *Detector) Classify(ctx context.Context, s Sample) (Result, error) {
	var box detect.Box
	rx, ry, rw, rh := 0, 0, 0, 0 // 0 ROI => whole image

	if !s.Cropped {
		b, err := d.largestFace(ctx, s.Image)
		if err != nil {
			return Result{}, err
		}
		box = b
		imgW, imgH, err := imageSize(s.Image)
		if err != nil {
			return Result{}, fmt.Errorf("read image size: %w", err)
		}
		rx, ry, rw, rh = detect.SquareROI(box, imgW, imgH, cropMargin)
		if rw <= 0 || rh <= 0 {
			return Result{}, fmt.Errorf("empty face roi")
		}
	}

	probs, err := d.infer(ctx, s.Image, rx, ry, rw, rh)
	if err != nil {
		return Result{}, err
	}

	res := Result{Female: probs[0], Male: probs[1], Box: box}
	sum := probs[0] + probs[1]
	if sum <= 0 {
		sum = 1
	}
	if probs[1] >= probs[0] {
		res.Label, res.Confidence = classes[1], probs[1]/sum
	} else {
		res.Label, res.Confidence = classes[0], probs[0]/sum
	}
	return res, nil
}

// ClassifyPhoto is a convenience for a full photo (auto-crops the largest face).
func (d *Detector) ClassifyPhoto(ctx context.Context, photo []byte) (Result, error) {
	return d.Classify(ctx, Sample{Image: photo})
}

// ClassifyCropped is a convenience for an already-cropped face image.
func (d *Detector) ClassifyCropped(ctx context.Context, faceImage []byte) (Result, error) {
	return d.Classify(ctx, Sample{Image: faceImage, Cropped: true})
}

func (d *Detector) infer(ctx context.Context, img []byte, rx, ry, rw, rh int) ([2]float32, error) {
	args := []string{
		"ncnn", "infer",
		"/models/mbv2_0.35_rep_gender.param",
		"/models/mbv2_0.35_rep_gender.bin",
		"1", // text param
		ncnn.ImageGuestPath + "/image",
		strconv.Itoa(inputSize), strconv.Itoa(inputSize),
		"0", // keep RGB
		ftoa(mean[0]), ftoa(mean[1]), ftoa(mean[2]),
		ftoa(norm[0]), ftoa(norm[1]), ftoa(norm[2]),
		strconv.Itoa(rx), strconv.Itoa(ry), strconv.Itoa(rw), strconv.Itoa(rh),
		"input", // input blob
		"1",     // one output
		"output",
	}

	run, err := d.session.Run(ctx, ncnn.RunRequest{
		Args: args,
		Mounts: map[string]fs.FS{
			"/models":           d.modelsFS,
			ncnn.ImageGuestPath: ncnn.ImageFS(img),
		},
	})
	if err != nil {
		return [2]float32{}, fmt.Errorf("gender inference failed: %w\nstderr: %s", err, run.Stderr)
	}

	infer, err := ncnn.ParseInferOutput(run.Stdout)
	if err != nil {
		return [2]float32{}, fmt.Errorf("parse infer output: %w", err)
	}
	out, ok := infer.Tensor("output")
	if !ok {
		return [2]float32{}, fmt.Errorf("missing 'output' tensor")
	}
	if len(out.Data) < 2 {
		return [2]float32{}, fmt.Errorf("gender tensor too small: %d", len(out.Data))
	}
	return [2]float32{out.Data[0], out.Data[1]}, nil
}

func (d *Detector) largestFace(ctx context.Context, photo []byte) (detect.Box, error) {
	faces, err := d.faces.Detect(ctx, photo, 0.7, 0.3)
	if err != nil {
		return detect.Box{}, fmt.Errorf("face detect: %w", err)
	}
	if len(faces) == 0 {
		return detect.Box{}, fmt.Errorf("no face found in photo")
	}
	best := faces[0]
	for _, f := range faces[1:] {
		if f.Box.Area() > best.Box.Area() {
			best = f
		}
	}
	return best.Box, nil
}

func imageSize(data []byte) (int, int, error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return 0, 0, err
	}
	return cfg.Width, cfg.Height, nil
}

func ftoa(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }

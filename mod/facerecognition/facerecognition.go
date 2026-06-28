// Package facerecognition provides face recognition (verification / matching) on
// a shared ncnn.Session using MBV2FaceNet (a MobileNetV2-backbone FaceNet). It
// maps a 112x112 face to a 128-d embedding; cosine similarity measures identity.
//
// It accepts both already-cropped face images and full photos (auto-cropping the
// largest face via mod/facedetector), so matching works cropped-vs-cropped,
// cropped-vs-photo, or photo-vs-photo. Each face is aligned to the canonical
// 5-point template (keypoints from PFLD landmarks) before embedding — see
// align.go — which is essential for usable accuracy.
//
// (The package also ships RetinaFace weights as an alternative keypoint source;
// its anchor decode is not implemented here.)
package facerecognition

import (
	"context"
	"embed"
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"io/fs"
	"math"
	"strconv"

	"cnnaio/mod/detect"
	"cnnaio/mod/facedetector"
	"cnnaio/mod/landmark"
	"cnnaio/mod/ncnn"
)

//go:embed models/mbv2facenet.param models/mbv2facenet.bin
var models embed.FS

const (
	inputSize = 112 // MBV2FaceNet square input
	embedDim  = 128 // length of the fc1 embedding
)

// MBV2FaceNet normalizes internally (BinaryOp (x-127.5)*0.0078125), so feed RAW
// RGB pixels [0,255] with NO mean/norm here — double-normalizing collapses output.
var (
	mean = [3]float64{0, 0, 0}
	norm = [3]float64{1, 1, 1}
)

// DefaultThreshold is a reasonable cosine-similarity cutoff for "same person".
const DefaultThreshold = 0.5

// Embedding is an L2-normalized 128-d face feature vector.
type Embedding []float32

// Sample is one input to match. If Cropped is true the whole image is treated as
// a face crop; otherwise it is a full photo and the largest detected face is used.
type Sample struct {
	Image   []byte
	Cropped bool
}

// Result is the outcome of matching two samples.
type Result struct {
	Similarity float32    // cosine similarity in [-1,1]
	Same       bool       // Similarity >= threshold
	BoxA, BoxB detect.Box // face boxes used (zero for cropped inputs)
}

// Matcher runs MBV2FaceNet on a shared ncnn.Session. For full photos it locates
// a face (facedetector) and aligns it to the canonical 112x112 template using
// 5 keypoints derived from PFLD landmarks before embedding — this alignment is
// what makes the embeddings discriminative enough for verification.
type Matcher struct {
	session  *ncnn.Session
	modelsFS fs.FS
	faces    *facedetector.Detector
	marks    *landmark.Detector
}

// New attaches a face matcher to an existing ncnn.Session. It also builds a
// facedetector and a PFLD landmark detector on the same session (for locating
// and aligning faces in full photos).
func New(session *ncnn.Session) (*Matcher, error) {
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
	lm, err := landmark.New(session)
	if err != nil {
		return nil, fmt.Errorf("init landmark: %w", err)
	}
	return &Matcher{session: session, modelsFS: sub, faces: fd, marks: lm}, nil
}

// Embed computes the face embedding for a sample. It detects the largest face,
// aligns it to the canonical template via 5 keypoints, and embeds the aligned
// crop — returning the face box used. If no face can be located/aligned, a
// cropped sample falls back to a plain whole-image resize; a full photo errors.
func (m *Matcher) Embed(ctx context.Context, s Sample) (Embedding, detect.Box, error) {
	aligned, box, err := m.alignedFace(ctx, s.Image)
	if err == nil {
		emb, e := m.embed(ctx, aligned, 0, 0, 0, 0) // aligned face fills the frame
		return emb, box, e
	}
	if s.Cropped {
		// Already-cropped face the detector/landmarker couldn't handle: embed as-is.
		emb, e := m.embed(ctx, s.Image, 0, 0, 0, 0)
		return emb, detect.Box{}, e
	}
	return nil, detect.Box{}, err
}

// alignedFace locates the largest face, derives 5 ArcFace keypoints from PFLD
// landmarks, warps the face to the canonical 112x112 template, and returns it as
// PNG bytes ready to embed.
func (m *Matcher) alignedFace(ctx context.Context, img []byte) ([]byte, detect.Box, error) {
	box, err := m.largestFace(ctx, img)
	if err != nil {
		return nil, detect.Box{}, err
	}
	pts, err := m.marks.Detect(ctx, img, box)
	if err != nil {
		return nil, box, fmt.Errorf("landmark: %w", err)
	}
	five, ok := fivePoints(pts)
	if !ok {
		return nil, box, fmt.Errorf("insufficient landmarks (%d)", len(pts))
	}
	src, err := decodeRGBA(img)
	if err != nil {
		return nil, box, fmt.Errorf("decode image: %w", err)
	}
	a, b, tx, ty := solveSimilarity(five)
	out, err := encodePNG(warpFace(src, a, b, tx, ty, inputSize))
	if err != nil {
		return nil, box, fmt.Errorf("encode aligned face: %w", err)
	}
	return out, box, nil
}

// EmbedCropped is a convenience for an already-cropped face image.
func (m *Matcher) EmbedCropped(ctx context.Context, faceImage []byte) (Embedding, error) {
	emb, _, err := m.Embed(ctx, Sample{Image: faceImage, Cropped: true})
	return emb, err
}

// EmbedPhoto is a convenience for a full photo (auto-crops the largest face).
func (m *Matcher) EmbedPhoto(ctx context.Context, photo []byte) (Embedding, detect.Box, error) {
	return m.Embed(ctx, Sample{Image: photo, Cropped: false})
}

// Match embeds both samples and compares them with the given cosine threshold
// (use DefaultThreshold if unsure).
func (m *Matcher) Match(ctx context.Context, a, b Sample, threshold float32) (Result, error) {
	ea, boxA, err := m.Embed(ctx, a)
	if err != nil {
		return Result{}, fmt.Errorf("embed A: %w", err)
	}
	eb, boxB, err := m.Embed(ctx, b)
	if err != nil {
		return Result{}, fmt.Errorf("embed B: %w", err)
	}
	sim := Similarity(ea, eb)
	return Result{Similarity: sim, Same: sim >= threshold, BoxA: boxA, BoxB: boxB}, nil
}

// Similarity returns the cosine similarity of two embeddings. Since embeddings
// are L2-normalized, this is simply their dot product.
func Similarity(a, b Embedding) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

// embed runs MBV2FaceNet over the given ROI (0,0,0,0 = whole image) and returns
// the L2-normalized embedding.
func (m *Matcher) embed(ctx context.Context, img []byte, rx, ry, rw, rh int) (Embedding, error) {
	args := []string{
		"ncnn", "infer",
		"/models/mbv2facenet.param",
		"/models/mbv2facenet.bin",
		"1", // text param
		ncnn.ImageGuestPath + "/image",
		strconv.Itoa(inputSize), strconv.Itoa(inputSize),
		"0", // keep RGB (model handles normalization internally)
		ftoa(mean[0]), ftoa(mean[1]), ftoa(mean[2]),
		ftoa(norm[0]), ftoa(norm[1]), ftoa(norm[2]),
		strconv.Itoa(rx), strconv.Itoa(ry), strconv.Itoa(rw), strconv.Itoa(rh),
		"data", // input blob
		"1",    // one output
		"fc1",
	}

	run, err := m.session.Run(ctx, ncnn.RunRequest{
		Args: args,
		Mounts: map[string]fs.FS{
			"/models":           m.modelsFS,
			ncnn.ImageGuestPath: ncnn.ImageFS(img),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("mbv2facenet inference failed: %w\nstderr: %s", err, run.Stderr)
	}

	infer, err := ncnn.ParseInferOutput(run.Stdout)
	if err != nil {
		return nil, fmt.Errorf("parse infer output: %w", err)
	}
	out, ok := infer.Tensor("fc1")
	if !ok {
		return nil, fmt.Errorf("missing 'fc1' tensor")
	}
	if len(out.Data) < embedDim {
		return nil, fmt.Errorf("embedding too small: %d (want %d)", len(out.Data), embedDim)
	}

	return l2normalize(out.Data[:embedDim]), nil
}

// largestFace detects faces in a photo and returns the highest-area box.
func (m *Matcher) largestFace(ctx context.Context, photo []byte) (detect.Box, error) {
	faces, err := m.faces.Detect(ctx, photo, 0.7, 0.3)
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

func l2normalize(v []float32) Embedding {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	n := float32(math.Sqrt(sum))
	if n == 0 {
		n = 1
	}
	out := make(Embedding, len(v))
	for i, x := range v {
		out[i] = x / n
	}
	return out
}

func ftoa(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }

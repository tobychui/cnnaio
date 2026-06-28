package yolo11

import (
	"context"
	_ "embed"
	"fmt"
	"io/fs"
	"sort"
	"strings"

	"cnnaio/mod/ncnn"
)

const clsSize = 224 // YOLO11-cls input size

//go:embed models/imagenet.txt
var imagenetTxt string

// Prediction is one ranked class result from the classifier.
type Prediction struct {
	Index int
	Score float32
	Label string
}

// Classifier runs YOLO11 image classification (ImageNet, 1000 classes).
type Classifier struct {
	session *ncnn.Session
	param   fs.FS
	labels  []string
}

// NewClassifier attaches a YOLO11 classifier to an existing ncnn.Session.
func NewClassifier(session *ncnn.Session) (*Classifier, error) {
	if session == nil {
		return nil, fmt.Errorf("nil ncnn.Session")
	}
	sub, err := modelsFS()
	if err != nil {
		return nil, err
	}
	return &Classifier{session: session, param: sub, labels: parseImageNet(imagenetTxt)}, nil
}

// Classify returns the top-k ImageNet predictions for an image. topK <= 0 -> 5.
func (c *Classifier) Classify(ctx context.Context, image []byte, topK int) ([]Prediction, error) {
	if topK <= 0 {
		topK = 5
	}
	infer, err := runInfer(ctx, c.session, c.param, "yolo11n_cls.ncnn.param", "yolo11n_cls.ncnn.bin",
		clsSize, "in0", []string{"out0"}, image)
	if err != nil {
		return nil, err
	}
	out, ok := infer.Tensor("out0")
	if !ok {
		return nil, fmt.Errorf("missing 'out0' tensor")
	}

	// out0 is already a softmax distribution; rank it.
	idx := make([]int, len(out.Data))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool { return out.Data[idx[i]] > out.Data[idx[j]] })
	if topK > len(idx) {
		topK = len(idx)
	}
	preds := make([]Prediction, topK)
	for k := 0; k < topK; k++ {
		i := idx[k]
		preds[k] = Prediction{Index: i, Score: out.Data[i], Label: c.labelOf(i)}
	}
	return preds, nil
}

func (c *Classifier) labelOf(i int) string {
	if i >= 0 && i < len(c.labels) {
		return c.labels[i]
	}
	return fmt.Sprintf("class_%d", i)
}

// parseImageNet turns lines like "'n01440764 tench, Tinca tinca'" into "tench,
// Tinca tinca" (strip quotes and the leading WNID).
func parseImageNet(s string) []string {
	var out []string
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(strings.Trim(strings.TrimSpace(line), "'"))
		if line == "" {
			continue
		}
		if sp := strings.IndexByte(line, ' '); sp >= 0 {
			line = line[sp+1:]
		}
		out = append(out, line)
	}
	return out
}

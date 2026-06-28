// Package mobilenet provides MobileNet image classification on top of a shared
// ncnn.Session (see package ncnn). The model files are embedded via go:embed, so
// a built binary needs no external files; the wazero runtime itself is owned by
// the ncnn.Session, which can be reused by other classifiers too.
package mobilenet

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"strconv"
	"strings"
	"time"

	"cnnaio/mod/ncnn"
)

// models holds the MobileNet param/model/synset files. They are mounted into the
// wasm sandbox at /models, where the C driver reads them by path.
//
//go:embed models/*.bin models/*.txt
var models embed.FS

// Variant describes one MobileNet model plus the preprocessing the network
// expects. The fields map 1:1 onto the argv the C driver (build/classify.c) takes.
type Variant struct {
	Name   string     // human-readable label for output
	Param  string     // file name inside models/ (binary ncnn param)
	Model  string     // file name inside models/ (weights)
	Synset string     // file name inside models/ (one label per line)
	Target int        // square input size the net expects (e.g. 224)
	ToBGR  bool       // convert decoded RGB -> BGR before feeding the net
	Mean   [3]float64 // per-channel mean (in the net's channel order)
	Norm   [3]float64 // per-channel scale applied after mean subtraction
}

// Built-in variants matching the bundled model files.
var (
	// V2 is MobileNetV2 trained on ImageNet (1000 classes), 224x224, BGR input.
	V2 = Variant{
		Name:   "MobileNetV2 / ImageNet-1000",
		Param:  "mobilenet_v2.param.bin",
		Model:  "mobilenet_v2.bin",
		Synset: "synset_v2.txt",
		Target: 224,
		ToBGR:  true,
		Mean:   [3]float64{103.94, 116.78, 123.68},
		Norm:   [3]float64{0.017, 0.017, 0.017},
	}
)

// Prediction is one ranked class result.
type Prediction struct {
	Index int     // class index within the synset
	Score float64 // probability in [0,1]
	Label string  // synset label text
}

// Result is the outcome of one Classify call.
type Result struct {
	Variant     string
	Predictions []Prediction
	Duration    time.Duration // wall-clock time of the wasm round-trip
	Log         string        // raw diagnostic output the wasm wrote to stderr
}

// MobileNetClassifier runs MobileNet inference on top of a shared ncnn.Session.
// It does not own the runtime — many classifiers can share one Session — so there
// is no Close method here; close the Session when you're done with all of them.
type MobileNetClassifier struct {
	session  *ncnn.Session
	modelsFS fs.FS // models/ subtree, mounted at /models in the guest
}

// NewMobileNetClassifier wires a classifier onto an existing ncnn.Session. The
// Session carries the wazero runtime + compiled wasm and may be reused by other
// classifiers/models, so a single session is shared across the whole program.
func NewMobileNetClassifier(session *ncnn.Session) (*MobileNetClassifier, error) {
	if session == nil {
		return nil, fmt.Errorf("nil ncnn.Session")
	}

	modelsFS, err := fs.Sub(models, "models")
	if err != nil {
		return nil, fmt.Errorf("sub models fs: %w", err)
	}

	return &MobileNetClassifier{session: session, modelsFS: modelsFS}, nil
}

// The C driver (build/classify.c) writes results to stdout as tab-delimited
// lines and all diagnostics to stderr. Each result line is:
//
//	PRED \t <rank> \t <index> \t <score 0..1> \t <label>
const predPrefix = "PRED\t"

// Classify runs the given MobileNet variant over a decoded-image's raw bytes
// (JPEG, PNG, BMP, … via stb_image; WebP is transcoded to PNG by ncnn.ImageFS).
// topK <= 0 defaults to 5.
//
// The model files come from the embedded fs; the image is handed to the sandbox
// through an in-memory fs mounted at /input, so nothing touches the host disk.
func (c *MobileNetClassifier) Classify(ctx context.Context, v Variant, image []byte, topK int) (*Result, error) {
	if topK <= 0 {
		topK = 5
	}

	// argv: [program, subcommand, ...] — positional order defined by build/classify.c.
	args := []string{
		"ncnn", "classify",
		"/models/" + v.Param,
		"/models/" + v.Model,
		"/models/" + v.Synset,
		"/input/image",
		strconv.Itoa(v.Target), strconv.Itoa(v.Target),
		boolArg(v.ToBGR),
		f(v.Mean[0]), f(v.Mean[1]), f(v.Mean[2]),
		f(v.Norm[0]), f(v.Norm[1]), f(v.Norm[2]),
		strconv.Itoa(topK),
	}

	// Hand execution to the shared session: it mounts the embedded models and
	// the in-memory image fs, runs the wasm, and captures stdout/stderr.
	run, err := c.session.Run(ctx, ncnn.RunRequest{
		Args: args,
		Mounts: map[string]fs.FS{
			"/models":           c.modelsFS,
			ncnn.ImageGuestPath: ncnn.ImageFS(image),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("wasm inference failed: %w\nstderr: %s", err, strings.TrimSpace(run.Stderr))
	}

	preds, err := parsePredictions(run.Stdout)
	if err != nil {
		return nil, fmt.Errorf("%w\nstderr: %s", err, strings.TrimSpace(run.Stderr))
	}

	return &Result{
		Variant:     v.Name,
		Predictions: preds,
		Duration:    run.Duration,
		Log:         strings.TrimSpace(run.Stderr), // diagnostics live on stderr
	}, nil
}

// parsePredictions extracts the ranked results from the C driver's stdout.
// Lines are: PRED \t rank \t index \t score \t label  (see predPrefix).
func parsePredictions(out string) ([]Prediction, error) {
	var preds []Prediction
	for _, line := range strings.Split(out, "\n") {
		if !strings.HasPrefix(line, predPrefix) {
			continue
		}
		// fields: [PRED, rank, index, score, label]; label may contain spaces
		// but never tabs, so SplitN with 5 is safe.
		fields := strings.SplitN(strings.TrimRight(line, "\r"), "\t", 5)
		if len(fields) != 5 {
			return nil, fmt.Errorf("malformed PRED line: %q", line)
		}
		idx, err := strconv.Atoi(fields[2])
		if err != nil {
			return nil, fmt.Errorf("bad index in %q: %w", line, err)
		}
		score, err := strconv.ParseFloat(fields[3], 64)
		if err != nil {
			return nil, fmt.Errorf("bad score in %q: %w", line, err)
		}
		preds = append(preds, Prediction{Index: idx, Score: score, Label: fields[4]})
	}
	if len(preds) == 0 {
		return nil, fmt.Errorf("no predictions parsed from wasm output")
	}
	return preds, nil
}

func boolArg(b bool) string {
	if b {
		return "1"
	}
	return "0"
}

func f(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}

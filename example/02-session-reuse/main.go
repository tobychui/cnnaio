// Example 2 — Session reuse and a conditional pipeline.
//
// A single ncnn.Session is shared across *three* models, which is the whole
// point of the Session design: building it compiles the wasm once, then every
// model reuses that one runtime instead of spawning its own.
//
// Pipeline:
//
//	image ─▶ object detection (YOLO11)        always
//	      ─▶ face detection   (Ultra-Light)   always
//	          └─ if any face ─▶ landmarks (PFLD, 98 pts)   conditional
//
// All results are written to ./output.json, and the session is explicitly
// closed at the end so the wazero runtime is released cleanly.
//
// Run it:
//
//	go run ./example/02-session-reuse                 # uses the bundled test.png
//	go run ./example/02-session-reuse path/to/img.jpg # your own image
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"cnnaio/mod/detect"
	"cnnaio/mod/facedetector"
	"cnnaio/mod/landmark"
	"cnnaio/mod/ncnn"
	"cnnaio/mod/yolo11"
)

const (
	defaultImage = "example/testdata/test.png"
	outputFile   = "output.json"
)

// report is the JSON document written to output.json.
type report struct {
	Image   string       `json:"image"`
	Objects []objectJSON `json:"objects"`
	Faces   []faceJSON   `json:"faces"`
	HasFace bool         `json:"has_face"`
}

type objectJSON struct {
	Label string     `json:"label"`
	Score float32    `json:"score"`
	Box   detect.Box `json:"box"`
}

type faceJSON struct {
	Score     float32        `json:"score"`
	Box       detect.Box     `json:"box"`
	Landmarks []detect.Point `json:"landmarks,omitempty"` // 98 PFLD points, when computed
}

func main() {
	imagePath := defaultImage
	if len(os.Args) > 1 {
		imagePath = os.Args[1]
	}
	img, err := os.ReadFile(imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot read image %q: %v\n", imagePath, err)
		os.Exit(1)
	}

	ctx := context.Background()

	// One session, reused by every model below.
	session, err := ncnn.NewNcnnSession()
	if err != nil {
		fmt.Fprintf(os.Stderr, "init ncnn session: %v\n", err)
		os.Exit(1)
	}
	// Run the pipeline in a helper so that the explicit session.Close below always
	// happens before we exit (defer in main would also work; this makes the
	// "terminate the session correctly" step obvious).
	rep := run(ctx, session, imagePath, img)

	if err := session.Close(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "warning: closing session: %v\n", err)
	}

	// Write output.json.
	out, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal results: %v\n", err)
		os.Exit(1)
	}
	if err := os.WriteFile(outputFile, out, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write %s: %v\n", outputFile, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %d object(s) and %d face(s) to %s\n", len(rep.Objects), len(rep.Faces), outputFile)
}

func run(ctx context.Context, s *ncnn.Session, imagePath string, img []byte) report {
	rep := report{Image: imagePath}

	// ── Model 1: object detection ────────────────────────────────────────────
	det, err := yolo11.New(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init yolo11: %v\n", err)
		return rep
	}
	objects, err := det.Detect(ctx, img, 0.25, 0.45)
	if err != nil {
		fmt.Fprintf(os.Stderr, "detect: %v\n", err)
	}
	for _, o := range objects {
		rep.Objects = append(rep.Objects, objectJSON{Label: o.Label, Score: o.Score, Box: o.Box})
	}
	fmt.Printf("objects: %d\n", len(objects))

	// ── Model 2: face detection (same session) ───────────────────────────────
	fd, err := facedetector.New(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init facedetector: %v\n", err)
		return rep
	}
	faces, err := fd.Detect(ctx, img, 0.7, 0.3)
	if err != nil {
		fmt.Fprintf(os.Stderr, "face detect: %v\n", err)
	}
	rep.HasFace = len(faces) > 0
	fmt.Printf("faces:   %d\n", len(faces))

	// ── Model 3: landmarks — only if we actually found a face ────────────────
	var lm *landmark.Detector
	if rep.HasFace {
		if lm, err = landmark.New(s); err != nil {
			fmt.Fprintf(os.Stderr, "init landmark: %v\n", err)
		}
	}

	for i, f := range faces {
		fj := faceJSON{Score: f.Score, Box: f.Box}
		if lm != nil {
			pts, err := lm.Detect(ctx, img, f.Box)
			if err != nil {
				fmt.Fprintf(os.Stderr, "landmark (face %d): %v\n", i+1, err)
			} else {
				fj.Landmarks = pts
				fmt.Printf("  face %d: %d landmarks\n", i+1, len(pts))
			}
		}
		rep.Faces = append(rep.Faces, fj)
	}

	return rep
}

// Example: use cnnaio as a library. It runs the ncnn inference runtime *inside*
// a pure-Go WebAssembly runtime (wazero) — no cgo, no native libs, cross-platform;
// all models and the wasm are embedded via go:embed.
//
// One shared ncnn.Session drives every model package:
//   - mobilenet / yolo11(cls) : image classification
//   - yolo11 / nanodet        : object detection (COCO-80)
//   - yolo11(seg/pose/obb)    : segmentation, pose, oriented boxes
//   - facedetector            : face detection (Ultra-Light)
//   - landmark                : 98-point facial landmarks (PFLD)
//   - facerecognition         : face embedding + matching
//   - genderdetector          : gender classification
//
// Run it over an image (results rendered to ./output/*.png):
//
//	go run ./example/full-pipeline                # bundled test.png, full pipeline
//	go run ./example/full-pipeline my.jpg         # full pipeline on one image
//	go run ./example/full-pipeline a.jpg b.jpg    # face matching between two images
//
// See the focused single-task examples (01–03) alongside this one, and
// example/README.md, for minimal copy-pasteable snippets.
package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"

	"cnnaio/mod/detect"
	"cnnaio/mod/facedetector"
	"cnnaio/mod/facerecognition"
	"cnnaio/mod/genderdetector"
	"cnnaio/mod/landmark"
	"cnnaio/mod/mobilenet"
	"cnnaio/mod/nanodet"
	"cnnaio/mod/ncnn"
	"cnnaio/mod/render"
	"cnnaio/mod/yolo11"
)

const outputDir = "output"

func main() {
	ctx := context.Background()

	// One shared session owns the wazero runtime + compiled wasm; every model
	// below reuses it instead of spawning its own runtime.
	fmt.Println("Initialising ncnn (wazero) session ...")
	session, err := ncnn.NewNcnnSession()
	if err != nil {
		fmt.Fprintf(os.Stderr, "init ncnn session: %v\n", err)
		os.Exit(1)
	}
	defer session.Close(ctx)

	// Two image args -> face matching mode; otherwise the single-image pipeline.
	if len(os.Args) > 2 {
		matchFaces(ctx, session, os.Args[1], os.Args[2])
		return
	}

	imagePath := "example/testdata/test.png"
	if len(os.Args) > 1 {
		imagePath = os.Args[1]
	}
	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot read image %q: %v\n", imagePath, err)
		os.Exit(1)
	}

	classify(ctx, session, imagePath, imageBytes)
	classifyYolo(ctx, session, imagePath, imageBytes)
	objects := detectObjects(ctx, session, imagePath, imageBytes)
	yoloDets := detectYolo(ctx, session, imagePath, imageBytes)
	faces, faceLandmarks := detectFaces(ctx, session, imagePath, imageBytes)
	classifyGender(ctx, session, imagePath, imageBytes)
	poses := detectPose(ctx, session, imagePath, imageBytes)
	obbs := detectOBB(ctx, session, imagePath, imageBytes)
	seg := segment(ctx, session, imagePath, imageBytes)

	renderDebug(imagePath, imageBytes, scene{
		nanodet: objects,
		yolo:    yoloDets,
		faces:   faces,
		marks:   faceLandmarks,
		poses:   poses,
		obbs:    obbs,
		seg:     seg,
	})
}

// matchFaces compares the faces in two images (each may be a full photo or an
// already-cropped face) using MobileFaceNet on the shared session.
func matchFaces(ctx context.Context, s *ncnn.Session, pathA, pathB string) {
	imgA, err := os.ReadFile(pathA)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %q: %v\n", pathA, err)
		os.Exit(1)
	}
	imgB, err := os.ReadFile(pathB)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %q: %v\n", pathB, err)
		os.Exit(1)
	}

	header("MBV2FaceNet / face matching", pathA+"  vs  "+pathB)

	m, err := facerecognition.New(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init matcher: %v\n", err)
		return
	}

	// Treat each input as a full photo (auto-crop the largest face).
	res, err := m.Match(ctx,
		facerecognition.Sample{Image: imgA, Cropped: false},
		facerecognition.Sample{Image: imgB, Cropped: false},
		facerecognition.DefaultThreshold)
	if err != nil {
		fmt.Fprintf(os.Stderr, "match error: %v\n", err)
		return
	}

	verdict := "DIFFERENT people"
	if res.Same {
		verdict = "SAME person"
	}
	fmt.Printf(" cosine similarity: %.4f  (threshold %.2f)\n", res.Similarity, facerecognition.DefaultThreshold)
	fmt.Printf(" verdict: %s\n", verdict)
}

func classify(ctx context.Context, s *ncnn.Session, imagePath string, img []byte) {
	clf, err := mobilenet.NewMobileNetClassifier(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init classifier: %v\n", err)
		return
	}
	v := mobilenet.V2
	header(v.Name, imagePath)
	res, err := clf.Classify(ctx, v, img, 5)
	if err != nil {
		fmt.Fprintf(os.Stderr, "classify error: %v\n", err)
		return
	}
	for i, p := range res.Predictions {
		fmt.Printf("%2d. [%4d] %6.2f%%  %s\n", i+1, p.Index, p.Score*100, p.Label)
	}
	fmt.Printf("[host] inference round-trip: %s\n", res.Duration.Round(1e6))
}

func detectObjects(ctx context.Context, s *ncnn.Session, imagePath string, img []byte) []detect.Detection {
	header("NanoDet-Plus / COCO object detection", imagePath)
	det, err := nanodet.New(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init nanodet: %v\n", err)
		return nil
	}
	dets, err := det.Detect(ctx, img, 0.4, 0.5)
	if err != nil {
		fmt.Fprintf(os.Stderr, "nanodet error: %v\n", err)
		return nil
	}
	printDetections(dets)
	return dets
}

func detectYolo(ctx context.Context, s *ncnn.Session, imagePath string, img []byte) []detect.Detection {
	header("YOLO11n / COCO object detection", imagePath)
	det, err := yolo11.New(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init yolo11: %v\n", err)
		return nil
	}
	dets, err := det.Detect(ctx, img, 0.25, 0.45)
	if err != nil {
		fmt.Fprintf(os.Stderr, "yolo11 error: %v\n", err)
		return nil
	}
	printDetections(dets)
	return dets
}

func classifyYolo(ctx context.Context, s *ncnn.Session, imagePath string, img []byte) {
	header("YOLO11n-cls / ImageNet classification", imagePath)
	clf, err := yolo11.NewClassifier(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init yolo11 cls: %v\n", err)
		return
	}
	preds, err := clf.Classify(ctx, img, 5)
	if err != nil {
		fmt.Fprintf(os.Stderr, "yolo11 cls error: %v\n", err)
		return
	}
	for i, p := range preds {
		fmt.Printf("%2d. [%4d] %6.2f%%  %s\n", i+1, p.Index, p.Score*100, p.Label)
	}
}

func detectPose(ctx context.Context, s *ncnn.Session, imagePath string, img []byte) []yolo11.PoseDetection {
	header("YOLO11n-pose / person keypoints", imagePath)
	det, err := yolo11.NewPoseDetector(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init yolo11 pose: %v\n", err)
		return nil
	}
	poses, err := det.Detect(ctx, img, 0.25, 0.45)
	if err != nil {
		fmt.Fprintf(os.Stderr, "yolo11 pose error: %v\n", err)
		return nil
	}
	for i, p := range poses {
		fmt.Printf("%2d. %6.2f%%  person  [%.0f,%.0f %.0f,%.0f]  %d keypoints\n",
			i+1, p.Score*100, p.Box.X1, p.Box.Y1, p.Box.X2, p.Box.Y2, len(p.Keypoints))
	}
	if len(poses) == 0 {
		fmt.Println("(no people)")
	}
	return poses
}

func detectOBB(ctx context.Context, s *ncnn.Session, imagePath string, img []byte) []yolo11.OBBDetection {
	header("YOLO11n-obb / oriented boxes (DOTA aerial)", imagePath)
	det, err := yolo11.NewOBBDetector(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init yolo11 obb: %v\n", err)
		return nil
	}
	obbs, err := det.Detect(ctx, img, 0.25, 0.45)
	if err != nil {
		fmt.Fprintf(os.Stderr, "yolo11 obb error: %v\n", err)
		return nil
	}
	for i, o := range obbs {
		fmt.Printf("%2d. %6.2f%%  %s\n", i+1, o.Score*100, o.Label)
	}
	if len(obbs) == 0 {
		fmt.Println("(no oriented objects — expected on non-aerial photos)")
	}
	return obbs
}

func segment(ctx context.Context, s *ncnn.Session, imagePath string, img []byte) *yolo11.SegResult {
	header("YOLO11n-seg / instance segmentation", imagePath)
	seg, err := yolo11.NewSegmenter(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init yolo11 seg: %v\n", err)
		return nil
	}
	res, err := seg.Detect(ctx, img, 0.25, 0.45)
	if err != nil {
		fmt.Fprintf(os.Stderr, "yolo11 seg error: %v\n", err)
		return nil
	}
	printDetections(res.Detections)
	return res
}

func classifyGender(ctx context.Context, s *ncnn.Session, imagePath string, img []byte) {
	header("GenderDetector / largest face", imagePath)
	gd, err := genderdetector.New(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init genderdetector: %v\n", err)
		return
	}
	r, err := gd.ClassifyPhoto(ctx, img)
	if err != nil {
		fmt.Printf("(no face: %v)\n", err)
		return
	}
	fmt.Printf(" %s (%.0f%%)   [female=%.2f male=%.2f]\n", r.Label, r.Confidence*100, r.Female, r.Male)
}

// detectFaces returns the face boxes and, for each face, its 98 landmark points.
func detectFaces(ctx context.Context, s *ncnn.Session, imagePath string, img []byte) ([]detect.Detection, [][]detect.Point) {
	header("Ultra-Light / face detection + PFLD landmarks", imagePath)
	fd, err := facedetector.New(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init facedetector: %v\n", err)
		return nil, nil
	}
	faces, err := fd.Detect(ctx, img, 0.7, 0.3)
	if err != nil {
		fmt.Fprintf(os.Stderr, "facedetector error: %v\n", err)
		return nil, nil
	}
	printDetections(faces)

	lm, err := landmark.New(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init landmark: %v\n", err)
		return faces, nil
	}
	all := make([][]detect.Point, 0, len(faces))
	for i, f := range faces {
		pts, err := lm.Detect(ctx, img, f.Box)
		if err != nil {
			fmt.Fprintf(os.Stderr, "landmark error (face %d): %v\n", i+1, err)
			continue
		}
		all = append(all, pts)
		fmt.Printf("    face %d: %d landmarks\n", i+1, len(pts))
	}
	return faces, all
}

// scene bundles all model outputs for one image, for rendering.
type scene struct {
	nanodet []detect.Detection
	yolo    []detect.Detection
	faces   []detect.Detection
	marks   [][]detect.Point
	poses   []yolo11.PoseDetection
	obbs    []yolo11.OBBDetection
	seg     *yolo11.SegResult
}

// renderDebug writes annotated PNGs to ./output using the render package, one per
// model output (boxes, landmarks, pose skeletons, oriented boxes, masks).
func renderDebug(imagePath string, imgBytes []byte, sc scene) {
	img, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		fmt.Fprintf(os.Stderr, "decode image for render: %v\n", err)
		return
	}
	stem := strings.TrimSuffix(filepath.Base(imagePath), filepath.Ext(imagePath))
	st := render.AutoStyle(img)

	if len(sc.nanodet) > 0 {
		save(stem+"_objects.png", render.Render(img, render.Overlay{Detections: sc.nanodet}))
	}
	if len(sc.yolo) > 0 {
		save(stem+"_yolo.png", render.Render(img, render.Overlay{Detections: sc.yolo}))
	}
	if len(sc.faces) > 0 {
		save(stem+"_faces.png", render.Render(img, render.Overlay{Detections: sc.faces, Landmarks: sc.marks}))
	}

	// pose: boxes + skeleton + keypoint dots
	if len(sc.poses) > 0 {
		canvas := render.ToRGBA(img)
		for _, p := range sc.poses {
			render.DrawBox(canvas, p.Box, st.BoxColor, st.Thickness)
			render.DrawSkeleton(canvas, p.Keypoints, yolo11.PoseSkeleton, render.ColorSkeleton, st.Thickness)
			render.DrawLandmarks(canvas, p.Keypoints, render.ColorLandmark, st.DotRadius)
		}
		save(stem+"_pose.png", canvas)
	}

	// obb: rotated-box polygons + labels
	if len(sc.obbs) > 0 {
		canvas := render.ToRGBA(img)
		for _, o := range sc.obbs {
			render.DrawPolygon(canvas, o.Poly[:], render.ColorPoly, st.Thickness)
			render.DrawLabel(canvas, int(o.Poly[0].X), int(o.Poly[0].Y),
				fmt.Sprintf("%s %.0f%%", o.Label, o.Score*100), color.Black, render.ColorPoly, st.FontSize)
		}
		save(stem+"_obb.png", canvas)
	}

	// seg: translucent mask + boxes
	if sc.seg != nil && len(sc.seg.Detections) > 0 {
		canvas := render.ToRGBA(img)
		render.DrawMask(canvas, sc.seg.Mask, sc.seg.W, sc.seg.H, render.ColorMask)
		render.DrawDetectionsLabeled(canvas, sc.seg.Detections, st.BoxColor, st.Thickness, st.FontSize)
		save(stem+"_seg.png", canvas)
	}
}

func save(name string, img image.Image) {
	path := filepath.Join(outputDir, name)
	if err := render.SavePNG(path, img); err != nil {
		fmt.Fprintf(os.Stderr, "save %s: %v\n", path, err)
		return
	}
	fmt.Printf("[output] wrote %s\n", path)
}

func printDetections(dets []detect.Detection) {
	if len(dets) == 0 {
		fmt.Println("(no detections)")
		return
	}
	for i, d := range dets {
		fmt.Printf("%2d. %6.2f%%  %-14s  [%.0f,%.0f %.0f,%.0f]\n",
			i+1, d.Score*100, d.Label, d.Box.X1, d.Box.Y1, d.Box.X2, d.Box.Y2)
	}
}

func header(name, imagePath string) {
	fmt.Printf("\n========================================================\n")
	fmt.Printf(" %s\n image: %s\n", name, imagePath)
	fmt.Printf("========================================================\n")
}

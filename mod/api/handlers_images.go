package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/png"
	"math"
	"net/http"

	"cnnaio/mod/detect"
	"cnnaio/mod/mobilenet"
	"cnnaio/mod/nanodet"
	"cnnaio/mod/ncnn"
	"cnnaio/mod/render"
	"cnnaio/mod/yolo11"
)

// ---- JSON item types ----

type clsItem struct {
	Label string  `json:"label"`
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

type detItem struct {
	Label   string  `json:"label"`
	ClassID int     `json:"class_id"`
	Score   float64 `json:"score"`
	Box     BoxJSON `json:"box"`
}

type keypointItem struct {
	Name string `json:"name"`
	X    int    `json:"x"`
	Y    int    `json:"y"`
}

type poseItem struct {
	Score     float64        `json:"score"`
	Box       BoxJSON        `json:"box"`
	Keypoints []keypointItem `json:"keypoints"`
}

type obbItem struct {
	Label    string      `json:"label"`
	ClassID  int         `json:"class_id"`
	Score    float64     `json:"score"`
	AngleRad float64     `json:"angle_rad"`
	Polygon  []PointJSON `json:"polygon"`
}

type maskJSON struct {
	Encoding string    `json:"encoding"`
	Width    int       `json:"width"`
	Height   int       `json:"height"`
	Origin   PointJSON `json:"origin"`
	Data     string    `json:"data"`
}

type segItem struct {
	Label   string   `json:"label"`
	ClassID int      `json:"class_id"`
	Score   float64  `json:"score"`
	Box     BoxJSON  `json:"box"`
	Mask    maskJSON `json:"mask"`
}

var cocoKeypointNames = []string{
	"nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
	"right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
	"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
}

// ---- handlers ----

func (s *Server) handleClassify(w http.ResponseWriter, r *http.Request) error {
	return s.serveImages(w, r, "image.classification", s.classifyTask)
}
func (s *Server) handleDetect(w http.ResponseWriter, r *http.Request) error {
	return s.serveImages(w, r, "image.detection", s.detectTask)
}
func (s *Server) handleSegment(w http.ResponseWriter, r *http.Request) error {
	return s.serveImages(w, r, "image.segmentation", s.segmentTask)
}
func (s *Server) handlePose(w http.ResponseWriter, r *http.Request) error {
	return s.serveImages(w, r, "image.pose", s.poseTask)
}
func (s *Server) handleOriented(w http.ResponseWriter, r *http.Request) error {
	return s.serveImages(w, r, "image.oriented", s.orientedTask)
}

// ---- tasks ----

func (s *Server) classifyTask(ctx context.Context, sess *ncnn.Session, img []byte, pr *parsedRequest, _ bool) (taskResult, error) {
	model, aerr := pickModel(pr.model, "mobilenet-v2", "mobilenet-v2", "yolo11n-cls")
	if aerr != nil {
		return taskResult{}, aerr
	}
	wd, ht, err := imageDims(img)
	if err != nil {
		return taskResult{}, err
	}
	topK := pr.topK
	if topK <= 0 {
		topK = 5
	}
	var items []clsItem
	switch model {
	case "mobilenet-v2":
		clf, e := mobilenet.NewMobileNetClassifier(sess)
		if e != nil {
			return taskResult{}, errServer(e.Error())
		}
		res, e := clf.Classify(ctx, mobilenet.V2, img, topK)
		if e != nil {
			return taskResult{}, errServer(e.Error())
		}
		for _, p := range res.Predictions {
			items = append(items, clsItem{p.Label, p.Index, round4(float32(p.Score))})
		}
	case "yolo11n-cls":
		clf, e := yolo11.NewClassifier(sess)
		if e != nil {
			return taskResult{}, errServer(e.Error())
		}
		preds, e := clf.Classify(ctx, img, topK)
		if e != nil {
			return taskResult{}, errServer(e.Error())
		}
		for _, p := range preds {
			items = append(items, clsItem{p.Label, p.Index, round4(p.Score)})
		}
	}
	return taskResult{model: model, data: items, dims: Dims{wd, ht}}, nil
}

func (s *Server) detectTask(ctx context.Context, sess *ncnn.Session, img []byte, pr *parsedRequest, needRender bool) (taskResult, error) {
	model, aerr := pickModel(pr.model, "yolo11n", "yolo11n", "nanodet-plus-m")
	if aerr != nil {
		return taskResult{}, aerr
	}
	wd, ht, err := imageDims(img)
	if err != nil {
		return taskResult{}, err
	}

	var dets []detect.Detection
	switch model {
	case "yolo11n":
		d, e := yolo11.New(sess)
		if e != nil {
			return taskResult{}, errServer(e.Error())
		}
		dets, e = d.Detect(ctx, img, thr(pr.scoreThresh, 0.25), thr(pr.nmsThresh, 0.45))
		if e != nil {
			return taskResult{}, errServer(e.Error())
		}
	case "nanodet-plus-m":
		d, e := nanodet.New(sess)
		if e != nil {
			return taskResult{}, errServer(e.Error())
		}
		dets, e = d.Detect(ctx, img, thr(pr.scoreThresh, 0.4), thr(pr.nmsThresh, 0.5))
		if e != nil {
			return taskResult{}, errServer(e.Error())
		}
	}
	dets = capDetections(dets, s.effMaxResults(pr))

	items := make([]detItem, 0, len(dets))
	for _, d := range dets {
		items = append(items, detItem{d.Label, d.ClassID, round4(d.Score), box(d.Box)})
	}
	tr := taskResult{model: model, data: items, dims: Dims{wd, ht}}
	if needRender {
		if base := decodeImage(img); base != nil {
			tr.canvas = render.Render(base, render.Overlay{Detections: dets})
		}
	}
	return tr, nil
}

func (s *Server) segmentTask(ctx context.Context, sess *ncnn.Session, img []byte, pr *parsedRequest, needRender bool) (taskResult, error) {
	wd, ht, err := imageDims(img)
	if err != nil {
		return taskResult{}, err
	}
	seg, e := yolo11.NewSegmenter(sess)
	if e != nil {
		return taskResult{}, errServer(e.Error())
	}
	res, e := seg.Detect(ctx, img, thr(pr.scoreThresh, 0.25), thr(pr.nmsThresh, 0.45))
	if e != nil {
		return taskResult{}, errServer(e.Error())
	}
	insts := res.Instances
	if max := s.effMaxResults(pr); len(insts) > max {
		insts = insts[:max]
	}
	items := make([]segItem, 0, len(insts))
	for _, in := range insts {
		items = append(items, segItem{
			Label: in.Detection.Label, ClassID: in.Detection.ClassID,
			Score: round4(in.Detection.Score), Box: box(in.Detection.Box),
			Mask: maskJSON{
				Encoding: "png", Width: in.W, Height: in.H,
				Origin: PointJSON{in.OriginX, in.OriginY},
				Data:   maskPNGBase64(in.Mask, in.W, in.H),
			},
		})
	}
	tr := taskResult{model: "yolo11n-seg", data: items, dims: Dims{wd, ht}}
	if needRender {
		if base := decodeImage(img); base != nil {
			canvas := render.ToRGBA(base)
			st := render.AutoStyle(base)
			render.DrawMask(canvas, res.Mask, res.W, res.H, render.ColorMask)
			render.DrawDetectionsLabeled(canvas, res.Detections, st.BoxColor, st.Thickness, st.FontSize)
			tr.canvas = canvas
		}
	}
	return tr, nil
}

func (s *Server) poseTask(ctx context.Context, sess *ncnn.Session, img []byte, pr *parsedRequest, needRender bool) (taskResult, error) {
	wd, ht, err := imageDims(img)
	if err != nil {
		return taskResult{}, err
	}
	d, e := yolo11.NewPoseDetector(sess)
	if e != nil {
		return taskResult{}, errServer(e.Error())
	}
	poses, e := d.Detect(ctx, img, thr(pr.scoreThresh, 0.25), thr(pr.nmsThresh, 0.45))
	if e != nil {
		return taskResult{}, errServer(e.Error())
	}
	if max := s.effMaxResults(pr); len(poses) > max {
		poses = poses[:max]
	}
	items := make([]poseItem, 0, len(poses))
	for _, p := range poses {
		kps := make([]keypointItem, 0, len(p.Keypoints))
		for i, pt := range p.Keypoints {
			name := ""
			if i < len(cocoKeypointNames) {
				name = cocoKeypointNames[i]
			}
			kps = append(kps, keypointItem{name, ri(pt.X), ri(pt.Y)})
		}
		items = append(items, poseItem{round4(p.Score), box(p.Box), kps})
	}
	tr := taskResult{model: "yolo11n-pose", data: items, dims: Dims{wd, ht}}
	if needRender {
		if base := decodeImage(img); base != nil {
			canvas := render.ToRGBA(base)
			st := render.AutoStyle(base)
			for _, p := range poses {
				render.DrawBox(canvas, p.Box, st.BoxColor, st.Thickness)
				render.DrawSkeleton(canvas, p.Keypoints, yolo11.PoseSkeleton, render.ColorSkeleton, st.Thickness)
				render.DrawLandmarks(canvas, p.Keypoints, render.ColorLandmark, st.DotRadius)
			}
			tr.canvas = canvas
		}
	}
	return tr, nil
}

func (s *Server) orientedTask(ctx context.Context, sess *ncnn.Session, img []byte, pr *parsedRequest, needRender bool) (taskResult, error) {
	wd, ht, err := imageDims(img)
	if err != nil {
		return taskResult{}, err
	}
	d, e := yolo11.NewOBBDetector(sess)
	if e != nil {
		return taskResult{}, errServer(e.Error())
	}
	obbs, e := d.Detect(ctx, img, thr(pr.scoreThresh, 0.25), thr(pr.nmsThresh, 0.45))
	if e != nil {
		return taskResult{}, errServer(e.Error())
	}
	if max := s.effMaxResults(pr); len(obbs) > max {
		obbs = obbs[:max]
	}
	items := make([]obbItem, 0, len(obbs))
	for _, o := range obbs {
		poly := []PointJSON{point(o.Poly[0]), point(o.Poly[1]), point(o.Poly[2]), point(o.Poly[3])}
		angle := math.Atan2(float64(o.Poly[1].Y-o.Poly[0].Y), float64(o.Poly[1].X-o.Poly[0].X))
		items = append(items, obbItem{o.Label, o.ClassID, round4(o.Score), angle, poly})
	}
	tr := taskResult{model: "yolo11n-obb", data: items, dims: Dims{wd, ht}}
	if needRender {
		if base := decodeImage(img); base != nil {
			canvas := render.ToRGBA(base)
			st := render.AutoStyle(base)
			for _, o := range obbs {
				render.DrawPolygon(canvas, o.Poly[:], render.ColorPoly, st.Thickness)
			}
			tr.canvas = canvas
		}
	}
	return tr, nil
}

// ---- helpers ----

func capDetections(d []detect.Detection, max int) []detect.Detection {
	if max > 0 && len(d) > max {
		return d[:max]
	}
	return d
}

func decodeImage(b []byte) image.Image {
	img, _, err := image.Decode(bytes.NewReader(b))
	if err != nil {
		return nil
	}
	return img
}

// maskPNGBase64 encodes a boolean mask as an 8-bit grayscale PNG (255 = object).
func maskPNGBase64(mask []bool, w, h int) string {
	if w <= 0 || h <= 0 || len(mask) < w*h {
		return ""
	}
	g := image.NewGray(image.Rect(0, 0, w, h))
	for i, on := range mask {
		if on {
			g.Pix[i] = 255
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, g)
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

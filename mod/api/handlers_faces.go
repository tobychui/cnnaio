package api

import (
	"context"
	"encoding/json"
	"net/http"

	"cnnaio/mod/detect"
	"cnnaio/mod/facedetector"
	"cnnaio/mod/facerecognition"
	"cnnaio/mod/genderdetector"
	"cnnaio/mod/landmark"
	"cnnaio/mod/ncnn"
	"cnnaio/mod/render"
)

// ---- JSON item types ----

type faceItem struct {
	Score float64 `json:"score"`
	Box   BoxJSON `json:"box"`
}

type landmarkItem struct {
	Box    BoxJSON     `json:"box"`
	Points []PointJSON `json:"points"`
}

type embeddingItem struct {
	Box       BoxJSON   `json:"box"`
	Embedding []float32 `json:"embedding"`
	Dim       int       `json:"dim"`
}

type genderItem struct {
	Box    BoxJSON `json:"box"`
	Gender struct {
		Label      string             `json:"label"`
		Confidence float64            `json:"confidence"`
		Scores     map[string]float64 `json:"scores"`
	} `json:"gender"`
}

// ---- handlers ----

func (s *Server) handleFaceDetect(w http.ResponseWriter, r *http.Request) error {
	return s.serveImages(w, r, "face.detection", s.faceDetectTask)
}
func (s *Server) handleLandmarks(w http.ResponseWriter, r *http.Request) error {
	return s.serveImages(w, r, "face.landmarks", s.landmarkTask)
}
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) error {
	return s.serveImages(w, r, "face.embedding", s.embeddingTask)
}
func (s *Server) handleGender(w http.ResponseWriter, r *http.Request) error {
	return s.serveImages(w, r, "face.gender", s.genderTask)
}

// ---- tasks ----

func (s *Server) faceDetectTask(ctx context.Context, sess *ncnn.Session, img []byte, pr *parsedRequest, needRender bool) (taskResult, error) {
	model, aerr := pickModel(pr.model, "ultraface-rfb-320", "ultraface-rfb-320", "ultraface-slim-320")
	if aerr != nil {
		return taskResult{}, aerr
	}
	wd, ht, err := imageDims(img)
	if err != nil {
		return taskResult{}, err
	}
	variant := facedetector.RFB
	if model == "ultraface-slim-320" {
		variant = facedetector.Slim
	}
	fd, e := facedetector.NewVariant(sess, variant)
	if e != nil {
		return taskResult{}, errServer(e.Error())
	}
	faces, e := fd.Detect(ctx, img, thr(pr.scoreThresh, 0.7), thr(pr.nmsThresh, 0.3))
	if e != nil {
		return taskResult{}, errServer(e.Error())
	}
	faces = capDetections(faces, s.effMaxResults(pr))
	items := make([]faceItem, 0, len(faces))
	for _, f := range faces {
		items = append(items, faceItem{round4(f.Score), box(f.Box)})
	}
	tr := taskResult{model: model, data: items, dims: Dims{wd, ht}}
	if needRender {
		if base := decodeImage(img); base != nil {
			tr.canvas = render.Render(base, render.Overlay{Detections: faces})
		}
	}
	return tr, nil
}

func (s *Server) landmarkTask(ctx context.Context, sess *ncnn.Session, img []byte, pr *parsedRequest, needRender bool) (taskResult, error) {
	wd, ht, err := imageDims(img)
	if err != nil {
		return taskResult{}, err
	}
	lm, e := landmark.New(sess)
	if e != nil {
		return taskResult{}, errServer(e.Error())
	}

	// Which face boxes to run PFLD on?
	var boxes []detect.Box
	if pr.cropped {
		boxes = []detect.Box{{X1: 0, Y1: 0, X2: float32(wd), Y2: float32(ht)}}
	} else {
		fd, e := facedetector.New(sess)
		if e != nil {
			return taskResult{}, errServer(e.Error())
		}
		faces, e := fd.Detect(ctx, img, thr(pr.scoreThresh, 0.7), thr(pr.nmsThresh, 0.3))
		if e != nil {
			return taskResult{}, errServer(e.Error())
		}
		faces = capDetections(faces, s.effMaxResults(pr))
		for _, f := range faces {
			boxes = append(boxes, f.Box)
		}
	}

	items := make([]landmarkItem, 0, len(boxes))
	var allPts [][]detect.Point
	for _, b := range boxes {
		pts, e := lm.Detect(ctx, img, b)
		if e != nil {
			continue
		}
		pjs := make([]PointJSON, 0, len(pts))
		for _, p := range pts {
			pjs = append(pjs, point(p))
		}
		items = append(items, landmarkItem{box(b), pjs})
		allPts = append(allPts, pts)
	}
	tr := taskResult{model: "pfld", data: items, dims: Dims{wd, ht}}
	if needRender {
		if base := decodeImage(img); base != nil {
			dets := make([]detect.Detection, len(boxes))
			for i, b := range boxes {
				dets[i] = detect.Detection{Box: b, Label: "face"}
			}
			tr.canvas = render.Render(base, render.Overlay{Detections: dets, Landmarks: allPts})
		}
	}
	return tr, nil
}

func (s *Server) embeddingTask(ctx context.Context, sess *ncnn.Session, img []byte, pr *parsedRequest, needRender bool) (taskResult, error) {
	wd, ht, err := imageDims(img)
	if err != nil {
		return taskResult{}, err
	}
	m, e := facerecognition.New(sess)
	if e != nil {
		return taskResult{}, errServer(e.Error())
	}
	emb, fbox, e := m.Embed(ctx, facerecognition.Sample{Image: img, Cropped: pr.cropped})
	if e != nil {
		return taskResult{}, errUnprocessable(e.Error())
	}
	items := []embeddingItem{{Box: box(fbox), Embedding: emb, Dim: len(emb)}}
	tr := taskResult{model: "mbv2facenet", data: items, dims: Dims{wd, ht}}
	if needRender && !pr.cropped {
		if base := decodeImage(img); base != nil {
			tr.canvas = render.Render(base, render.Overlay{Detections: []detect.Detection{{Box: fbox, Label: "face"}}})
		}
	}
	return tr, nil
}

func (s *Server) genderTask(ctx context.Context, sess *ncnn.Session, img []byte, pr *parsedRequest, needRender bool) (taskResult, error) {
	wd, ht, err := imageDims(img)
	if err != nil {
		return taskResult{}, err
	}
	gd, e := genderdetector.New(sess)
	if e != nil {
		return taskResult{}, errServer(e.Error())
	}
	res, e := gd.Classify(ctx, genderdetector.Sample{Image: img, Cropped: pr.cropped})
	if e != nil {
		return taskResult{}, errUnprocessable(e.Error())
	}
	var it genderItem
	it.Box = box(res.Box)
	it.Gender.Label = res.Label
	it.Gender.Confidence = round4(res.Confidence)
	it.Gender.Scores = map[string]float64{"female": round4(res.Female), "male": round4(res.Male)}
	tr := taskResult{model: "gender-mbv2-0.35", data: []genderItem{it}, dims: Dims{wd, ht}}
	if needRender && !pr.cropped {
		if base := decodeImage(img); base != nil {
			tr.canvas = render.Render(base, render.Overlay{
				Detections: []detect.Detection{{Box: res.Box, Label: res.Label, Score: res.Confidence}},
			})
		}
	}
	return tr, nil
}

// ---- face comparison (two images; not the single-image pipeline) ----

type comparisonReq struct {
	Model     string   `json:"model"`
	ImageA    string   `json:"image_a"`
	ImageB    string   `json:"image_b"`
	ACropped  bool     `json:"a_cropped"`
	BCropped  bool     `json:"b_cropped"`
	Threshold *float32 `json:"threshold"`
}

func (s *Server) handleComparison(w http.ResponseWriter, r *http.Request) error {
	body, err := readLimited(r.Body, s.cfg.MaxImageBytes)
	if err != nil {
		return err
	}
	var req comparisonReq
	if e := json.Unmarshal(body, &req); e != nil {
		return errBadRequest("invalid JSON body: "+e.Error(), "", "")
	}
	a, e := decodeBase64Image(req.ImageA)
	if e != nil {
		return errBadRequest("image_a: "+e.Error(), "image_a", "")
	}
	b, e := decodeBase64Image(req.ImageB)
	if e != nil {
		return errBadRequest("image_b: "+e.Error(), "image_b", "")
	}
	threshold := thr(req.Threshold, facerecognition.DefaultThreshold)

	ctx, cancel := s.reqContext(r)
	defer cancel()
	var res facerecognition.Result
	if err := s.withSession(ctx, func(sess *ncnn.Session) error {
		m, e := facerecognition.New(sess)
		if e != nil {
			return errServer(e.Error())
		}
		var rerr error
		res, rerr = m.Match(ctx,
			facerecognition.Sample{Image: a, Cropped: req.ACropped},
			facerecognition.Sample{Image: b, Cropped: req.BCropped},
			threshold)
		if rerr != nil {
			return errUnprocessable(rerr.Error())
		}
		return nil
	}); err != nil {
		return err
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"object":     "face.comparison",
		"model":      "mbv2facenet",
		"created":    now(),
		"similarity": round4(res.Similarity),
		"same":       res.Same,
		"threshold":  round4(threshold),
		"box_a":      box(res.BoxA),
		"box_b":      box(res.BoxB),
	})
	return nil
}

package api

import (
	"context"
	"encoding/json"
	"net/http"

	"cnnaio/mod/ncnn"
)

type analyzeReq struct {
	Image   string                     `json:"image"`
	Tasks   []string                   `json:"tasks"`
	Options map[string]json.RawMessage `json:"options"`
	Render  bool                       `json:"render"`
	Async   bool                       `json:"async"`
}

type analyzeTask struct {
	object string
	fn     task
}

func (s *Server) analyzeTasks() map[string]analyzeTask {
	return map[string]analyzeTask{
		"classify":   {"image.classification", s.classifyTask},
		"detect":     {"image.detection", s.detectTask},
		"segment":    {"image.segmentation", s.segmentTask},
		"pose":       {"image.pose", s.poseTask},
		"oriented":   {"image.oriented", s.orientedTask},
		"faces":      {"face.detection", s.faceDetectTask},
		"landmarks":  {"face.landmarks", s.landmarkTask},
		"gender":     {"face.gender", s.genderTask},
		"attributes": {"face.gender", s.genderTask}, // alias for gender
	}
}

// handleAnalyze runs several tasks over one image in a single request.
func (s *Server) handleAnalyze(w http.ResponseWriter, r *http.Request) error {
	body, err := readLimited(r.Body, s.cfg.MaxImageBytes)
	if err != nil {
		return err
	}
	var req analyzeReq
	if e := json.Unmarshal(body, &req); e != nil {
		return errBadRequest("invalid JSON body: "+e.Error(), "", "")
	}
	img, e := decodeBase64Image(req.Image)
	if e != nil {
		return errBadRequest("image: "+e.Error(), "image", "")
	}
	if len(req.Tasks) == 0 {
		return errBadRequest("no tasks requested", "tasks", "")
	}
	wd, ht, e := imageDims(img)
	if e != nil {
		return e
	}

	available := s.analyzeTasks()
	// Validate task names up front.
	for _, name := range req.Tasks {
		if _, ok := available[name]; !ok {
			return errBadRequest("unknown task: "+name, "tasks", "")
		}
	}

	compute := func(ctx context.Context) (any, error) {
		results := make(map[string]any, len(req.Tasks))
		for _, name := range req.Tasks {
			spec := available[name]
			pr := prFromOptions(req.Options[name], req.Render)
			var tr taskResult
			if e := s.withSession(ctx, func(sess *ncnn.Session) error {
				var err error
				tr, err = spec.fn(ctx, sess, img, pr, req.Render)
				return err
			}); e != nil {
				return nil, e
			}
			sub := map[string]any{"object": spec.object, "model": tr.model, "data": tr.data}
			if req.Render && tr.canvas != nil {
				sub["rendered_image"] = pngDataURI(tr.canvas)
			}
			results[name] = sub
		}
		return map[string]any{
			"object": "vision.analysis", "created": now(),
			"image": Dims{wd, ht}, "results": results,
		}, nil
	}

	if req.Async {
		job := s.jobs.Submit(compute)
		writeJSON(w, http.StatusAccepted, job)
		return nil
	}
	ctx, cancel := s.reqContext(r)
	defer cancel()
	res, err := compute(ctx)
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, res)
	return nil
}

// prFromOptions builds a parsedRequest carrying only parameters (no image) from
// a per-task options object.
func prFromOptions(raw json.RawMessage, render bool) *parsedRequest {
	var jr jsonRequest
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &jr)
	}
	return &parsedRequest{
		model: jr.Model, scoreThresh: jr.ScoreThresh, nmsThresh: jr.NMSThresh,
		topK: jr.TopK, maxResults: jr.MaxResults, cropped: jr.Cropped, render: render,
	}
}

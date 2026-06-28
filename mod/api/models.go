package api

import "fmt"

// ModelInfo is one row of GET /v1/models.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "model"
	Task    string `json:"task"`
	Classes int    `json:"classes,omitempty"`
	Input   int    `json:"input,omitempty"`
}

// registry is the full set of available models.
var registry = []ModelInfo{
	{"mobilenet-v2", "model", "classification", 1000, 224},
	{"yolo11n-cls", "model", "classification", 1000, 224},
	{"yolo11n", "model", "detection", 80, 640},
	{"nanodet-plus-m", "model", "detection", 80, 416},
	{"yolo11n-seg", "model", "segmentation", 80, 640},
	{"yolo11n-pose", "model", "pose", 1, 640},
	{"yolo11n-obb", "model", "oriented", 15, 640},
	{"ultraface-rfb-320", "model", "face_detection", 1, 320},
	{"ultraface-slim-320", "model", "face_detection", 1, 320},
	{"pfld", "model", "landmarks", 0, 112},
	{"mbv2facenet", "model", "face_embedding", 0, 112},
	{"gender-mbv2-0.35", "model", "gender", 2, 64},
}

func findModel(id string) (ModelInfo, bool) {
	for _, m := range registry {
		if m.ID == id {
			return m, true
		}
	}
	return ModelInfo{}, false
}

// pickModel resolves a requested model id against the allowed set for a task,
// falling back to def when the request is empty.
func pickModel(requested, def string, allowed ...string) (string, *apiError) {
	if requested == "" {
		return def, nil
	}
	for _, a := range allowed {
		if requested == a {
			return requested, nil
		}
	}
	return "", errNotFound(
		fmt.Sprintf("model %q is not available for this endpoint (allowed: %v)", requested, allowed),
		"model_not_found")
}

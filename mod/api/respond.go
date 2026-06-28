package api

import (
	"encoding/json"
	"math"
	"net/http"
	"time"

	"cnnaio/mod/detect"
)

// Dims is the decoded source image size; all coordinates are in this space.
type Dims struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// Envelope is the standard OpenAI-style response wrapper for single-image tasks.
type Envelope struct {
	Object        string `json:"object"`
	Model         string `json:"model,omitempty"`
	Created       int64  `json:"created"`
	Image         *Dims  `json:"image,omitempty"`
	TimingMs      int64  `json:"timing_ms"`
	Data          any    `json:"data,omitempty"`
	RenderedImage string `json:"rendered_image,omitempty"`
}

// BoxJSON is a bounding box in absolute pixels.
type BoxJSON struct {
	X1 int `json:"x1"`
	Y1 int `json:"y1"`
	X2 int `json:"x2"`
	Y2 int `json:"y2"`
}

// PointJSON is a 2D point in absolute pixels.
type PointJSON struct {
	X int `json:"x"`
	Y int `json:"y"`
}

func box(b detect.Box) BoxJSON {
	return BoxJSON{ri(b.X1), ri(b.Y1), ri(b.X2), ri(b.Y2)}
}
func point(p detect.Point) PointJSON { return PointJSON{ri(p.X), ri(p.Y)} }

func ri(f float32) int         { return int(math.Round(float64(f))) }
func round4(f float32) float64 { return math.Round(float64(f)*1e4) / 1e4 }

// apiError carries an HTTP status plus the OpenAI-style error fields.
type apiError struct {
	Status  int    `json:"-"`
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

func (e *apiError) Error() string { return e.Message }

func errBadRequest(msg, param, code string) *apiError {
	return &apiError{http.StatusBadRequest, msg, "invalid_request_error", param, code}
}
func errAuth(msg string) *apiError {
	return &apiError{http.StatusUnauthorized, msg, "authentication_error", "", ""}
}
func errNotFound(msg, code string) *apiError {
	return &apiError{http.StatusNotFound, msg, "not_found_error", "", code}
}
func errTooLarge(msg string) *apiError {
	return &apiError{http.StatusRequestEntityTooLarge, msg, "payload_too_large", "", ""}
}
func errUnprocessable(msg string) *apiError {
	return &apiError{http.StatusUnprocessableEntity, msg, "unprocessable_entity", "", ""}
}
func errServer(msg string) *apiError {
	return &apiError{http.StatusInternalServerError, msg, "server_error", "", ""}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, err error) {
	ae, ok := err.(*apiError)
	if !ok {
		ae = errServer(err.Error())
	}
	writeJSON(w, ae.Status, map[string]any{"error": ae})
}

func now() int64 { return time.Now().Unix() }

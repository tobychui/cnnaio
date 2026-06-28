package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"strconv"
	"strings"
)

// parsedRequest holds the decoded inputs + common parameters for a recognition
// request (either a JSON body or a multipart upload).
type parsedRequest struct {
	model       string
	images      [][]byte // 1+ decoded images (>1 = batch)
	scoreThresh *float32
	nmsThresh   *float32
	topK        int
	maxResults  int
	render      bool
	cropped     bool
	async       bool
}

// jsonRequest mirrors the JSON body fields shared across endpoints.
type jsonRequest struct {
	Model       string   `json:"model"`
	Image       string   `json:"image"`
	Images      []string `json:"images"`
	ScoreThresh *float32 `json:"score_threshold"`
	NMSThresh   *float32 `json:"nms_threshold"`
	TopK        int      `json:"top_k"`
	MaxResults  int      `json:"max_results"`
	Render      bool     `json:"render"`
	Cropped     bool     `json:"cropped"`
	Async       bool     `json:"async"`
}

// parseRequest reads a recognition request from either application/json (image as
// base64/data-URI) or multipart/form-data (image file parts). maxBytes caps the
// total payload.
func parseRequest(r *http.Request, maxBytes int64) (*parsedRequest, error) {
	ct := r.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "multipart/form-data") {
		return parseMultipart(r, maxBytes)
	}
	return parseJSON(r, maxBytes)
}

func parseJSON(r *http.Request, maxBytes int64) (*parsedRequest, error) {
	body, err := readLimited(r.Body, maxBytes)
	if err != nil {
		return nil, err
	}
	var jr jsonRequest
	if len(body) > 0 {
		if err := json.Unmarshal(body, &jr); err != nil {
			return nil, errBadRequest("invalid JSON body: "+err.Error(), "", "")
		}
	}
	pr := &parsedRequest{
		model: jr.Model, scoreThresh: jr.ScoreThresh, nmsThresh: jr.NMSThresh,
		topK: jr.TopK, maxResults: jr.MaxResults, render: jr.Render,
		cropped: jr.Cropped, async: jr.Async,
	}
	srcs := jr.Images
	if len(srcs) == 0 && jr.Image != "" {
		srcs = []string{jr.Image}
	}
	for i, s := range srcs {
		b, err := decodeBase64Image(s)
		if err != nil {
			return nil, errBadRequest("image["+strconv.Itoa(i)+"]: "+err.Error(), "image", "")
		}
		pr.images = append(pr.images, b)
	}
	return pr, nil
}

func parseMultipart(r *http.Request, maxBytes int64) (*parsedRequest, error) {
	if err := r.ParseMultipartForm(maxBytes); err != nil {
		return nil, errTooLarge("multipart payload too large or malformed")
	}
	pr := &parsedRequest{
		model:      r.FormValue("model"),
		topK:       atoiDefault(r.FormValue("top_k"), 0),
		maxResults: atoiDefault(r.FormValue("max_results"), 0),
		render:     truthy(r.FormValue("render")),
		cropped:    truthy(r.FormValue("cropped")),
		async:      truthy(r.FormValue("async")),
	}
	if v := r.FormValue("score_threshold"); v != "" {
		if f, err := strconv.ParseFloat(v, 32); err == nil {
			ff := float32(f)
			pr.scoreThresh = &ff
		}
	}
	if v := r.FormValue("nms_threshold"); v != "" {
		if f, err := strconv.ParseFloat(v, 32); err == nil {
			ff := float32(f)
			pr.nmsThresh = &ff
		}
	}
	if r.MultipartForm != nil {
		for _, fh := range r.MultipartForm.File["image"] {
			f, err := fh.Open()
			if err != nil {
				return nil, errBadRequest("cannot read uploaded file", "image", "")
			}
			b, err := io.ReadAll(io.LimitReader(f, maxBytes))
			f.Close()
			if err != nil {
				return nil, errBadRequest("cannot read uploaded file", "image", "")
			}
			pr.images = append(pr.images, b)
		}
	}
	return pr, nil
}

// decodeBase64Image accepts a data URI ("data:image/...;base64,XXXX") or a bare
// base64 string and returns the raw image bytes. Remote URLs are not supported.
func decodeBase64Image(s string) ([]byte, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, errBadRequest("empty image", "image", "")
	}
	if strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://") {
		return nil, errBadRequest("remote image URLs are not supported; send base64 or multipart", "image", "")
	}
	if i := strings.Index(s, ";base64,"); i >= 0 {
		s = s[i+len(";base64,"):]
	} else if strings.HasPrefix(s, "data:") {
		return nil, errBadRequest("unsupported data URI (expected ;base64,)", "image", "")
	}
	for _, enc := range []*base64.Encoding{base64.StdEncoding, base64.RawStdEncoding, base64.URLEncoding, base64.RawURLEncoding} {
		if b, err := enc.DecodeString(s); err == nil {
			return b, nil
		}
	}
	return nil, errBadRequest("image is not valid base64", "image", "")
}

// imageDims decodes just the header to get pixel dimensions.
func imageDims(b []byte) (int, int, error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(b))
	if err != nil {
		return 0, 0, errBadRequest("cannot decode image: "+err.Error(), "image", "")
	}
	return cfg.Width, cfg.Height, nil
}

// wantsPreview reports whether the client asked for a raw PNG instead of JSON,
// via ?preview / ?preview=true or an Accept: image/png header.
func wantsPreview(r *http.Request) bool {
	q := r.URL.Query()
	if q.Has("preview") {
		v := q.Get("preview")
		return v == "" || truthy(v)
	}
	return strings.Contains(r.Header.Get("Accept"), "image/png")
}

func readLimited(rc io.ReadCloser, maxBytes int64) ([]byte, error) {
	defer rc.Close()
	b, err := io.ReadAll(io.LimitReader(rc, maxBytes+1))
	if err != nil {
		return nil, errBadRequest("cannot read request body", "", "")
	}
	if int64(len(b)) > maxBytes {
		return nil, errTooLarge("request body exceeds max_image_bytes")
	}
	return b, nil
}

func truthy(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func atoiDefault(s string, def int) int {
	if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
		return n
	}
	return def
}

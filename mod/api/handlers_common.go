package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/png"
	"net/http"
	"time"

	"cnnaio/mod/ncnn"
)

// taskResult is what a per-image task produces.
type taskResult struct {
	model  string
	data   any         // the data[] payload (a slice)
	dims   Dims        // source image dimensions
	canvas *image.RGBA // annotated image, or nil when render not requested
}

// task runs one image on a session. When needRender is true it must also build
// the annotated canvas.
type task func(ctx context.Context, sess *ncnn.Session, img []byte, pr *parsedRequest, needRender bool) (taskResult, error)

// serveImages is the generic pipeline for single-image tasks: it handles batch
// input, async jobs, JSON vs raw-PNG preview, and optional rendering.
func (s *Server) serveImages(w http.ResponseWriter, r *http.Request, object string, t task) error {
	pr, err := parseRequest(r, s.cfg.MaxImageBytes)
	if err != nil {
		return err
	}
	if len(pr.images) == 0 {
		return errBadRequest("no image provided", "image", "")
	}
	preview := wantsPreview(r) && !pr.async && len(pr.images) == 1
	needRender := pr.render || preview

	compute := func(ctx context.Context) (any, error) {
		envs := make([]*Envelope, 0, len(pr.images))
		for _, img := range pr.images {
			var tr taskResult
			start := time.Now()
			e := s.withSession(ctx, func(sess *ncnn.Session) error {
				var err error
				tr, err = t(ctx, sess, img, pr, needRender)
				return err
			})
			if e != nil {
				return nil, e
			}
			env := &Envelope{
				Object: object, Model: tr.model, Created: now(),
				Image: &tr.dims, TimingMs: time.Since(start).Milliseconds(), Data: tr.data,
			}
			if pr.render && tr.canvas != nil {
				env.RenderedImage = pngDataURI(tr.canvas)
			}
			envs = append(envs, env)
		}
		if len(envs) == 1 {
			return envs[0], nil
		}
		return map[string]any{"object": object + ".batch", "created": now(), "data": envs}, nil
	}

	// Async: enqueue and return 202 + the job handle.
	if pr.async {
		job := s.jobs.Submit(compute)
		writeJSON(w, http.StatusAccepted, job)
		return nil
	}

	// Preview: return the annotated PNG directly (single image only).
	if preview {
		ctx, cancel := s.reqContext(r)
		defer cancel()
		var tr taskResult
		err := s.withSession(ctx, func(sess *ncnn.Session) error {
			var e error
			tr, e = t(ctx, sess, pr.images[0], pr, true)
			return e
		})
		if err != nil {
			return err
		}
		if tr.canvas == nil {
			return errUnprocessable("nothing to render for preview")
		}
		writePNG(w, tr.canvas)
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

// effMaxResults clamps the per-request cap to the server cap.
func (s *Server) effMaxResults(pr *parsedRequest) int {
	m := s.cfg.MaxResults
	if pr.maxResults > 0 && pr.maxResults < m {
		m = pr.maxResults
	}
	if m <= 0 {
		m = 100
	}
	return m
}

func thr(p *float32, def float32) float32 {
	if p != nil {
		return *p
	}
	return def
}

func pngDataURI(img image.Image) string {
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

func writePNG(w http.ResponseWriter, img image.Image) {
	w.Header().Set("Content-Type", "image/png")
	w.WriteHeader(http.StatusOK)
	_ = png.Encode(w, img)
}

// decodeForRender decodes image bytes into an RGBA canvas for annotation.
func decodeForRender(b []byte) *image.RGBA {
	img, _, err := image.Decode(bytes.NewReader(b))
	if err != nil {
		return nil
	}
	if rgba, ok := img.(*image.RGBA); ok {
		return rgba
	}
	dst := image.NewRGBA(img.Bounds())
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			dst.Set(x, y, img.At(x, y))
		}
	}
	return dst
}

package ncnn

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/png"
	"io"
	"io/fs"
	"strings"
	"testing/fstest"

	_ "golang.org/x/image/webp" // pure-Go WebP decoder: registers it for image.Decode
)

// ImageGuestPath is where ImageFS should be mounted; the guest reads the input
// image from ImageGuestPath + "/image".
const ImageGuestPath = "/input"

// ImageFS wraps raw image bytes as a tiny read-only in-memory filesystem to hand
// to the wasm sandbox (mounted at ImageGuestPath). The guest path of the image
// is then "/input/image".
//
// The wasm decodes images with stb_image, which does NOT understand WebP, so any
// WebP input is transcoded to PNG here first (pure Go, no ffmpeg). Other formats
// (JPEG/PNG/BMP/…) are passed through untouched.
func ImageFS(data []byte) fs.FS {
	if isWebP(data) {
		if png, err := webpToPNG(data); err == nil {
			data = png
		}
		// On transcode failure, fall through with the original bytes; the wasm
		// will surface a decode error.
	}
	return fstest.MapFS{"image": &fstest.MapFile{Data: data}}
}

// isWebP reports whether b is a RIFF/WEBP container.
func isWebP(b []byte) bool {
	return len(b) >= 12 && string(b[0:4]) == "RIFF" && string(b[8:12]) == "WEBP"
}

// webpToPNG decodes WebP bytes (golang.org/x/image/webp) and re-encodes as PNG.
func webpToPNG(b []byte) ([]byte, error) {
	img, _, err := image.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Tensor is one named output blob returned by the wasm "infer" command.
// Data is laid out channel-major: index = ci*(H*W) + y*W + x.
type Tensor struct {
	Name    string
	W, H, C int
	Data    []float32
}

// At returns Data[ci*H*W + y*W + x] with no bounds checking beyond the slice.
func (t Tensor) At(ci, y, x int) float32 {
	return t.Data[(ci*t.H+y)*t.W+x]
}

// InferOutput is the decoded result of an "infer" run: the named tensors plus
// the original (pre-resize) image dimensions, which detectors need to map boxes
// back to pixel coordinates.
type InferOutput struct {
	OrigW, OrigH int
	Tensors      []Tensor
}

// Tensor looks up an output tensor by blob name.
func (o *InferOutput) Tensor(name string) (Tensor, bool) {
	for _, t := range o.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return Tensor{}, false
}

// ParseInferOutput decodes the binary stdout produced by the wasm "infer"
// command (see cmd_infer in build/classify.c):
//
//	INFER <n> <origW> <origH>\n
//	T <name> <w> <h> <c>\n   (×n)
//	<raw float32 LE payload: each tensor's c*h*w values, in header order>
func ParseInferOutput(stdout string) (*InferOutput, error) {
	r := bufio.NewReader(strings.NewReader(stdout))

	var n, origW, origH int
	header, err := r.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("read INFER header: %w", err)
	}
	if _, err := fmt.Sscanf(header, "INFER %d %d %d", &n, &origW, &origH); err != nil {
		return nil, fmt.Errorf("parse INFER header %q: %w", strings.TrimSpace(header), err)
	}

	out := &InferOutput{OrigW: origW, OrigH: origH, Tensors: make([]Tensor, n)}
	for i := 0; i < n; i++ {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("read tensor header %d: %w", i, err)
		}
		var name string
		var w, h, c int
		if _, err := fmt.Sscanf(line, "T %s %d %d %d", &name, &w, &h, &c); err != nil {
			return nil, fmt.Errorf("parse tensor header %q: %w", strings.TrimSpace(line), err)
		}
		out.Tensors[i] = Tensor{Name: name, W: w, H: h, C: c}
	}

	// Binary payload follows immediately, in header order.
	for i := range out.Tensors {
		t := &out.Tensors[i]
		count := t.W * max(t.H, 1) * max(t.C, 1)
		t.Data = make([]float32, count)
		if err := binary.Read(r, binary.LittleEndian, t.Data); err != nil {
			return nil, fmt.Errorf("read tensor %q payload (%d floats): %w", t.Name, count, err)
		}
	}

	// Sanity: there should be nothing left but EOF.
	if _, err := r.ReadByte(); err != io.EOF {
		return out, nil // tolerate trailing bytes; not fatal
	}
	return out, nil
}

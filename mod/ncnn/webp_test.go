package ncnn

import (
	"bytes"
	"image"
	"os"
	"path/filepath"
	"testing"
)

// TestImageFSTranscodesWebP verifies that WebP input is detected and transcoded
// to PNG (so the stb-based wasm can decode it), preserving dimensions.
func TestImageFSTranscodesWebP(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "tests", "sample.webp"))
	if os.IsNotExist(err) {
		t.Skip("tests/sample.webp not present; skipping")
	}
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !isWebP(b) {
		t.Fatalf("sample.webp not detected as WebP")
	}

	// Go-side decode (golang.org/x/image/webp must be registered).
	wcfg, format, err := image.DecodeConfig(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("DecodeConfig webp: %v", err)
	}
	if format != "webp" {
		t.Fatalf("format = %q, want webp", format)
	}

	// Transcode to PNG and confirm it decodes as PNG with matching dimensions.
	pngBytes, err := webpToPNG(b)
	if err != nil {
		t.Fatalf("webpToPNG: %v", err)
	}
	img, format, err := image.Decode(bytes.NewReader(pngBytes))
	if err != nil {
		t.Fatalf("decode transcoded: %v", err)
	}
	if format != "png" {
		t.Errorf("transcoded format = %q, want png", format)
	}
	if img.Bounds().Dx() != wcfg.Width || img.Bounds().Dy() != wcfg.Height {
		t.Errorf("dims %dx%d, want %dx%d", img.Bounds().Dx(), img.Bounds().Dy(), wcfg.Width, wcfg.Height)
	}
}

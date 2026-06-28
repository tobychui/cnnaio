package facerecognition

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"cnnaio/mod/ncnn"
)

// TestMatch validates that MBV2FaceNet (a) keeps different people apart and
// (b) matches two different photos of the same person. Ground truth from
// tests/face_reg_notes.txt: 4=person A, 6/7/8=person B, 9/10=person C.
func TestMatch(t *testing.T) {
	ctx := context.Background()
	s, err := ncnn.NewNcnnSession()
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	defer s.Close(ctx)

	m, err := New(s)
	if err != nil {
		t.Fatalf("new matcher: %v", err)
	}

	read := func(p string) []byte {
		b, err := os.ReadFile(filepath.Join("..", "..", p))
		if os.IsNotExist(err) {
			t.Skipf("sample image %s not present; skipping", p)
		}
		if err != nil {
			t.Fatalf("read %s: %v", p, err)
		}
		return b
	}
	imgA := read("tests/4.jpg")  // person A
	imgB := read("tests/6.jpg")  // person B
	imgC1 := read("tests/7.png") // person C
	imgC2 := read("tests/8.png") // person C

	// Different people -> should not match.
	diff, err := m.Match(ctx, Sample{Image: imgA}, Sample{Image: imgB}, DefaultThreshold)
	if err != nil {
		t.Fatalf("match A vs B: %v", err)
	}
	if diff.Same {
		t.Errorf("4 vs 6 are different people but matched (sim=%.3f)", diff.Similarity)
	}

	// Same person, two different photos -> should match.
	same, err := m.Match(ctx, Sample{Image: imgC1}, Sample{Image: imgC2}, DefaultThreshold)
	if err != nil {
		t.Fatalf("match C vs C: %v", err)
	}
	if !same.Same {
		t.Errorf("9 vs 10 are the same person but did not match (sim=%.3f)", same.Similarity)
	}

	// And same-person similarity must clearly exceed different-person similarity.
	if same.Similarity <= diff.Similarity {
		t.Errorf("expected same(9,10)=%.3f > different(4,6)=%.3f", same.Similarity, diff.Similarity)
	}
	t.Logf("different(4,6)=%.3f  same(9,10)=%.3f", diff.Similarity, same.Similarity)
}

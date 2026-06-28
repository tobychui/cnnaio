package facerecognition

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"cnnaio/mod/ncnn"
)

// TestMeasure prints the pairwise similarity matrix across the labelled test
// images so identity separation can be eyeballed. Run with: go test -run TestMeasure -v
// Ground truth (tests/face_reg_notes.txt): A=4 ; B=6,7,8 ; C=9,10.
func TestMeasure(t *testing.T) {
	ctx := context.Background()
	s, err := ncnn.NewNcnnSession()
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	defer s.Close(ctx)
	m, err := New(s)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	names := []string{"4.jpg", "6.jpg", "7.png", "8.png", "9.jpg", "10.jpg"}
	embs := make(map[string]Embedding, len(names))
	for _, n := range names {
		b, err := os.ReadFile(filepath.Join("..", "..", "tests", n))
		if err != nil {
			t.Skipf("missing %s: %v", n, err)
		}
		e, _, err := m.EmbedPhoto(ctx, b)
		if err != nil {
			t.Fatalf("embed %s: %v", n, err)
		}
		embs[n] = e
	}

	header := "         "
	for _, n := range names {
		header += fmt.Sprintf("%-8s", n)
	}
	t.Log(header)
	for _, a := range names {
		row := fmt.Sprintf("%-8s ", a)
		for _, b := range names {
			row += fmt.Sprintf("%-8.2f", Similarity(embs[a], embs[b]))
		}
		t.Log(row)
	}
}

package genderdetector

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"cnnaio/mod/ncnn"
)

// TestClassify checks that clearly-female and clearly-male faces are labelled
// correctly with the default (ImageNet) preprocessing.
func TestClassify(t *testing.T) {
	ctx := context.Background()
	s, err := ncnn.NewNcnnSession()
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	defer s.Close(ctx)
	d, err := New(s)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	cases := []struct {
		file string
		want string
	}{
		{"tests/4.jpg", "female"},
		{"tests/7.png", "female"},
		{"tests/male1.jpg", "male"},
	}
	for _, c := range cases {
		b, err := os.ReadFile(filepath.Join("..", "..", c.file))
		if os.IsNotExist(err) {
			t.Skipf("sample %s not present; skipping", c.file)
		}
		if err != nil {
			t.Fatalf("read %s: %v", c.file, err)
		}
		r, err := d.ClassifyPhoto(ctx, b)
		if err != nil {
			t.Fatalf("classify %s: %v", c.file, err)
		}
		t.Logf("%-14s -> %s (%.0f%%)  female=%.3f male=%.3f", c.file, r.Label, r.Confidence*100, r.Female, r.Male)
		if r.Label != c.want {
			t.Errorf("%s: got %s, want %s", c.file, r.Label, c.want)
		}
	}
}

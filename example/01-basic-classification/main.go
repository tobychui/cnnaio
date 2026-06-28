// Example 1 — Basic classification.
//
// The smallest useful cnnaio program: spin up one ncnn (wazero) session, load an
// image, run MobileNetV2 image classification, and print the top-5 labels to
// STDOUT. No HTTP server, no rendering, no external files — the model and the
// inference runtime are both embedded in the binary.
//
// Run it:
//
//	go run ./example/01-basic-classification                 # uses the bundled test.png
//	go run ./example/01-basic-classification path/to/img.jpg # your own image
package main

import (
	"context"
	"fmt"
	"os"

	"cnnaio/mod/mobilenet"
	"cnnaio/mod/ncnn"
)

// defaultImage is the sample shipped next to the examples. tests/ is gitignored,
// so examples default to example/testdata/test.png instead.
const defaultImage = "example/testdata/test.png"

func main() {
	imagePath := defaultImage
	if len(os.Args) > 1 {
		imagePath = os.Args[1]
	}

	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot read image %q: %v\n", imagePath, err)
		os.Exit(1)
	}

	ctx := context.Background()

	// 1. Create the shared inference session. This compiles the embedded ncnn
	//    wasm once (a few hundred ms) and owns the wazero runtime. Always Close it.
	session, err := ncnn.NewNcnnSession()
	if err != nil {
		fmt.Fprintf(os.Stderr, "init ncnn session: %v\n", err)
		os.Exit(1)
	}
	defer session.Close(ctx)

	// 2. Attach a MobileNet classifier to the session.
	clf, err := mobilenet.NewMobileNetClassifier(session)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init classifier: %v\n", err)
		os.Exit(1)
	}

	// 3. Classify — top 5 ImageNet-1000 labels for the image.
	res, err := clf.Classify(ctx, mobilenet.V2, imageBytes, 5)
	if err != nil {
		fmt.Fprintf(os.Stderr, "classify: %v\n", err)
		os.Exit(1)
	}

	// 4. Print results to STDOUT.
	fmt.Printf("Classification of %s  (%s)\n", imagePath, res.Variant)
	for i, p := range res.Predictions {
		fmt.Printf("%2d. %6.2f%%  [%d] %s\n", i+1, p.Score*100, p.Index, p.Label)
	}
	fmt.Printf("\ninference round-trip: %s\n", res.Duration.Round(1e6))
}

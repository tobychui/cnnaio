// Package render visualizes model outputs — detection boxes, labels, landmarks /
// keypoints, oriented (rotated) boxes, and segmentation masks — by drawing onto
// an image and saving a PNG. It is a debugging/visualization helper layered on
// top of the data types in mod/detect; model packages don't depend on it.
package render

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"

	"cnnaio/mod/detect"
)

// Palette for overlays.
var (
	ColorBox      = color.RGBA{0, 220, 0, 255}   // green boxes
	ColorLandmark = color.RGBA{255, 40, 40, 255} // red landmark/keypoint dots
	ColorPoly     = color.RGBA{255, 200, 0, 255} // amber oriented-box polygons
	ColorSkeleton = color.RGBA{0, 180, 255, 255} // cyan pose skeleton
	ColorMask     = color.RGBA{255, 0, 200, 110} // translucent magenta mask
)

// Style holds size/colour parameters scaled to an image so overlays stay legible.
type Style struct {
	BoxColor      color.Color
	LandmarkColor color.Color
	Thickness     int
	DotRadius     int
	FontSize      float64
}

// AutoStyle derives legible sizes from the image height.
func AutoStyle(img image.Image) Style {
	dim := img.Bounds().Dy()
	return Style{
		BoxColor:      ColorBox,
		LandmarkColor: ColorLandmark,
		Thickness:     clampi(dim/400, 2, 6),
		DotRadius:     clampi(dim/300, 2, 5),
		FontSize:      clampf(float64(dim)/35, 12, 40),
	}
}

// Overlay bundles everything to draw over one image.
type Overlay struct {
	Detections []detect.Detection // labeled boxes
	Landmarks  [][]detect.Point   // dot clusters (face landmarks / pose keypoints)
}

// Render draws an overlay onto a copy of img using AutoStyle and returns the
// canvas. Convenience for the common "boxes + dots" case.
func Render(img image.Image, ov Overlay) *image.RGBA {
	st := AutoStyle(img)
	canvas := ToRGBA(img)
	DrawDetectionsLabeled(canvas, ov.Detections, st.BoxColor, st.Thickness, st.FontSize)
	for _, pts := range ov.Landmarks {
		DrawLandmarks(canvas, pts, st.LandmarkColor, st.DotRadius)
	}
	return canvas
}

// ToRGBA returns img as a mutable *image.RGBA (a copy unless it already is one).
func ToRGBA(img image.Image) *image.RGBA {
	if rgba, ok := img.(*image.RGBA); ok {
		return rgba
	}
	b := img.Bounds()
	dst := image.NewRGBA(b)
	draw.Draw(dst, b, img, b.Min, draw.Src)
	return dst
}

// DrawDetections draws each detection's box outline.
func DrawDetections(dst *image.RGBA, dets []detect.Detection, col color.Color, thickness int) {
	for _, d := range dets {
		DrawBox(dst, d.Box, col, thickness)
	}
}

// DrawDetectionsLabeled draws each detection's box plus a "label NN%" caption.
func DrawDetectionsLabeled(dst *image.RGBA, dets []detect.Detection, col color.Color, thickness int, fontSize float64) {
	for _, d := range dets {
		DrawBox(dst, d.Box, col, thickness)
		label := fmt.Sprintf("%s %.0f%%", d.Label, d.Score*100)
		DrawLabel(dst, int(d.Box.X1), int(d.Box.Y1), label, color.Black, col, fontSize)
	}
}

// DrawBox draws a rectangle outline of the given thickness.
func DrawBox(dst *image.RGBA, b detect.Box, col color.Color, thickness int) {
	if thickness < 1 {
		thickness = 1
	}
	x1, y1 := int(b.X1), int(b.Y1)
	x2, y2 := int(b.X2), int(b.Y2)
	for t := 0; t < thickness; t++ {
		hLine(dst, x1, x2, y1+t, col)
		hLine(dst, x1, x2, y2-t, col)
		vLine(dst, x1+t, y1, y2, col)
		vLine(dst, x2-t, y1, y2, col)
	}
}

// DrawLandmarks draws a filled dot at each point.
func DrawLandmarks(dst *image.RGBA, pts []detect.Point, col color.Color, radius int) {
	for _, p := range pts {
		DrawDot(dst, int(p.X), int(p.Y), col, radius)
	}
}

// DrawDot draws a filled square dot of the given radius centered at (cx,cy).
func DrawDot(dst *image.RGBA, cx, cy int, col color.Color, radius int) {
	if radius < 1 {
		radius = 1
	}
	for y := cy - radius; y <= cy+radius; y++ {
		for x := cx - radius; x <= cx+radius; x++ {
			setPixel(dst, x, y, col)
		}
	}
}

// DrawPolygon draws a closed polygon (used for oriented/rotated boxes).
func DrawPolygon(dst *image.RGBA, pts []detect.Point, col color.Color, thickness int) {
	n := len(pts)
	for i := 0; i < n; i++ {
		a, b := pts[i], pts[(i+1)%n]
		DrawLine(dst, int(a.X), int(a.Y), int(b.X), int(b.Y), col, thickness)
	}
}

// DrawSkeleton connects keypoint index pairs with lines (e.g. a pose skeleton).
func DrawSkeleton(dst *image.RGBA, pts []detect.Point, edges [][2]int, col color.Color, thickness int) {
	for _, e := range edges {
		if e[0] < len(pts) && e[1] < len(pts) {
			a, b := pts[e[0]], pts[e[1]]
			if (a.X != 0 || a.Y != 0) && (b.X != 0 || b.Y != 0) {
				DrawLine(dst, int(a.X), int(a.Y), int(b.X), int(b.Y), col, thickness)
			}
		}
	}
}

// DrawLine draws a line via Bresenham with a square brush of the given thickness.
func DrawLine(dst *image.RGBA, x0, y0, x1, y1 int, col color.Color, thickness int) {
	r := thickness / 2
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx, sy := sign(x1-x0), sign(y1-y0)
	err := dx + dy
	for {
		DrawDot(dst, x0, y0, col, r)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

// DrawMask alpha-blends a boolean mask (mask[y*w+x]) of the given width/height
// over dst, in the image's coordinate space, using a translucent colour.
func DrawMask(dst *image.RGBA, mask []bool, w, h int, col color.RGBA) {
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if mask[y*w+x] {
				blend(dst, x, y, col)
			}
		}
	}
}

// SavePNG encodes img as PNG at path, creating parent directories as needed.
func SavePNG(path string, img image.Image) error {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("mkdir %s: %w", dir, err)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func hLine(dst *image.RGBA, x1, x2, y int, col color.Color) {
	if x1 > x2 {
		x1, x2 = x2, x1
	}
	for x := x1; x <= x2; x++ {
		setPixel(dst, x, y, col)
	}
}

func vLine(dst *image.RGBA, x, y1, y2 int, col color.Color) {
	if y1 > y2 {
		y1, y2 = y2, y1
	}
	for y := y1; y <= y2; y++ {
		setPixel(dst, x, y, col)
	}
}

func setPixel(dst *image.RGBA, x, y int, col color.Color) {
	if image.Pt(x, y).In(dst.Bounds()) {
		dst.Set(x, y, col)
	}
}

// blend alpha-composites col (with its alpha) over the existing pixel.
func blend(dst *image.RGBA, x, y int, col color.RGBA) {
	if !image.Pt(x, y).In(dst.Bounds()) {
		return
	}
	a := float64(col.A) / 255
	o := dst.RGBAAt(x, y)
	mix := func(s, d uint8) uint8 { return uint8(float64(s)*a + float64(d)*(1-a)) }
	dst.SetRGBA(x, y, color.RGBA{mix(col.R, o.R), mix(col.G, o.G), mix(col.B, o.B), 255})
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
func sign(x int) int {
	if x > 0 {
		return 1
	}
	if x < 0 {
		return -1
	}
	return 0
}
func clampi(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
func clampf(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

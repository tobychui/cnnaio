package facerecognition

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"

	"cnnaio/mod/detect"
)

// refPoints is the canonical ArcFace/insightface 5-point template for a 112x112
// aligned face, in image coordinates: left eye, right eye, nose, left & right
// mouth corners ("left"/"right" = smaller/larger x in the image).
var refPoints = [5][2]float64{
	{38.2946, 51.6963},
	{73.5318, 51.5014},
	{56.0252, 71.7366},
	{41.5493, 92.3655},
	{70.7299, 92.2041},
}

// WFLW-98 landmark indices for the points we need (PFLD output order).
const (
	wflwLeftPupil  = 96
	wflwRightPupil = 97
	wflwNoseTip    = 54
	wflwMouthL     = 76
	wflwMouthR     = 82
)

// fivePoints picks the 5 ArcFace keypoints from the 98 PFLD landmarks and orders
// eyes / mouth corners by x so they line up with refPoints.
func fivePoints(pts []detect.Point) ([5][2]float64, bool) {
	if len(pts) < 98 {
		return [5][2]float64{}, false
	}
	le, re := pts[wflwLeftPupil], pts[wflwRightPupil]
	if le.X > re.X {
		le, re = re, le
	}
	ml, mr := pts[wflwMouthL], pts[wflwMouthR]
	if ml.X > mr.X {
		ml, mr = mr, ml
	}
	nose := pts[wflwNoseTip]
	return [5][2]float64{
		{float64(le.X), float64(le.Y)},
		{float64(re.X), float64(re.Y)},
		{float64(nose.X), float64(nose.Y)},
		{float64(ml.X), float64(ml.Y)},
		{float64(mr.X), float64(mr.Y)},
	}, true
}

// solveSimilarity fits a similarity transform mapping template (ref) coordinates
// to source-image coordinates from the 5 correspondences refPoints[i] -> src[i]:
//
//	x = a*u - b*v + tx
//	y = b*u + a*v + ty
//
// This is the inverse mapping a warp needs: for each output pixel (u,v) in the
// 112x112 template it gives the source pixel (x,y) to sample.
func solveSimilarity(src [5][2]float64) (a, b, tx, ty float64) {
	var m [4][4]float64
	var c [4]float64
	add := func(row [4]float64, target float64) {
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				m[i][j] += row[i] * row[j]
			}
			c[i] += row[i] * target
		}
	}
	for i := 0; i < 5; i++ {
		u, v := refPoints[i][0], refPoints[i][1]
		add([4]float64{u, -v, 1, 0}, src[i][0]) // x equation
		add([4]float64{v, u, 0, 1}, src[i][1])  // y equation
	}
	p := solve4(m, c)
	return p[0], p[1], p[2], p[3]
}

// solve4 solves the 4x4 system m*x = c via Gaussian elimination with partial pivoting.
func solve4(m [4][4]float64, c [4]float64) [4]float64 {
	const n = 4
	for col := 0; col < n; col++ {
		piv := col
		for r := col + 1; r < n; r++ {
			if math.Abs(m[r][col]) > math.Abs(m[piv][col]) {
				piv = r
			}
		}
		m[col], m[piv] = m[piv], m[col]
		c[col], c[piv] = c[piv], c[col]
		d := m[col][col]
		if d == 0 {
			continue
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := m[r][col] / d
			for k := col; k < n; k++ {
				m[r][k] -= f * m[col][k]
			}
			c[r] -= f * c[col]
		}
	}
	var x [4]float64
	for i := 0; i < n; i++ {
		if m[i][i] != 0 {
			x[i] = c[i] / m[i][i]
		}
	}
	return x
}

// warpFace produces a size×size aligned face by inverse-mapping each output pixel
// through the similarity transform and bilinearly sampling the source image.
func warpFace(src *image.RGBA, a, b, tx, ty float64, size int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	for v := 0; v < size; v++ {
		for u := 0; u < size; u++ {
			x := a*float64(u) - b*float64(v) + tx
			y := b*float64(u) + a*float64(v) + ty
			r, g, bl := bilinear(src, x, y)
			dst.SetRGBA(u, v, color.RGBA{r, g, bl, 255})
		}
	}
	return dst
}

// bilinear samples src at fractional (x,y) with edge clamping.
func bilinear(src *image.RGBA, x, y float64) (uint8, uint8, uint8) {
	b := src.Bounds()
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	fx := x - float64(x0)
	fy := y - float64(y0)

	at := func(px, py int) (float64, float64, float64) {
		if px < b.Min.X {
			px = b.Min.X
		}
		if px > b.Max.X-1 {
			px = b.Max.X - 1
		}
		if py < b.Min.Y {
			py = b.Min.Y
		}
		if py > b.Max.Y-1 {
			py = b.Max.Y - 1
		}
		c := src.RGBAAt(px, py)
		return float64(c.R), float64(c.G), float64(c.B)
	}

	r00, g00, b00 := at(x0, y0)
	r10, g10, b10 := at(x0+1, y0)
	r01, g01, b01 := at(x0, y0+1)
	r11, g11, b11 := at(x0+1, y0+1)

	lerp := func(a, b, t float64) float64 { return a + (b-a)*t }
	mix := func(c00, c10, c01, c11 float64) uint8 {
		top := lerp(c00, c10, fx)
		bot := lerp(c01, c11, fx)
		return uint8(lerp(top, bot, fy) + 0.5)
	}
	return mix(r00, r10, r01, r11), mix(g00, g10, g01, g11), mix(b00, b10, b01, b11)
}

// decodeRGBA decodes image bytes into a mutable *image.RGBA for sampling.
func decodeRGBA(data []byte) (*image.RGBA, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	if rgba, ok := img.(*image.RGBA); ok {
		return rgba, nil
	}
	b := img.Bounds()
	dst := image.NewRGBA(b)
	draw.Draw(dst, b, img, b.Min, draw.Src)
	return dst, nil
}

// encodePNG encodes an image as PNG bytes (to feed the aligned face to the wasm).
func encodePNG(img image.Image) ([]byte, error) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

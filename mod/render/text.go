package render

import (
	_ "embed"
	"image"
	"image/color"
	"image/draw"
	"sync"

	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

// PixelifySans-Regular, embedded so labels need no external font file at runtime.
//
//go:embed PixelifySans-Regular.ttf
var fontTTF []byte

var (
	parsedFont     *opentype.Font
	parseFontOnce  sync.Once
	parseFontErr   error
	faceCache      = map[int]font.Face{}
	faceCacheMutex sync.Mutex
)

func loadFont() (*opentype.Font, error) {
	parseFontOnce.Do(func() {
		parsedFont, parseFontErr = opentype.Parse(fontTTF)
	})
	return parsedFont, parseFontErr
}

// face returns a cached font.Face at the given pixel size (DPI 72 -> size == px).
func face(sizePx float64) (font.Face, error) {
	f, err := loadFont()
	if err != nil {
		return nil, err
	}
	key := int(sizePx + 0.5)
	if key < 1 {
		key = 1
	}
	faceCacheMutex.Lock()
	defer faceCacheMutex.Unlock()
	if fc, ok := faceCache[key]; ok {
		return fc, nil
	}
	fc, err := opentype.NewFace(f, &opentype.FaceOptions{
		Size:    float64(key),
		DPI:     72,
		Hinting: font.HintingFull,
	})
	if err != nil {
		return nil, err
	}
	faceCache[key] = fc
	return fc, nil
}

// DrawLabel draws text with a filled background, anchored so the label sits just
// above (x, y) — i.e. above a box whose top-left corner is (x, y). If there isn't
// room above, it drops the label just below y instead. Silently no-ops if the
// font can't be loaded.
func DrawLabel(dst *image.RGBA, x, y int, text string, textCol, bgCol color.Color, sizePx float64) {
	fc, err := face(sizePx)
	if err != nil || text == "" {
		return
	}

	m := fc.Metrics()
	ascent := m.Ascent.Ceil()
	lineH := ascent + m.Descent.Ceil()
	const padX = 3

	d := &font.Drawer{Dst: dst, Src: image.NewUniform(textCol), Face: fc}
	textW := d.MeasureString(text).Ceil()

	bgX1 := x
	bgY1 := y - lineH // above the box
	if bgY1 < dst.Bounds().Min.Y {
		bgY1 = y // no room above -> draw inside the top of the box
	}
	bgRect := image.Rect(bgX1, bgY1, bgX1+textW+2*padX, bgY1+lineH)
	draw.Draw(dst, bgRect, image.NewUniform(bgCol), image.Point{}, draw.Src)

	d.Dot = fixed.P(bgX1+padX, bgY1+ascent)
	d.DrawString(text)
}
